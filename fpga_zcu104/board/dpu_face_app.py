"""Face detection (YOLOv8n) + recognition (MobileFaceNet) for ZCU104 / ZCU102 / KV260.
  --backend dpu  : detector on the FPGA DPU via VART (on the board)
  --backend onnx : same head-cut model on CPU via onnxruntime (laptop testing)
Recogniser + preprocessing are unchanged, so existing face_embeddings.jsonl stays valid."""
import argparse, json, math, os, threading, time, urllib.request
import cv2
import numpy as np
from yolo_decode import postprocess, preprocess_hwc


class DpuDetector:
    """Runs the compiled .xmodel on the DPU through VART (Vitis AI 3.0 runtime on the board)."""
    def __init__(self, xmodel):
        import vart, xir
        g = self.graph = xir.Graph.deserialize(xmodel)   # keep alive: the runner points into it
        subs = [s for s in g.get_root_subgraph().toposort_child_subgraph()
                if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if len(subs) != 1:
            print(f"[WARN] {len(subs)} DPU subgraphs (expected 1) - using the first")
        self.runner = vart.Runner.create_runner(subs[0], "run")
        self.in_t = self.runner.get_input_tensors()[0]
        self.out_t = self.runner.get_output_tensors()
        _, h, w, _ = tuple(self.in_t.dims)
        self.input_hw = (h, w)
        self.in_scale = np.float32(2.0 ** self.in_t.get_attr("fix_point") / 255.0)
        self.out_scales = [np.float32(2.0 ** -t.get_attr("fix_point")) for t in self.out_t]
        self.in_buf = np.empty(tuple(self.in_t.dims), dtype=np.int8, order="C")
        self.out_bufs = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in self.out_t]
        print(f"[DPU] input {tuple(self.in_t.dims)}, outputs {[tuple(t.dims) for t in self.out_t]}")

    def infer(self, frame_bgr):
        h, w = self.input_hw
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
        self.in_buf[0] = np.clip(np.rint(rgb * self.in_scale), -128, 127)
        self.runner.wait(self.runner.execute_async([self.in_buf], self.out_bufs))
        return [b[:1].astype(np.float32) * s for b, s in zip(self.out_bufs, self.out_scales)]


class OnnxHeadDetector:
    """CPU stand-in for the DPU: same head-cut graph via onnxruntime."""
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name, self.input_hw = inp.name, (int(inp.shape[2]), int(inp.shape[3]))

    def infer(self, frame_bgr):
        x = preprocess_hwc(frame_bgr, self.input_hw[0]).transpose(2, 0, 1)[None]
        return [o.transpose(0, 2, 3, 1) for o in self.sess.run(None, {self.name: np.ascontiguousarray(x)})]


class FaceEmbedder:
    """MobileFaceNet on the CPU; all faces of a frame in one batch."""
    def __init__(self, path):
        try:
            if os.environ.get("NO_ORT"):
                raise ImportError("onnxruntime disabled")   # it aborts on import on the ZCU104 image
            import onnxruntime as ort
            so = ort.SessionOptions(); so.log_severity_level = 3
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            self._run, self.engine = (lambda x: sess.run(None, {name: x})[0]), "onnxruntime"
        except ImportError:
            net = cv2.dnn.readNetFromONNX(path)
            def run(x):
                net.setInput(x); return net.forward()
            self._run, self.engine = run, "opencv-dnn"

    @staticmethod
    def _prep(face):
        f = cv2.cvtColor(cv2.resize(face, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((f / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def embed(self, crops):
        if not crops:
            return np.zeros((0, 512), np.float32)
        e = self._run(np.ascontiguousarray(np.stack([self._prep(c) for c in crops]))).astype(np.float32)
        e = e.reshape(len(crops), -1)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class FaceDB:
    """{name: embedding}; matching is one matrix multiply."""
    def __init__(self):
        self.lock, self.db = threading.Lock(), {}
        self._names, self._mat = [], np.zeros((0, 512), np.float32)

    def _rebuild(self):
        self._names = list(self.db)
        m = np.array([self.db[n] for n in self._names], np.float32).reshape(-1, 512)
        self._mat = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    def set_all(self, d):
        with self.lock:
            self.db = {k: np.asarray(v, np.float32) for k, v in d.items()}; self._rebuild()

    def upsert(self, name, emb):
        with self.lock:
            self.db[name] = np.asarray(emb, np.float32); self._rebuild()

    def load_jsonl(self, path):
        d = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if o.get("name") and isinstance(o.get("embedding"), list):
                        d[o["name"]] = o["embedding"]
        self.set_all(d); print(f"[DB] {len(d)} identities from {path}")

    def sync_from_host(self, host, port=8000):
        """Same protocol as server_connect.py: GET /db, then live updates on /updates."""
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/db", timeout=3) as r:
                self.set_all(json.loads(r.read()))
            print(f"[DB] {len(self.db)} identities from host {host}")
        except Exception as e:
            print(f"[DB] bootstrap from host failed: {e}")
        try:
            from websocket import WebSocketApp
        except ImportError:
            print("[DB] websocket-client not installed - no live updates"); return
        def on_msg(_ws, message):
            msg = json.loads(message)
            if msg.get("type") == "full":
                self.set_all(msg.get("db", {}))
            elif msg.get("type") == "add":
                r = msg.get("record", {})
                if r.get("name") and isinstance(r.get("embedding"), list):
                    self.upsert(r["name"], r["embedding"]); print(f"[WS] updated {r['name']}")
        def runner():
            while True:
                WebSocketApp(f"ws://{host}:{port}/updates", on_message=on_msg).run_forever(); time.sleep(2)
        threading.Thread(target=runner, daemon=True).start()

    def match(self, embs, threshold=0.5):
        with self.lock:
            names, mat = self._names, self._mat
        if len(embs) == 0 or not names:
            return ["Unknown"] * len(embs)
        sims = embs @ mat.T
        best = sims.argmax(axis=1)
        return [names[b] if sims[i, b] >= threshold else "Unknown" for i, b in enumerate(best)]


def box_gap(a, b):
    clamp = lambda c, lo, hi: max(lo, min(c, hi))
    p = (clamp((a[0] + a[2]) / 2, b[0], b[2]), clamp((a[1] + a[3]) / 2, b[1], b[3]))
    q = (clamp((b[0] + b[2]) / 2, a[0], a[2]), clamp((b[1] + b[3]) / 2, a[1], a[3]))
    return math.hypot(p[0] - q[0], p[1] - q[1]), p, q


def draw(frame, boxes, names):
    unk = 0
    for (x1, y1, x2, y2, _), name in zip(boxes, names):
        if name == "Unknown":
            unk += 1; name = f"Unknown{unk}"
        color = (0, 0, 255) if name.startswith("Unknown") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(12, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            d, p, q = box_gap(boxes[i][:4], boxes[j][:4])
            p, q = tuple(map(int, p)), tuple(map(int, q))
            cv2.line(frame, p, q, (255, 0, 0), 2)
            cv2.putText(frame, f"{d:.1f}px", ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dpu", "onnx"], default="dpu")
    ap.add_argument("--model", default="yolov8n_face_416.xmodel")
    ap.add_argument("--rec", default="w600k_mbf.onnx")
    ap.add_argument("--head", default="head_weights.npz", help="CPU head weights (feat-head models)")
    ap.add_argument("--db", default="face_embeddings.jsonl")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--camera", default="0")
    ap.add_argument("--image", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    det = DpuDetector(a.model) if a.backend == "dpu" else OnnxHeadDetector(a.model)
    emb = FaceEmbedder(a.rec)
    HEAD = dict(np.load(a.head)) if a.head and os.path.exists(a.head) else None
    print("[INFO] CPU head:", "on" if HEAD else "off", a.head)
    print(f"[INFO] detector: {a.backend} {det.input_hw}, recogniser: {emb.engine}")
    db = FaceDB()
    if os.path.exists(a.db):
        db.load_jsonl(a.db)
    if a.host:
        db.sync_from_host(a.host, a.port)

    cap = None
    if a.image is None:
        src = int(a.camera) if a.camera.isdigit() else a.camera
        cap = cv2.VideoCapture(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {a.camera}")

    acc, n = np.zeros(4), 0
    try:
        while True:
            if cap is None:
                frame = cv2.imread(a.image)
                if frame is None:
                    raise SystemExit(f"Could not read {a.image}")
            else:
                ok, frame = cap.read()
                if not ok:
                    continue
                if not a.no_mirror:
                    frame = cv2.flip(frame, 1)
            t0 = time.perf_counter(); outs = det.infer(frame)
            t1 = time.perf_counter(); boxes = postprocess(outs, frame.shape[:2], det.input_hw, a.conf, a.iou, head=HEAD)
            t2 = time.perf_counter()
            crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _ in boxes]
            keep = [i for i, c in enumerate(crops) if c.size]
            boxes, crops = [boxes[i] for i in keep], [crops[i] for i in keep]
            names = db.match(emb.embed(crops), a.threshold)
            t3 = time.perf_counter()
            draw(frame, boxes, names)
            acc += (t1 - t0, t2 - t1, t3 - t2, t3 - t0); n += 1
            if cap is None or n % 30 == 0:
                ms = acc / n * 1000
                print(f"faces={len(boxes)} {names} | detect {ms[0]:.1f} ms, decode+NMS {ms[1]:.1f} ms, "
                      f"recognise {ms[2]:.1f} ms, total {ms[3]:.1f} ms (~{1000 / ms[3]:.1f} FPS)")
                acc[:], n = 0, 0
            if cap is None:
                if a.save:
                    cv2.imwrite(a.save, frame); print(f"[INFO] saved {a.save}")
                break
            if a.show:
                cv2.imshow("Face Recognition (FPGA DPU)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
