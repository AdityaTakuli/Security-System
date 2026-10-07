"""
Face detection + recognition on Zynq UltraScale+ (ZCU104 / ZCU102 / KV260).

    YOLOv8n face detector   -> DPU in the FPGA fabric, via VART          (--backend dpu)
    DFL decode + NMS        -> Cortex-A53, NumPy (yolo_decode.py)
    MobileFaceNet (512-d)   -> Cortex-A53, onnxruntime / OpenCV-DNN      (PReLU isn't a DPU op)

The recogniser and its preprocessing are unchanged from the original scripts, so the
face_embeddings.jsonl you registered on the laptop / Raspberry Pi still works, and the
board can stay a client of host_server.py exactly like server_connect.py.

The same file runs on a laptop or RPi with --backend onnx (the head-cut ONNX from step 2),
so everything except the DPU itself can be debugged before you touch the board.

On the board:
    python3 dpu_face_app.py --model yolov8n_face_416.xmodel --camera 0 --show
    python3 dpu_face_app.py --model yolov8n_face_416.xmodel --camera 0 --host 192.168.1.20
On a laptop:
    python dpu_face_app.py --backend onnx --model best_hswish_headcut_416.onnx --image group.jpg --save out.jpg
"""
import argparse
import json
import math
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

from yolo_decode import postprocess, preprocess_hwc


# ----------------------------------------------------------------------------- detectors
class DpuDetector:
    """Runs the compiled .xmodel on the DPU through VART (Vitis AI 3.5 runtime on the board)."""

    def __init__(self, xmodel):
        import vart
        import xir
        graph = xir.Graph.deserialize(xmodel)
        subs = [s for s in graph.get_root_subgraph().toposort_child_subgraph()
                if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if len(subs) != 1:
            print(f"[WARN] {len(subs)} DPU subgraphs in {xmodel} (expected 1) - using the first")
        self.runner = vart.Runner.create_runner(subs[0], "run")
        self.in_t = self.runner.get_input_tensors()[0]
        self.out_t = self.runner.get_output_tensors()
        _, h, w, _ = tuple(self.in_t.dims)                       # NHWC
        self.input_hw = (h, w)
        self.in_scale = np.float32(2.0 ** self.in_t.get_attr("fix_point") / 255.0)  # /255 folded in
        self.out_scales = [np.float32(2.0 ** -t.get_attr("fix_point")) for t in self.out_t]
        self.in_buf = np.empty(tuple(self.in_t.dims), dtype=np.int8, order="C")
        self.out_bufs = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in self.out_t]
        print(f"[DPU] input {tuple(self.in_t.dims)}, outputs {[tuple(t.dims) for t in self.out_t]}")

    def infer(self, frame_bgr):
        h, w = self.input_hw
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
        self.in_buf[0] = np.clip(np.rint(rgb * self.in_scale), -128, 127)
        job = self.runner.execute_async([self.in_buf], self.out_bufs)
        self.runner.wait(job)
        return [b[:1].astype(np.float32) * s for b, s in zip(self.out_bufs, self.out_scales)]


class OnnxHeadDetector:
    """CPU stand-in for the DPU: same head-cut graph, float32, via onnxruntime."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))

    def infer(self, frame_bgr):
        x = preprocess_hwc(frame_bgr, self.input_hw[0]).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.name: np.ascontiguousarray(x)})
        return [o.transpose(0, 2, 3, 1) for o in outs]          # NCHW -> NHWC, like VART


# ----------------------------------------------------------------------------- recogniser
class FaceEmbedder:
    """MobileFaceNet (w600k_mbf.onnx) on the ARM cores. Batches all faces of a frame in one call."""

    def __init__(self, path):
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3          # model declares output [1,512]; batching faces is fine
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            self._run = lambda x: sess.run(None, {name: x})[0]
            self.engine = "onnxruntime"
        except ImportError:                                      # fallback if no onnxruntime wheel
            net = cv2.dnn.readNetFromONNX(path)

            def run(x):
                net.setInput(x)
                return net.forward()
            self._run = run
            self.engine = "opencv-dnn"

    @staticmethod
    def _prep(face_bgr):                                         # identical to preprocess_face()
        f = cv2.cvtColor(cv2.resize(face_bgr, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((f / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)

    def embed(self, crops):
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = np.ascontiguousarray(np.stack([self._prep(c) for c in crops]))
        e = self._run(x).astype(np.float32).reshape(len(crops), -1)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------- face database
class FaceDB:
    """{name: 512-d embedding}, matched with one matrix multiply instead of a Python loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.db = {}
        self._names, self._mat = [], np.zeros((0, 512), np.float32)

    def _rebuild(self):
        self._names = list(self.db)
        mat = np.array([self.db[n] for n in self._names], np.float32).reshape(-1, 512)
        self._mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    def set_all(self, d):
        with self.lock:
            self.db = {k: np.asarray(v, np.float32) for k, v in d.items()}
            self._rebuild()

    def upsert(self, name, emb):
        with self.lock:
            self.db[name] = np.asarray(emb, np.float32)
            self._rebuild()

    def load_jsonl(self, path):
        d = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if o.get("name") and isinstance(o.get("embedding"), list):
                        d[o["name"]] = o["embedding"]
        self.set_all(d)
        print(f"[DB] {len(d)} identities from {path}")

    def sync_from_host(self, host, port=8000):
        """Same protocol as server_connect.py: GET /db once, then live updates over /updates."""
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/db", timeout=3) as r:
                self.set_all(json.loads(r.read()))
            print(f"[DB] {len(self.db)} identities from host {host}")
        except Exception as e:
            print(f"[DB] bootstrap from host failed: {e}")
        try:
            from websocket import WebSocketApp                  # pip install websocket-client
        except ImportError:
            print("[DB] websocket-client not installed - no live updates")
            return

        def on_msg(_ws, message):
            msg = json.loads(message)
            if msg.get("type") == "full":
                self.set_all(msg.get("db", {}))
            elif msg.get("type") == "add":
                rec = msg.get("record", {})
                if rec.get("name") and isinstance(rec.get("embedding"), list):
                    self.upsert(rec["name"], rec["embedding"])
                    print(f"[WS] updated {rec['name']}")

        def runner():
            while True:
                WebSocketApp(f"ws://{host}:{port}/updates", on_message=on_msg).run_forever()
                time.sleep(2)
        threading.Thread(target=runner, daemon=True).start()

    def match(self, embs, threshold=0.5):
        with self.lock:
            names, mat = self._names, self._mat
        if len(embs) == 0 or not names:
            return ["Unknown"] * len(embs)
        sims = embs @ mat.T                                      # cosine similarity (all unit vectors)
        best = sims.argmax(axis=1)
        return [names[b] if sims[i, b] >= threshold else "Unknown" for i, b in enumerate(best)]


# ----------------------------------------------------------------------------- drawing
def box_gap(a, b):
    """Shortest distance between two boxes and the two points that realise it (from server_connect.py)."""
    clamp = lambda c, lo, hi: max(lo, min(c, hi))  # noqa: E731
    cax, cay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cbx, cby = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    p = (clamp(cax, b[0], b[2]), clamp(cay, b[1], b[3]))
    q = (clamp(cbx, a[0], a[2]), clamp(cby, a[1], a[3]))
    return math.hypot(p[0] - q[0], p[1] - q[1]), p, q


def draw(frame, boxes, names):
    unk = 0
    for (x1, y1, x2, y2, _s), name in zip(boxes, names):
        if name == "Unknown":
            unk += 1
            name = f"Unknown{unk}"
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dpu", "onnx"], default="dpu")
    ap.add_argument("--model", default="yolov8n_face_416.xmodel", help=".xmodel (dpu) or head-cut .onnx (onnx)")
    ap.add_argument("--rec", default="w600k_mbf.onnx")
    ap.add_argument("--db", default="face_embeddings.jsonl")
    ap.add_argument("--host", default=None, help="IP of the PC running host_server.py")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--camera", default="0", help="V4L2 index or a GStreamer pipeline")
    ap.add_argument("--image", default=None, help="run once on an image instead of a camera")
    ap.add_argument("--save", default=None, help="write the annotated frame here (image mode)")
    ap.add_argument("--show", action="store_true", help="cv2.imshow window (needs a display)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threshold", type=float, default=0.5, help="cosine similarity to accept a match")
    args = ap.parse_args()

    det = DpuDetector(args.model) if args.backend == "dpu" else OnnxHeadDetector(args.model)
    emb = FaceEmbedder(args.rec)
    print(f"[INFO] detector: {args.backend} {det.input_hw}, recogniser: {emb.engine}")
    db = FaceDB()
    if os.path.exists(args.db):
        db.load_jsonl(args.db)
    if args.host:
        db.sync_from_host(args.host, args.port)

    cap = None
    if args.image is None:
        src = int(args.camera) if args.camera.isdigit() else args.camera
        cap = cv2.VideoCapture(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")

    acc, n = np.zeros(4), 0
    try:
        while True:
            if cap is None:
                frame = cv2.imread(args.image)
                if frame is None:
                    raise SystemExit(f"Could not read {args.image}")
            else:
                ok, frame = cap.read()
                if not ok:
                    continue
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            outs = det.infer(frame)
            t1 = time.perf_counter()
            boxes = postprocess(outs, frame.shape[:2], det.input_hw, args.conf, args.iou)
            t2 = time.perf_counter()
            crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _ in boxes]
            keep = [i for i, c in enumerate(crops) if c.size]
            boxes, crops = [boxes[i] for i in keep], [crops[i] for i in keep]
            names = db.match(emb.embed(crops), args.threshold)
            t3 = time.perf_counter()
            draw(frame, boxes, names)

            acc += (t1 - t0, t2 - t1, t3 - t2, t3 - t0)
            n += 1
            if cap is None or n % 30 == 0:
                ms = acc / n * 1000
                print(f"faces={len(boxes)} {names} | detect {ms[0]:.1f} ms, decode+NMS {ms[1]:.1f} ms, "
                      f"recognise {ms[2]:.1f} ms, total {ms[3]:.1f} ms (~{1000 / ms[3]:.1f} FPS)")
                acc[:], n = 0, 0
            if cap is None:
                if args.save:
                    cv2.imwrite(args.save, frame)
                    print(f"[INFO] saved {args.save}")
                print(json.dumps([{"box": b[:4], "score": round(b[4], 3), "name": nm}
                                  for b, nm in zip(boxes, names)]))
                break
            if args.show:
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
