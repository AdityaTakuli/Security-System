"""Vitis AI 3.0 (vai_q_pytorch) INT8 quantization of the plain YOLOv8n face model.
Run inside the Vitis AI container from /workspace.
  --mode inspect : which ops go to DPU vs CPU
  --mode calib   : INT8 calibration on calibration images
  --mode test    : float vs INT8 accuracy, golden data, previews, export .xmodel"""
import argparse, glob, json, os, random, time
import cv2
import numpy as np
import sys
import torch

# yolo_decode.py lives in ../board (shared with the board); a flat copy in the docker workspace also works
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "board"))
from yolov8n_plain import load_plain
from yolo_decode import postprocess, preprocess_hwc

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["inspect", "calib", "test"], required=True)
ap.add_argument("--weights", default="yolov8n_face_relu_plain.pth")
ap.add_argument("--calib-dir", default="lab_bundle/calib")
ap.add_argument("--eval-dir", default="lab_bundle/eval")
ap.add_argument("--n-calib", type=int, default=500)
ap.add_argument("--n-eval", type=int, default=500)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
ap.add_argument("--out", default="quantize_result")
a = ap.parse_args()

from pytorch_nndct.apis import Inspector, torch_quantizer  # only exists inside Vitis AI

torch.set_num_threads(max(1, os.cpu_count() or 1))
dev = torch.device("cpu")
S = a.imgsz
net = load_plain(a.weights)
from yolov8n_plain import head_weights
HEAD = head_weights(net) if os.environ.get('YOLO_HEAD') == 'feat' else None
dummy = torch.rand(1, 3, S, S)
IOUV = np.linspace(0.5, 0.95, 10)


def to_tensor(imgs):
    return torch.from_numpy(np.stack([preprocess_hwc(im, S) for im in imgs])).permute(0, 3, 1, 2).contiguous()


def predict(model, img, conf=0.001, iou=0.7):
    with torch.no_grad():
        outs = model(to_tensor([img]))
    return postprocess([o.permute(0, 2, 3, 1).numpy() for o in outs], img.shape[:2], (S, S), conf, iou, head=HEAD)


def box_iou(b1, b2):
    tl = np.maximum(b1[:, None, :2], b2[None, :, :2])
    br = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    a1 = (b1[:, 2:] - b1[:, :2]).prod(1)
    a2 = (b2[:, 2:] - b2[:, :2]).prod(1)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-9)


def load_gt(path, w, h):
    rows = [l.split() for l in open(path).read().splitlines() if l.strip()] if os.path.exists(path) else []
    if not rows:
        return np.zeros((0, 4), np.float32)
    r = np.array(rows, np.float32)[:, 1:5]
    cx, cy, bw, bh = r[:, 0] * w, r[:, 1] * h, r[:, 2] * w, r[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def evaluate(fn, paths, label_dir, name):
    """mAP50 and mAP50-95 (101-point interpolation, like Ultralytics)."""
    scores, tps, n_gt, t0 = [], [], 0, time.time()
    for k, p in enumerate(paths):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gt = load_gt(os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + ".txt"), w, h)
        n_gt += len(gt)
        preds = fn(img)
        if (k + 1) % 100 == 0:
            print("  [%s] %d/%d images, %.1f s" % (name, k + 1, len(paths), time.time() - t0), flush=True)
        if not preds:
            continue
        P = np.array(preds, np.float32)
        P = P[np.argsort(-P[:, 4])]
        tp = np.zeros((len(P), len(IOUV)), bool)
        if len(gt):
            ious = box_iou(P[:, :4], gt)
            for t, thr in enumerate(IOUV):
                matched = np.zeros(len(gt), bool)
                for i in range(len(P)):
                    row = np.where(matched, -1.0, ious[i])
                    j = int(row.argmax())
                    if row[j] >= thr:
                        matched[j] = True
                        tp[i, t] = True
        scores.append(P[:, 4])
        tps.append(tp)
    s, tp = np.concatenate(scores), np.concatenate(tps)
    tp = tp[np.argsort(-s)]
    x, aps = np.linspace(0, 1, 101), []
    for t in range(len(IOUV)):
        tpc = np.cumsum(tp[:, t])
        fpc = np.cumsum(~tp[:, t])
        rec = tpc / max(n_gt, 1)
        prec = tpc / (tpc + fpc)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], prec, [0.0]]))))
        aps.append(np.trapz(np.interp(x, mrec, mpre), x))
    return float(aps[0]), float(np.mean(aps))


if a.mode == "inspect":
    Inspector(a.target).inspect(net, (dummy,), device=dev, output_dir=a.out + "_inspect")
    raise SystemExit

q = torch_quantizer(a.mode, net, (dummy,), output_dir=a.out, device=dev, target=a.target)
qnet = q.quant_model

if a.mode == "calib":
    paths = sorted(glob.glob(os.path.join(a.calib_dir, "*.jpg")))[: a.n_calib]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            qnet(to_tensor([cv2.imread(p) for p in paths[i:i + a.batch]]))
            print("  calibrated %d/%d (%.0f s)" % (min(i + a.batch, len(paths)), len(paths), time.time() - t0), flush=True)
    q.export_quant_config()
    print("CALIBRATION DONE ->", a.out)
else:  # test
    paths = sorted(glob.glob(os.path.join(a.eval_dir, "images", "*.jpg")))
    random.Random(0).shuffle(paths)
    paths = paths[: a.n_eval]
    lbl = os.path.join(a.eval_dir, "labels")
    f50, f5095 = evaluate(lambda im: predict(net, im), paths, lbl, "float")
    i50, i5095 = evaluate(lambda im: predict(qnet, im), paths, lbl, "int8")
    rep = {"images": len(paths), "float_mAP50": f50, "float_mAP50_95": f5095,
           "int8_mAP50": i50, "int8_mAP50_95": i5095, "drop_mAP50": f50 - i50}
    json.dump(rep, open(os.path.join(a.out, "accuracy_report.json"), "w"), indent=2)
    print("ACCURACY:", json.dumps(rep, indent=2))

    # golden reference: exact preprocessed input + INT8 outputs (NHWC) for the board check
    img = cv2.imread(paths[0])
    rgb = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        outs = qnet(torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous())
    gold = {"input_rgb_u8": rgb}
    for k, o in enumerate(outs):
        gold["out%d" % k] = o.permute(0, 2, 3, 1).numpy()
    np.savez(os.path.join(a.out, "golden.npz"), **gold)
    cv2.imwrite(os.path.join(a.out, "golden_image.jpg"), img)

    # previews drawn from the INT8 model
    os.makedirs(os.path.join(a.out, "preview"), exist_ok=True)
    for k, p in enumerate(paths[:6]):
        im = cv2.imread(p)
        for x1, y1, x2, y2, s in predict(qnet, im, conf=0.25, iou=0.45):
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(a.out, "preview", "%02d.jpg" % k), im)

    if HEAD is not None:
        np.savez(os.path.join(a.out, 'head_weights.npz'), **HEAD)
    q.export_xmodel(output_dir=a.out, deploy_check=False)
    print("EXPORTED:", sorted(os.listdir(a.out)))
