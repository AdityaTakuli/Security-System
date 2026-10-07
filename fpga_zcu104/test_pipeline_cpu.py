"""
Sanity checks for the FPGA port that run on any PC (no board, no Vitis AI needed).

  1. The head-cut model + NumPy decode reproduces Ultralytics' own decoded boxes exactly.
  2. NumPy NMS gives the same detections as Ultralytics' NMS.
  3. The new pipeline finds the same faces as the repo's original best.onnx.

    python test_pipeline_cpu.py --weights ../best.pt --onnx ../best.onnx img1.jpg img2.jpg
"""
import argparse

import numpy as np
import onnxruntime as ort
import torch
import cv2

from dpu_head import DPUHead, load_detection_model
from yolo_decode import decode, nms, postprocess, preprocess_hwc, split_outputs


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="../best.pt")
ap.add_argument("--onnx", default="../best.onnx", help="original end-to-end ONNX (NMS inside)")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--iou", type=float, default=0.45)
ap.add_argument("images", nargs="+")
args = ap.parse_args()
S = args.imgsz

from ultralytics.utils.ops import non_max_suppression  # noqa: E402

model = load_detection_model(args.weights)
head = DPUHead(model).eval()
orig = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
oname = orig.get_inputs()[0].name
ok = True

for path in args.images:
    frame = cv2.imread(path)
    x = torch.from_numpy(preprocess_hwc(frame, S)).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        y_ref, _ = model(x)                                  # Ultralytics decode: [1, 4+nc, N], xywh
        raw = head(x)
    nhwc = [r.permute(0, 2, 3, 1).numpy() for r in raw]

    # 1. full decode, every anchor kept
    b, s, _ = decode(split_outputs(nhwc), (S, S), conf_thres=0)
    ref = y_ref[0].numpy()
    ref_xyxy = np.stack([ref[0] - ref[2] / 2, ref[1] - ref[3] / 2, ref[0] + ref[2] / 2, ref[1] + ref[3] / 2], 1)
    d_box = np.abs(b - ref_xyxy).max()
    d_score = np.abs(s - ref[4]).max()

    # 2. NMS vs Ultralytics
    bb, ss, _ = decode(split_outputs(nhwc), (S, S), args.conf)
    keep = nms(bb, ss, args.iou)
    mine = np.c_[bb[keep], ss[keep]]
    ul = non_max_suppression(y_ref, args.conf, args.iou)[0].numpy()
    nms_match = len(mine) == len(ul) and np.allclose(mine, ul[:, :5], atol=1e-2)

    # 3. vs original best.onnx (same 416 plain-resize input the current scripts feed it)
    o = orig.run(None, {oname: x.numpy()})[0][0]
    o = o[o[:, 4] >= args.conf]
    oh, ow = frame.shape[:2]
    old = [(r[0] * ow / S, r[1] * oh / S, r[2] * ow / S, r[3] * oh / S) for r in o]
    new = postprocess(nhwc, frame.shape[:2], (S, S), args.conf, args.iou)
    matched = sum(any(iou(n[:4], q) > 0.9 for q in old) for n in new)

    print(f"{path}: anchors={len(b)}  max|box diff|={d_box:.2e}px  max|score diff|={d_score:.2e}  "
          f"NMS {len(mine)} vs ultralytics {len(ul)} -> {'match' if nms_match else 'MISMATCH'}  "
          f"| new pipeline {len(new)} faces, original best.onnx {len(old)}, matched {matched}")
    ok &= d_box < 1e-2 and d_score < 1e-4 and nms_match and matched == len(new) == len(old)

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
