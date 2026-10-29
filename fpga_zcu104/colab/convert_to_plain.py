"""Convert the Ultralytics ReLU checkpoint to the plain-PyTorch model and export the CPU head.

Runs in Colab (needs ultralytics). Produces, next to --out-dir:
  yolov8n_face_relu_plain.pth   weights for yolov8n_plain.py (loaded by quantize_vai.py)
  head_weights.npz              float32 final 1x1 convs, run on the board's CPU

Checks printed:
  1. plain model vs Ultralytics: max difference must be 0 (or ~1e-5)
  2. feature-output model + CPU head vs full model: same faces on test images (0 mismatches)

    python convert_to_plain.py --weights best_relu.pt --images lab_bundle/eval/images
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("YOLO_ACT", "relu")          # must be set before yolov8n_plain is imported
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "board"))   # yolo_decode.py

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import yolo_decode  # noqa: E402
import yolov8n_plain  # noqa: E402
from dpu_head import DPUHead, activation_types, load_detection_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="best_relu.pt", help="Ultralytics ReLU checkpoint")
ap.add_argument("--images", default="lab_bundle/eval/images", help="a few images for the float check")
ap.add_argument("--out-dir", default=".")
ap.add_argument("--imgsz", type=int, default=416)
a = ap.parse_args()
S = a.imgsz

# 1) Ultralytics -> plain PyTorch (C2f cv1 split into cv1a/cv1b; DFL dropped, it runs on the CPU)
m = load_detection_model(a.weights)
print("Ultralytics activations:", activation_types(m))
sd = {k: v for k, v in m.state_dict().items() if ".dfl." not in k}
net = yolov8n_plain.YoloV8nFace("full")
net.load_state_dict(yolov8n_plain.convert_state_dict(sd), strict=True)
net.eval()
ref = DPUHead(m).eval()
with torch.no_grad():
    x = torch.rand(2, 3, S, S)
    worst = max((p - q).abs().max().item() for p, q in zip(net(x), ref(x)))
print("max difference vs Ultralytics: %.2e" % worst)
pth = os.path.join(a.out_dir, "yolov8n_face_relu_plain.pth")
torch.save(net.state_dict(), pth)

# 2) feature-output model + float CPU head must find the same faces as the full model
full = yolov8n_plain.load_plain(pth, "full")
feat = yolov8n_plain.load_plain(pth, "feat")
head = yolov8n_plain.head_weights(feat)
np.savez(os.path.join(a.out_dir, "head_weights.npz"), **head)
mism, worst_px = 0, 0.0
paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[:15]
for p in paths:
    img = cv2.imread(p)
    x = torch.from_numpy(yolo_decode.preprocess_hwc(img, S)).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        fa = [o.permute(0, 2, 3, 1).numpy() for o in full(x)]
        fb = [o.permute(0, 2, 3, 1).numpy() for o in feat(x)]
    ra = yolo_decode.postprocess(fa, img.shape[:2], (S, S))
    rb = yolo_decode.postprocess(fb, img.shape[:2], (S, S), head=head)
    if len(ra) != len(rb):
        mism += 1
        continue
    if ra:
        worst_px = max(worst_px, float(np.abs(np.array(ra) - np.array(rb)).max()))
print("%d images: face-count mismatches=%d, worst box diff=%.3f px (expect 0 and <=1)" % (len(paths), mism, worst_px))
print("Saved:", pth, "and", os.path.join(a.out_dir, "head_weights.npz"))
