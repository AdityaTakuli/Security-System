"""
Step 1 - Fine-tune the face detector with Hardswish instead of SiLU (run on Colab / any GPU).

Starts from the existing SiLU-trained best.pt (all weights transfer; only the activation
changes), so this is a short fine-tune, not training from scratch. Trains at 416 because
that's the fixed input size the DPU model will be compiled for.

Prereq: the WIDER FACE split + dataset/data.yaml made by Yolov8n_training.py.

    pip install ultralytics==8.3.155
    python 1_retrain_hardswish.py --data dataset/data.yaml --weights ../best.pt
"""
import argparse

from ultralytics import YOLO

from dpu_head import activation_types

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--weights", default="../best.pt", help="SiLU-trained checkpoint to start from")
ap.add_argument("--cfg", default="yolov8n-face-hswish.yaml")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

model = YOLO(args.cfg).load(args.weights)          # prints "Transferred 355/355 items"
print("Activations in model:", activation_types(model.model))   # expect ['Hardswish']

model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
            name="face_hswish_416", close_mosaic=10)
metrics = model.val(data=args.data, imgsz=args.imgsz)
print(f"mAP50 = {metrics.box.map50:.4f}   mAP50-95 = {metrics.box.map:.4f}")
print("Weights: runs/detect/face_hswish_416/weights/best.pt  -> copy next to these scripts as best_hswish.pt")
