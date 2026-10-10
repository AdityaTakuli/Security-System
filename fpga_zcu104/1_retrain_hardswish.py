"""Fine-tune the face detector with Hardswish instead of SiLU (run on Colab GPU)."""
import argparse
from ultralytics import YOLO
from dpu_head import activation_types

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--weights", default="../best.pt")
ap.add_argument("--cfg", default="yolov8n-face-hswish.yaml")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

model = YOLO(args.cfg).load(args.weights)                 # expect "Transferred 355/355 items"
print("Activations in model:", activation_types(model.model))   # expect ['Hardswish']
model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
            name="face_hswish_416", close_mosaic=10)
m = model.val(data=args.data, imgsz=args.imgsz)
print(f"mAP50 = {m.box.map50:.4f}   mAP50-95 = {m.box.map:.4f}")
