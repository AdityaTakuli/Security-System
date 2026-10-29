"""Fine-tune the YOLOv8n face detector with a DPU-friendly activation (Colab GPU).

SiLU is not a DPU operator. The deployed model was made in two fine-tunes,
each starting from the previous weights:

    python finetune.py --act hardswish --weights ../../best.pt         # -> best_hswish.pt
    python finetune.py --act relu      --weights best_hswish.pt        # -> best_relu.pt (deployed)

Hardswish compiles to 11 DPU subgraphs; ReLU compiles to 1, so ReLU is the one used.
After a Colab disconnect, resume with:
    python -c "from ultralytics import YOLO; YOLO('<project>/face_relu_416/weights/last.pt').train(resume=True)"
"""
import argparse
import os
import shutil

from ultralytics import YOLO

from dpu_head import activation_types

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"hardswish": "yolov8n-face-hswish.yaml", "relu": "yolov8n-face-relu.yaml"}

ap = argparse.ArgumentParser()
ap.add_argument("--act", choices=list(CFG), default="relu")
ap.add_argument("--weights", required=True, help="checkpoint to start from (best.pt or best_hswish.pt)")
ap.add_argument("--data", default="/content/work/dataset/data.yaml", help="WIDER FACE data.yaml (nc: 1, names: ['face'])")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--project", default="/content/drive/MyDrive/fpga_runs")
a = ap.parse_args()

name = "face_%s_%d" % ({"hardswish": "hswish"}.get(a.act, a.act), a.imgsz)
model = YOLO(os.path.join(HERE, CFG[a.act])).load(a.weights)   # all weight tensors transfer
print("Activations:", activation_types(model.model))

model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=2,
            close_mosaic=5, plots=False, exist_ok=True, project=a.project, name=name)

best = os.path.join(a.project, name, "weights", "best.pt")
out = os.path.join(a.project, "best_%s.pt" % ({"hardswish": "hswish"}.get(a.act, a.act)))
shutil.copy(best, out)
m = YOLO(out).val(data=a.data, imgsz=a.imgsz, batch=a.batch, plots=False)
print("Saved %s   mAP50=%.4f  mAP50-95=%.4f" % (out, m.box.map50, m.box.map))
