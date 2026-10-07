"""
Step 3 - INT8-quantize the head-cut detector with Vitis AI 3.5 (run INSIDE the Vitis AI docker).

    # on an x86 Linux PC (Ubuntu 20.04/22.04), from a clone of Vitis-AI at tag v3.5:
    ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
    conda activate vitis-ai-pytorch
    # --no-deps matters: letting pip upgrade torch inside the docker breaks the quantizer
    pip install ultralytics==8.3.155 --no-deps && pip install py-cpuinfo ultralytics-thop --no-deps

    python 3_quantize_vitis_ai.py --mode inspect --weights best_hswish.pt        # what runs where
    python 3_quantize_vitis_ai.py --mode calib   --weights best_hswish.pt --calib-dir WIDER_val/images
    python 3_quantize_vitis_ai.py --mode test    --weights best_hswish.pt --calib-dir WIDER_val/images

Output: quantize_result/DPUHead_int.xmodel  (+ preview/*.jpg drawn from the INT8 model)

Calibration images must go through the SAME preprocessing as the board (plain resize,
RGB, /255) — that's why this imports preprocess_hwc from yolo_decode.py.
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch

from dpu_head import DPUHead, activation_types, load_detection_model
from yolo_decode import postprocess, preprocess_hwc

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["inspect", "calib", "test"], required=True)
ap.add_argument("--weights", default="best_hswish.pt")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--calib-dir", default="WIDER_val/images")
ap.add_argument("--n-calib", type=int, default=500, help="300-1000 images is plenty")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--target", default="DPUCZDX8G_ISA1_B4096", help="B4096 for ZCU104/ZCU102/KV260 default TRDs")
ap.add_argument("--out", default="quantize_result")
args = ap.parse_args()

from pytorch_nndct.apis import Inspector, torch_quantizer  # noqa: E402  (only exists in the docker)

device = torch.device("cpu")
model = load_detection_model(args.weights)
acts = activation_types(model)
print("Activations:", acts)
if "SiLU" in acts:
    print("WARNING: SiLU found - the DPU can't fuse it. Use the Hardswish checkpoint from step 1.")
net = DPUHead(model).eval()
dummy = torch.randn(1, 3, args.imgsz, args.imgsz)


def image_paths():
    paths = sorted(glob.glob(os.path.join(args.calib_dir, "**", "*.jpg"), recursive=True))
    if not paths:
        raise SystemExit(f"No .jpg under {args.calib_dir}")
    random.Random(0).shuffle(paths)
    return paths[: args.n_calib]


def to_tensor(paths):
    arr = np.stack([preprocess_hwc(cv2.imread(p), args.imgsz) for p in paths])   # N,H,W,3
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()                # N,3,H,W


if args.mode == "inspect":
    # Writes a report + graph image showing which ops land on DPU vs CPU. Goal: everything on DPU.
    Inspector(args.target).inspect(net, (dummy,), device=device, output_dir="inspect", image_format="png")
    print("See inspect/inspect_*.txt - every node should say DPU.")
    raise SystemExit

quantizer = torch_quantizer(args.mode, net, (dummy,), output_dir=args.out, device=device, target=args.target)
qnet = quantizer.quant_model
paths = image_paths()

with torch.no_grad():
    if args.mode == "calib":
        for i in range(0, len(paths), args.batch):
            qnet(to_tensor(paths[i:i + args.batch]))
            print(f"\rcalibrating {min(i + args.batch, len(paths))}/{len(paths)}", end="")
        print()
        quantizer.export_quant_config()
        print(f"Calibration saved in {args.out}/  ->  now run --mode test")
    else:
        # test mode: batch size must be 1. Also draw INT8 detections so you can eyeball accuracy.
        os.makedirs("preview", exist_ok=True)
        for k, p in enumerate(paths[:20]):
            frame = cv2.imread(p)
            outs = qnet(to_tensor([p]))
            nhwc = [o.permute(0, 2, 3, 1).numpy() for o in outs]
            for x1, y1, x2, y2, s in postprocess(nhwc, frame.shape[:2], (args.imgsz, args.imgsz)):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{s:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"preview/{k:02d}.jpg", frame)
        quantizer.export_xmodel(output_dir=args.out, deploy_check=False)
        print(f"Quantized model: {args.out}/DPUHead_int.xmodel   (INT8 previews in preview/)")
