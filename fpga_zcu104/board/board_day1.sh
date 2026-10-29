#!/bin/bash
# Day-1 checklist on the ZCU104. Run from inside this folder:  bash board_day1.sh
cd "$(dirname "$0")"
echo "===== 1. DPU hardware ====="
xdputil query
echo
echo "Model was compiled for DPU fingerprint 0x101000056010407 (DPUCZDX8G_ISA1_B4096)."
echo "It must match the 'fingerprint' printed above."
echo
echo "===== 2. Python packages ====="
python3 -c "import vart, xir, cv2, numpy; print('vart/xir OK | numpy', numpy.__version__, '| opencv', cv2.__version__)"
python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)" 2>/dev/null \
  || echo "onnxruntime not installed -> recogniser will use OpenCV-DNN (fine), or: pip3 install onnxruntime"
echo
echo "===== 3. Camera ====="
ls /dev/video* 2>/dev/null || echo "No camera found - plug in the USB camera"
echo
echo "===== 4. Golden check: real DPU vs quantizer prediction ====="
python3 golden_check.py --model yolov8n_face_416.xmodel --golden golden.npz --head head_weights.npz
echo
echo "===== 5. Start the app ====="
echo "NO_ORT=1 python3 dpu_face_app.py --model yolov8n_face_416.xmodel --head head_weights.npz --rec w600k_mbf.onnx --db face_embeddings.jsonl --image test_images/test1.png --save results/test1.jpg"
echo "or: ./run.sh  (all images in test_images/)   |   ./cam.sh  (live camera)"
