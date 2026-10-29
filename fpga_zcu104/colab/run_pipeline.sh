#!/bin/bash
# Quantise -> evaluate -> compile -> check, inside the Vitis AI 3.0 docker (/workspace).
# Needs in /workspace: yolov8n_plain.py, yolo_decode.py, quantize_vai.py, check_xmodel.py,
#   yolov8n_face_relu_plain.pth, lab_bundle/{calib,eval/images,eval/labels}
# From Colab:  vai_background("run_pipeline.sh")   (see vitis_ai_colab.py)
source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch
cd /workspace
export YOLO_ACT=relu YOLO_HEAD=feat        # ReLU model, final 1x1 head convs left to the CPU
NOISE='UNILOG|^I20|^W20|it/s|Find subgraph|node name|VAIQ_NOTE|^\s*$|^\s+\w+ --- '
Q="python -u quantize_vai.py --weights yolov8n_face_relu_plain.pth --out quantize_feat"

rm -rf quantize_feat compiled_ZCU104
$Q --mode calib --n-calib 200 2>&1 | grep --line-buffered -vE "$NOISE"
$Q --mode test  --n-eval 300  2>&1 | grep --line-buffered -vE "$NOISE"
vai_c_xir -x quantize_feat/YoloV8nFace_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  -o compiled_ZCU104 -n yolov8n_face_416 2>&1 | grep -iE 'subgraph number|compile done|error'
python check_xmodel.py compiled_ZCU104/yolov8n_face_416.xmodel
ls -la compiled_ZCU104
echo PIPELINE_FINISHED
