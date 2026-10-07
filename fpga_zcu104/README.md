# Security-System on FPGA: ZCU104 (Zynq UltraScale+ MPSoC) port

Runs the YOLOv8n face detector on the **DPU** (AMD's INT8 CNN accelerator, built in the FPGA fabric)
and keeps everything else on the board's quad Cortex-A53. Same flow works on **ZCU102** and **Kria KV260**:
only the `arch.json` in step 4 and the SD image change.

```
USB3 camera ─► A53: resize/RGB/INT8 ─► DPU (PL): YOLOv8n-Hardswish, 416x416, INT8 ─► 6 raw head maps
           ─► A53: DFL decode + NMS (yolo_decode.py) ─► crops ─► A53: MobileFaceNet (onnxruntime)
           ─► cosine match against face_embeddings.jsonl / host_server.py ─► draw / stream
```

## Why the model has to change (findings from this repo)

| Problem in the current model | Why the DPU cares | Fix in this kit |
|---|---|---|
| `best.onnx` has NMS + dynamic shapes baked in (`NonZero`, `NonMaxSuppression`, `ScatterND`, `Shape`...) | DPU can't run these; graph would split into many CPU pieces | Cut the head before DFL (`dpu_head.py`), decode + NMS in NumPy (`yolo_decode.py`) |
| YOLOv8n uses **SiLU** = 57 `Sigmoid` nodes | DPUCZDX8G only fuses ReLU / ReLU6 / LeakyReLU / Hard-Swish / Hard-Sigmoid | Fine-tune with **Hardswish** (`yolov8n-face-hswish.yaml`, step 1) |
| Recogniser `w600k_mbf.onnx` (MobileFaceNet) uses **PReLU** x34 | Not a DPUCZDX8G op | Phase 1: keep it on the A53 (unchanged, so old registrations still match). Phase 2: see bottom |
| Scripts feed 416x416 but README says 640 | DPU models have one fixed input size | Compile at **416** (2.4x less compute than 640) |

`test_pipeline_cpu.py` proves the new decode/NMS is equivalent: vs Ultralytics' own decode the max box
error is < 0.0001 px, NMS output is identical, and it finds the same faces as the original `best.onnx`.

---

## What you need

- ZCU104 kit (comes with a 1080p USB3 camera), 16-32 GB microSD, Ethernet cable, micro-USB for the UART console
- x86-64 Linux PC (Ubuntu 20.04/22.04) with Docker and ~60 GB free disk for the Vitis AI image
- GPU (Colab is fine) for the fine-tune in step 1, plus your WIDER FACE split from `Yolov8n_training.py`
- Tool versions are **frozen**: Vitis AI **3.5** + Vivado/PetaLinux **2023.1**. Vitis AI 5.x only targets Versal, not Zynq UltraScale+.

---

## Step 1 - Fine-tune the detector with Hardswish (Colab / GPU)

```bash
pip install ultralytics==8.3.155          # same version that trained best.pt
python 1_retrain_hardswish.py --data dataset/data.yaml --weights ../best.pt --imgsz 416 --epochs 40
cp runs/detect/face_hswish_416/weights/best.pt best_hswish.pt
```
All 355 weight tensors transfer from the SiLU model, so this is a fine-tune, not training from scratch.
Without it the swapped model produces extra false positives (checked on test images). Compare mAP50 with your
original model; Hardswish usually lands within a point or two.

## Step 2 - Export the head-cut model and test the whole pipeline on your laptop

```bash
python 2_export_headcut_onnx.py --weights best_hswish.pt --imgsz 416
#  -> best_hswish_headcut_416.onnx, prints the op list; must end with "OK: only DPU-friendly ops left"

python test_pipeline_cpu.py --weights best_hswish.pt --onnx ../best.onnx some_group_photo.jpg
python dpu_face_app.py --backend onnx --model best_hswish_headcut_416.onnx \
       --rec ../w600k_mbf.onnx --camera 0 --show            # webcam, same code the board runs
```
Fix anything here first. On the board, the only thing that changes is `--backend dpu`.

## Step 3 - Quantize to INT8 (Vitis AI 3.5 docker, on the x86 PC)

```bash
git clone -b v3.5 https://github.com/Xilinx/Vitis-AI.git && cd Vitis-AI
docker pull xilinx/vitis-ai-pytorch-cpu:latest
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
# inside the container:
conda activate vitis-ai-pytorch
pip install ultralytics==8.3.155 --no-deps && pip install py-cpuinfo ultralytics-thop --no-deps
cd /workspace/<path-to>/fpga_zcu104          # put best_hswish.pt + WIDER FACE val images here

python 3_quantize_vitis_ai.py --mode inspect --weights best_hswish.pt                 # every node -> DPU?
python 3_quantize_vitis_ai.py --mode calib   --weights best_hswish.pt --calib-dir WIDER_val/images
python 3_quantize_vitis_ai.py --mode test    --weights best_hswish.pt --calib-dir WIDER_val/images
```
Output: `quantize_result/DPUHead_int.xmodel` plus `preview/*.jpg` drawn by the INT8 model. Look at them.
If INT8 accuracy drops noticeably: more calibration images (up to ~1000), then vai_q_pytorch fast-finetune or QAT.

## Step 4 - Compile for the board's DPU (same docker)

```bash
bash 4_compile.sh ZCU104        # or ZCU102 / KV260
```
The log should say **`DPU subgraph number 1`**. More than 1 = something fell off the DPU; go back to `--mode inspect`.

## Step 5 - Prepare the ZCU104

1. Download the prebuilt **Vitis AI 3.5 SD-card image for ZCU104** (linked from the MPSoC "Target Setup" /
   Quick Start page of the Vitis AI 3.5 docs) and flash it with balenaEtcher or `dd`.
2. Set the boot-mode switch **SW6** to SD boot (table in ZCU104 user guide UG1267), plug in UART + Ethernet,
   power on. Serial console: 115200 8N1. The Vitis AI images normally log in as `root` / `root`.
3. Check the DPU is there and note its fingerprint:
   ```bash
   xdputil query          # should list DPUCZDX8G, B4096, 2 cores on the ZCU104 image
   ```
4. Python deps on the board:
   ```bash
   pip3 install onnxruntime websocket-client     # if no onnxruntime wheel, the app falls back to OpenCV-DNN
   ```

## Step 6 - Run it

```bash
scp compiled_ZCU104/yolov8n_face_416.xmodel dpu_face_app.py yolo_decode.py \
    ../w600k_mbf.onnx face_embeddings.jsonl root@<board-ip>:~/face/
ssh root@<board-ip>
cd ~/face
export DISPLAY=:0.0                                    # if a DisplayPort monitor is attached
python3 dpu_face_app.py --model yolov8n_face_416.xmodel --camera 0 --show
python3 dpu_face_app.py --model yolov8n_face_416.xmodel --camera 0 --host <laptop-ip>   # live DB from host_server.py
```
Every 30 frames it prints `detect / decode+NMS / recognise` milliseconds, which is your FPGA-vs-RPi/Jetson comparison table.

## Step 7 - Making it faster (once it works)

- **Two DPU cores**: the ZCU104 image has 2x B4096. Run capture, DPU and recognition in separate threads and
  create one VART runner per core to pipeline frames.
- **Recognition becomes the bottleneck** with several faces (MobileFaceNet on A53). Phase 2 below.
- 320x320 input roughly halves DPU work again if faces are close to the camera.
- The ZCU104's **VCU** (hardware H.264/H.265) can encode the annotated stream for a CCTV-style RTSP feed at ~0 CPU.

## Phase 2 - Recogniser on the DPU too

MobileFaceNet's PReLU blocks the DPU. Options, easiest first:
1. Use a DPU-native face-feature model from the Vitis AI Library (`FaceFeature` + face-landmark alignment).
   You must **re-register everyone**: embeddings from different models aren't comparable.
2. Retrain MobileFaceNet with PReLU -> LeakyReLU (needs a face-recognition training set, e.g. WebFace / MS1M).
3. Rewrite PReLU as `(1-a)*ReLU(x) + a*x` (per-channel scales + add) and check with the Inspector whether it
   stays on the DPU. Experimental.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `fingerprint mismatch` when loading the xmodel | Compiled for a different DPU config. Use the arch.json that matches the image (`xdputil query`). |
| Boxes in wrong places on board, fine on laptop | Preprocessing differs. Board, calibration and training must all use plain resize + RGB + /255. |
| Many detections with score ~0.5 everywhere | Output fix-points ignored. `DpuDetector` scales each output by `2**-fix_point`; don't remove it. |
| Inspector shows CPU nodes | A SiLU/Sigmoid survived (wrong checkpoint) or the Ultralytics version differs; use 8.3.155. |

## Files

| File | Runs on | Purpose |
|---|---|---|
| `yolov8n-face-hswish.yaml` | - | YOLOv8n, nc=1, Hardswish |
| `1_retrain_hardswish.py` | GPU / Colab | Fine-tune from `best.pt` at 416 |
| `dpu_head.py` | PC + docker | YOLOv8 with head cut before DFL (6 outputs) |
| `2_export_headcut_onnx.py` | PC | ONNX for laptop testing + DPU op check |
| `3_quantize_vitis_ai.py` | Vitis AI 3.5 docker | Inspect, calibrate, export INT8 xmodel |
| `4_compile.sh` | Vitis AI 3.5 docker | `vai_c_xir` for ZCU104 / ZCU102 / KV260 |
| `yolo_decode.py` | board + PC | NumPy DFL decode + NMS |
| `dpu_face_app.py` | board (`--backend dpu`) / PC (`--backend onnx`) | Detection + recognition + host DB sync + distances |
| `test_pipeline_cpu.py` | PC | Proves the decode/NMS matches Ultralytics and the original model |
