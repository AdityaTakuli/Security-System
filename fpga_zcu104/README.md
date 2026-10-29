# Face Recognition on the AMD ZCU104 FPGA

The YOLOv8n face detector runs on the ZCU104's **DPU** (AMD's INT8 CNN accelerator in the FPGA fabric).
Everything around it runs on the board's quad Cortex-A53: preprocessing, the float head + decode + NMS,
MobileFaceNet recognition and cosine matching. Faces registered on a laptop with `FaceRegister.py`
are recognised on the board unchanged.

```
camera / image ─► A53: resize 416x416, RGB, INT8 ─► DPU: YOLOv8n-ReLU (INT8, 1 subgraph, 6.24 ms)
   ─► 3 feature maps (52/26/13 x 128 ch) ─► A53: float 1x1 head convs, DFL, NMS (yolo_decode.py)
   ─► face crops ─► A53: MobileFaceNet w600k_mbf (512-D) ─► cosine match vs face_embeddings.jsonl
```

## Results (measured on the real board)

| Measure | Result |
|---|---|
| Detection mAP50, 300 WIDER FACE val images | 0.4649 float → **0.4560 INT8** (−0.88 pts) |
| Same with the whole model in INT8 (rejected) | 0.3587 (−10.62 pts) |
| DPU subgraphs | **1** (the Hardswish version split into 11) |
| Golden check, DPU vs Colab reference | **100% exact, 0 LSB difference**, 13/13 faces |
| DPU latency, one of the two B4096 cores | **6.24 ms/frame** (≈160 FPS, network only) |
| Full app, 15-face image | 1870 ms/frame (recognition 1817 ms = 97%) |
| Recognition on the A53 (OpenCV-DNN) | ≈120 ms per face |

Hardware: ZCU104 (XCZU7EV), DPUCZDX8G_ISA1_B4096 ×2 @ 300 MHz, fingerprint `0x101000056010407`,
prebuilt image `xilinx-zcu104-dpu-v2022.2-v3.0.0.img.gz` (PetaLinux 2022.2, VART/XIR 3.0.0).
No Vivado or bitstream build is needed: the prebuilt image already contains the DPU.

## What had to change for the DPU

| Problem | Why the DPU refuses it | Fix |
|---|---|---|
| SiLU activations | Not a DPU operator; Hardswish split the graph into 11 DPU subgraphs | Fine-tuned with **ReLU** → 1 subgraph (`finetune.py`) |
| NMS / dynamic ops inside `best.onnx` | Can't run on the DPU | Head cut before DFL; decode + NMS in NumPy (`board/yolo_decode.py`) |
| `torch.chunk` in every C2f | XIR can't export multi-output ops | Two 1x1 convs `cv1a`/`cv1b` (`yolov8n_plain.py`) |
| Ultralytics needs Python ≥3.8 | Vitis AI 3.0 docker has Python 3.7 | Plain-PyTorch YOLOv8n, bit-identical (max diff 0.00e+00) |
| INT8 box logits got fix point −1 (step 2.0) → −10.6 mAP | DPU uses power-of-two scales | Final 1x1 head convs run in **float on the CPU** (box conv only where a face is likely) → −0.88 mAP |

## Folder layout

| File | Runs on | Purpose |
|---|---|---|
| `colab/finetune.py` | Colab GPU | Ultralytics fine-tune with Hardswish or ReLU (`--act`) |
| `colab/yolov8n-face-relu.yaml`, `yolov8n-face-hswish.yaml` | Colab | Model configs (nc=1) |
| `colab/dpu_head.py` | Colab | Ultralytics helpers: head-cut reference model, activation check |
| `colab/yolov8n_plain.py` | Colab / Vitis AI docker | YOLOv8n in plain PyTorch; `YOLO_ACT`, `YOLO_HEAD=feat`, `head_weights()` |
| `colab/convert_to_plain.py` | Colab | Ultralytics → plain weights + `head_weights.npz`, with equivalence checks |
| `colab/vitis_ai_colab.py` | Colab | Run the Vitis AI 3.0 docker without Docker (skopeo + umoci + chroot) |
| `colab/quantize_vai.py` | Vitis AI docker | Inspect, calibrate, INT8 mAP, golden reference, export xmodel |
| `colab/run_pipeline.sh` | Vitis AI docker | Calibrate → evaluate → compile for ZCU104 → check |
| `colab/check_xmodel.py` | Vitis AI docker / board | DPU subgraph count, fingerprint, tensor fix points |
| `board/dpu_face_app.py` | ZCU104 (laptop with `--backend onnx`) | Detection + recognition + DB sync with `host_server.py` |
| `board/yolo_decode.py` | ZCU104, laptop, Colab | Preprocessing, float CPU head, DFL decode, NMS |
| `board/golden_check.py` | ZCU104 | Bit-exact check of the DPU against the Colab reference |
| `board/board_day1.sh` | ZCU104 | First-boot checklist (DPU, packages, camera, golden check) |
| `board/run.sh`, `cam.sh`, `camcheck.sh` | ZCU104 | Run on images, live camera, camera diagnosis |

Generated files (not in git): `yolov8n_face_416.xmodel`, `head_weights.npz`, `golden.npz`,
`golden_image.jpg`, `*.pth`, and `face_embeddings.jsonl` (biometric data, never commit it).

---

## Part A: build the model (Google Colab)

**A1. Dataset.** WIDER FACE in YOLO format from KaggleHub (`lylmsc/wider-face-for-yolo-training`):
12,880 labelled images, split 80/20 (seed 42) into 10,304 train / 2,576 val, with a `data.yaml`
(`nc: 1`, `names: ['face']`). See `Yolov8n_training.py` in the repo root.

**A2. Fine-tune the activations** (T4 GPU, about 25 epochs each, from `fpga_zcu104/colab`):
```bash
python finetune.py --act hardswish --weights ../../best.pt     # SiLU -> Hardswish
python finetune.py --act relu      --weights /content/drive/MyDrive/fpga_runs/best_hswish.pt
```
Ultralytics validation (2,576 images, 416): SiLU 0.4775, Hardswish 0.5142, ReLU 0.5119 mAP50.

**A3. Convert to plain PyTorch and export the CPU head:**
```bash
python convert_to_plain.py --weights /content/drive/MyDrive/fpga_runs/best_relu.pt --images <val images>
# expect: max difference vs Ultralytics 0.00e+00; 0 face-count mismatches, 0.000 px
```

**A4. Vitis AI 3.0 in Colab** (no Docker daemon there):
```python
!apt-get -qq update && apt-get -qq install -y skopeo umoci
!skopeo copy docker://docker.io/xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 oci:/content/vai_oci:3.0
!umoci unpack --image /content/vai_oci:3.0 /content/vai_bundle && rm -rf /content/vai_oci
from vitis_ai_colab import setup, vai, vai_background, WORK
setup()
```
Copy into `WORK` (`/content/vai_work` = `/workspace` inside): `colab/yolov8n_plain.py`, `colab/quantize_vai.py`,
`colab/check_xmodel.py`, `colab/run_pipeline.sh`, `board/yolo_decode.py`, `yolov8n_face_relu_plain.pth`,
and a `lab_bundle/` with `calib/*.jpg` and `eval/images` + `eval/labels` (YOLO txt).

**A5. Quantise, evaluate, compile** (in the background so a Colab disconnect doesn't kill it):
```python
vai("YOLO_ACT=relu YOLO_HEAD=feat python quantize_vai.py --mode inspect")   # all ops on the DPU?
vai_background("run_pipeline.sh")                                           # then: !tail -f /content/vai_work/pipeline.log
```
Expected end of the log: `int8_mAP50 0.456`, `DPU subgraph number 1`, three outputs `[1, H, W, 128] fix_point 3`.
Outputs: `compiled_ZCU104/yolov8n_face_416.xmodel`, and in `quantize_feat/`: `head_weights.npz`,
`golden.npz`, `golden_image.jpg`, `accuracy_report.json`, `preview/`.

## Part B: run it on the ZCU104

**B1. Flash and boot.** Write the Vitis AI 3.0 ZCU104 image with balenaEtcher, set boot switch **SW6** to SD,
connect the UART micro-USB and power. Ethernet and a monitor are optional.

**B2. Serial console** (the board logs in as root):
```bash
sudo env TERM=xterm-256color picocom -b 115200 /dev/ttyUSB1      # exit: Ctrl+A, Ctrl+X
```
Paste commands **one line at a time**: multi-line pastes over the serial link arrive garbled.

**B3. Copy the deploy kit** via the SD card's FAT boot partition (on the laptop it mounts as e.g. `/media/$USER/XXXX-XXXX`):
put `board/*`, `yolov8n_face_416.xmodel`, `head_weights.npz`, `golden.npz`, `golden_image.jpg`,
`w600k_mbf.onnx`, `face_embeddings.jsonl` and a `test_images/` folder into `fpga_test/` on that partition. On the board:
```bash
mkdir -p /mnt/sd && mount /dev/mmcblk0p1 /mnt/sd
mkdir -p ~/deploy_kit && cp -r /mnt/sd/fpga_test/* ~/deploy_kit/ && cd ~/deploy_kit && chmod +x *.sh
```

**B4. Check the DPU and the model** (must end with `GOLDEN CHECK: PASS`):
```bash
bash board_day1.sh
```

**B5. Run on images:**
```bash
./run.sh test1.png        # one image  -> results/test1.jpg
./run.sh                  # every image in test_images/
cp -r results /mnt/sd/fpga_test/ && sync && poweroff   # then read the card on the laptop
```
`NO_ORT=1` (set inside the scripts) makes recognition use OpenCV-DNN, because the onnxruntime
package on this image aborts on import.

**B6. Live camera:**
```bash
./camcheck.sh     # needs a full row of '<' in the stream test
./cam.sh          # prints faces/names and timings every 30 frames; Ctrl+C to stop
```

## Registering people

Registration runs on the laptop, with the same MobileFaceNet and preprocessing as the board, so
embeddings are directly compatible:
```bash
python FaceRegister.py        # repo root; appends one line per person to face_embeddings.jsonl
```
Copy the updated `face_embeddings.jsonl` to the board, or run `host_server.py` on the laptop and start
the app with `--host <laptop-ip>` to receive new registrations live over the network.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not read test_images/...` | Wrong name; avoid spaces in file names |
| Segfault in `execute_async` | XIR graph freed while the runner uses it; fixed by `self.graph` in `DpuDetector` |
| App aborts at start | onnxruntime crash on import; run with `NO_ORT=1` (the scripts do) |
| `fingerprint mismatch` | xmodel compiled for another DPU; use the ZCU104 `arch.json` of Vitis AI 3.0 |
| Camera detected but no frames, `-71` in `dmesg` | USB signal/power problem; use a powered USB 3 hub or a direct OTG adapter and a good cable |
| OpenCV camera open hangs | Use `cam.sh` (fixed GStreamer pipeline) instead of `--camera 0` |
| Out of memory, programs killed | Hung test runs left processes; reboot |
| HDMI monitor shows nothing | HDMI needs a video design in the FPGA fabric; use the **DisplayPort** output |
| Serial console garbled or laggy | One console session only; `sudo systemctl stop ModemManager` |

## Status and next steps

- [x] Detector on the DPU, bit-exact, 0.88 mAP50 loss
- [x] Recognition on the board from still images, with laptop registrations
- [ ] Live camera: blocked by a USB link problem (error −71 through the current hub)
- [ ] Recognise each tracked face once instead of every frame; use the second DPU core
- [ ] Full WIDER FACE evaluation and power measurement vs Raspberry Pi and Jetson Nano
- [ ] MobileFaceNet on the DPU without re-registering anyone (PReLU → ReLU student distilled to the old embeddings)
