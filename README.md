# 🎯 Face Detection and Recognition: Edge Devices and FPGA

Real-time face detection with **YOLOv8n** (trained on WIDER FACE) and face recognition with
**MobileFaceNet** 512-D embeddings (`w600k_mbf.onnx`) matched by cosine similarity.
Runs on a PC, **Raspberry Pi 5**, **Jetson Nano**, and the **AMD ZCU104 FPGA**, where the detector
runs on the DPU accelerator in INT8 ([`fpga_zcu104/`](fpga_zcu104/README.md)).

---

## 🚀 Features

- 🧠 **YOLOv8n** face detector trained on WIDER FACE (`best.pt`, `best.onnx`)
- 🧷 **MobileFaceNet** (`w600k_mbf.onnx`) 512-D face embeddings
- 🗃️ Face database in **JSONL** (`face_embeddings.jsonl`, one `{"name", "embedding"}` per line)
- 🧪 Matching by **cosine similarity** (threshold 0.5), with face-to-face distances drawn on the frame
- 🌐 **Host server** that pushes new registrations live to edge devices over WebSocket
- ⚡ **FPGA port** on the ZCU104: detector on the DPU in 6.24 ms/frame, bit-exact with the reference

---

## 📁 Repository layout

| File | Purpose |
|---|---|
| `Yolov8n_training.py`, `Yolov8n_training_notebook.ipynb` | Train YOLOv8n on WIDER FACE |
| `Yolov8n_testOnImage_notebook.ipynb` | Test the detector on images |
| `best.pt`, `best.onnx` | Trained detector (PyTorch / ONNX with NMS inside) |
| `w600k_mbf.onnx` | MobileFaceNet face-embedding model |
| `Yolov8n_onnx_live.py` | Webcam face detection only |
| `FaceRegister.py` | Register a person (frontal, right, left views) into `face_embeddings.jsonl` |
| `face_store_json.py` | Read/write helpers for `face_embeddings.jsonl` |
| `FaceRecognition_Image_Inferencing.py` | Recognition on an image or a folder of images (CSV + annotated output) |
| `FaceRecognition_Video_Inferencing.py` | Recognition on the webcam |
| `host_server.py` | FastAPI server: serves the face DB (`GET /db`) and pushes updates (`/updates` WebSocket) |
| `server_connect.py` | Edge client: webcam recognition with the DB synced live from `host_server.py` |
| `fpga_zcu104/` | ZCU104 FPGA port: model adaptation, Vitis AI quantisation, board app ([README](fpga_zcu104/README.md)) |

---

## 🧭 Quick start (PC / Raspberry Pi / Jetson)

```bash
pip install opencv-python onnxruntime numpy scikit-learn
```

1. **Register people** (webcam; follow the prompts for frontal, right and left views, then type the name):
   ```bash
   python FaceRegister.py
   ```
2. **Recognise on the webcam:**
   ```bash
   python FaceRecognition_Video_Inferencing.py
   ```
3. **Recognise on images:** set `IN_PATH` at the bottom of `FaceRecognition_Image_Inferencing.py`
   (a file or a folder), then run it.
4. **Several devices, one database:** on the host run `pip install fastapi uvicorn` and
   `python host_server.py` (port 8000). On each device set `HOST_IP` in `server_connect.py`,
   `pip install requests websocket-client`, and run `python server_connect.py`.

---

## 📁 Project stages

1. ✅ Train YOLOv8n on the WIDER FACE dataset
2. ✅ Save the trained face detector (`best.pt`) and export to ONNX
3. ✅ Face recognition with MobileFaceNet embeddings
4. ✅ JSONL face database, cosine matching, live sync from a host server
5. ✅ Real-time demo on webcam; tested on Raspberry Pi and Jetson Nano
6. ✅ FPGA port to the AMD ZCU104: detector on the DPU (INT8), recognition on the ARM cores
7. 🔜 Live camera on the ZCU104 (USB link issue), recognition on the DPU, power benchmarks

---

## 📦 Dataset

We use **WIDER FACE** in YOLO format from KaggleHub.

```python
import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "/content/Dataset"
path = kagglehub.dataset_download("lylmsc/wider-face-for-yolo-training")
```

`Yolov8n_training.py` splits it into train/val and trains YOLOv8n (`epochs=20`, `imgsz=640`).

---

## 🔄 Export to ONNX

```bash
yolo export model=best.pt format=onnx simplify=True dynamic=False nms=True imgsz=640 opset=12
```

| Parameter | Purpose |
|---|---|
| `model=best.pt` | The trained YOLOv8 model |
| `format=onnx` | ONNX for cross-platform / edge deployment |
| `simplify=True` | Removes redundant ops for faster inference |
| `dynamic=False` | Fixed input size for speed |
| `nms=True` | NMS built into the ONNX graph |
| `imgsz=640` | Input size; inference frames must be resized to match |
| `opset=12` | ONNX opset 12 for compatibility |

> ⚠️ With `dynamic=False`, inference must use the export size (`cv2.resize(frame, (640, 640))`).
> The `best.onnx` in this repo has a dynamic input size, and the recognition scripts feed it 416x416.

The FPGA port does **not** use this `best.onnx`: NMS and dynamic ops can't run on the DPU, so it uses
a head-cut ReLU model at 416x416 instead. See [`fpga_zcu104/README.md`](fpga_zcu104/README.md).

---

## 🧠 Pipeline

**Detection (YOLOv8n):** resize, BGR → RGB, scale to [0, 1], CHW + batch → boxes + scores
(NMS inside `best.onnx`) → confidence threshold → boxes rescaled to the original frame.

**Recognition (MobileFaceNet):** crop each face → resize 112x112 → RGB → `(x/255 − 0.5)/0.5` →
512-D embedding → L2-normalise → cosine similarity against every registered face; best match
≥ 0.5 gives the name, otherwise `Unknown`.

**Registration:** 100 samples each of frontal, right and left views, averaged and normalised into one
embedding per person.

---

## ⚡ FPGA port: AMD ZCU104

| Measure | Result |
|---|---|
| Detector | YOLOv8n-ReLU, INT8 on the DPU (B4096), 1 subgraph |
| Accuracy (mAP50, 300 val images) | 0.4649 float → 0.4560 INT8 (−0.88 pts) |
| DPU latency | 6.24 ms/frame, bit-exact with the Colab golden reference |
| Recognition | MobileFaceNet on the Cortex-A53, ≈120 ms/face |

Laptop registrations work on the board unchanged. Full method, step-by-step procedure, results and
troubleshooting: **[`fpga_zcu104/README.md`](fpga_zcu104/README.md)**.

---

## 🔒 Privacy

`face_embeddings.jsonl` holds biometric data and is git-ignored; never commit or share it.
