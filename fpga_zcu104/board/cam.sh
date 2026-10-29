#!/bin/sh
# Live face recognition from a USB camera (ZCU104). Ctrl+C to stop.
#   ./cam.sh        /dev/video0
#   ./cam.sh 2      /dev/video2
# A fixed 640x480 UYVY GStreamer pipeline is used because OpenCV's default
# camera open waits forever when the camera delivers no frames.
cd "$(dirname "$0")"
DEV=${1:-0}
echo "Camera /dev/video$DEV - press Ctrl+C to stop"
NO_ORT=1 python3 dpu_face_app.py \
    --model yolov8n_face_416.xmodel --head head_weights.npz \
    --rec w600k_mbf.onnx --db face_embeddings.jsonl \
    --camera "v4l2src device=/dev/video$DEV ! video/x-raw,format=UYVY,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
