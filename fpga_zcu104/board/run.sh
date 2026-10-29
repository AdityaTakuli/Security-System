#!/bin/sh
# Face detection + recognition on still images (ZCU104).
#   ./run.sh test1.png          one image from test_images/
#   ./run.sh /path/to/img.jpg   any image
#   ./run.sh                    every image in test_images/
# Annotated results go to results/<name>.jpg
cd "$(dirname "$0")" && mkdir -p results

run() {
    NO_ORT=1 python3 dpu_face_app.py \
        --model yolov8n_face_416.xmodel --head head_weights.npz \
        --rec w600k_mbf.onnx --db face_embeddings.jsonl \
        --image "$1" --save "results/$(basename "${1%.*}").jpg"
}

if [ -n "$1" ]; then
    [ -f "$1" ] || set -- "test_images/$1"
    run "$1"
else
    for f in test_images/*; do
        echo "== $f"
        run "$f"
    done
fi
