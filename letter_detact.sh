#!/usr/bin/env bash
set -e

IMAGE_PATH="/root/letter_tasks/assets/letter_9.jpeg"
YOLO_MODEL="/root/letter_tasks/runs/detect/train2/weights/best.pt"
CNN_MODEL="/root/letter_tasks/models/letter_cnn.pt"

YOLO_JSON_DIR="/root/letter_tasks/output/yolo_json"
YOLO_OVERLAY_DIR="/root/letter_tasks/output/yolo_json/overlays"

DETECT_JSON="$YOLO_JSON_DIR/letter_9.json"
RECOGNIZED_JSON="/root/letter_tasks/output/letter_9_cnn.json"
RECOGNIZED_OVERLAY="/root/letter_tasks/output/letter_9_cnn_overlay.png"

python /root/letter_tasks/yolo_detect_to_json.py \
  --model "$YOLO_MODEL" \
  --source "$IMAGE_PATH" \
  --output-dir "$YOLO_JSON_DIR" \
  --overlay-dir "$YOLO_OVERLAY_DIR" \
  --conf 0.25 \
  --save-overlay

python /root/letter_tasks/recognize_letter_cnn.py \
  --image "$IMAGE_PATH" \
  --input-json "$DETECT_JSON" \
  --model-path "$CNN_MODEL" \
  --output-json "$RECOGNIZED_JSON" \
  --output-overlay "$RECOGNIZED_OVERLAY"
