# src/main.py
from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

MODEL_PATH = "/home/sanvyx/Documents/TUBITAK/colony-counter/runs/detect/colony_s_laptop2/weights/best.pt"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGE_IOU_THRESHOLD = 0.5
IMG_SIZE = 640
CONF_THRESHOLD = 0.17

model = YOLO(MODEL_PATH)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


def run_inference(image_path: Path, conf: float, iou_thresh: float, merge_iou: float):
    if not image_path.exists():
        return

    img = cv2.imread(str(image_path))
    if img is None:
        return

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou_thresh,
        show=False,
        verbose=False,
    )

    detections = []

    r = results[0]
    if r.boxes is not None:
        for cls_id, conf_val, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
            detections.append({
                "cls": int(cls_id),
                "conf": float(conf_val),
                "box": list(map(int, xyxy)),
            })

    detections.sort(key=lambda x: x["conf"], reverse=True)

    filtered = []
    for det in detections:
        keep = True
        for kept in filtered:
            if iou(det["box"], kept["box"]) > merge_iou:
                keep = False
                break
        if keep:
            filtered.append(det)

    for det in filtered:
        x1, y1, x2, y2 = det["box"]
        cls_name = model.names[det["cls"]]
        conf_val = det["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls_name} {conf_val:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    # Total Colonies by Species (displayed on bottom left of images)
    species_counts = {}
    for det in filtered:
        name = model.names[det["cls"]]
        species_counts[name] = species_counts.get(name, 0) + 1

    if species_counts:
        # Prepare lines like "species: count"
        lines = [f"{k}: {v}" for k, v in species_counts.items()]

        # Drawing parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.4
        thickness = 3
        margin = 8

        # Compute line height and background size
        (w0, h0), baseline = cv2.getTextSize(lines[0], font, font_scale, thickness)
        line_height = h0 + 6
        text_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if w > text_width:
                text_width = w

        rect_w = text_width + margin * 2
        rect_h = line_height * len(lines) + margin

        # Position rectangle at bottom-left
        rect_x1 = margin
        rect_y1 = img.shape[0] - rect_h - margin
        rect_x2 = rect_x1 + rect_w
        rect_y2 = rect_y1 + rect_h

        # Draw semi-transparent background for readability
        overlay = img.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Put each line of text in white
        for i, line in enumerate(lines):
            text_x = rect_x1 + margin
            text_y = rect_y1 + margin + (i + 1) * line_height - 4
            cv2.putText(img, line, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    out_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(out_path), img)
