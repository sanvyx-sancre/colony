# src/main.py
from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

# ================= CONFIG =================
MODEL_PATH = "/home/sanvyx/Documents/TUBITAK/colony-counter/runs/detect/colony_s_laptop2/weights/best.pt"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.17
IOU_THRESHOLD = 0.4          # model nms (class-aware, we override later)
MERGE_IOU_THRESHOLD = 0.4    # our custom class-agnostic merge
IMG_SIZE = 640
# =========================================

# load model once
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


def run_inference(image_path: Path):
    if not image_path.exists():
        print(f"[!] file not found: {image_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] failed to read image: {image_path}")
        return

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show=False,
        verbose=False,
    )

    detections = []

    r = results[0]
    if r.boxes is not None:
        for cls_id, conf, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
            detections.append({
                "cls": int(cls_id),
                "conf": float(conf),
                "box": list(map(int, xyxy)),
            })

    # sort by confidence (highest first)
    detections.sort(key=lambda x: x["conf"], reverse=True)

    # class-agnostic suppression
    filtered = []
    for det in detections:
        keep = True
        for kept in filtered:
            if iou(det["box"], kept["box"]) > MERGE_IOU_THRESHOLD:
                keep = False
                break
        if keep:
            filtered.append(det)

    # count + draw
    counts = {}
    for det in filtered:
        cls_name = model.names[det["cls"]]
        counts[cls_name] = counts.get(cls_name, 0) + 1

        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    # print results
    print(f"\n[✓] results for: {image_path.name}")
    total = 0
    for k, v in counts.items():
        print(f"  {k}: {v}")
        total += v
    print(f"  total colonies: {total}")

    # save output
    out_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(out_path), img)
    print(f"[✓] saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/main.py <image_or_folder>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        images = [p for p in path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        if not images:
            print("[!] no images found in folder")
            sys.exit(1)

        for img_path in images:
            run_inference(img_path)
    else:
        run_inference(path)
