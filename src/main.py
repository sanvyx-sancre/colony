# src/main.py
from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

# adjust path to your trained model
MODEL_PATH = "runs/detect/colony_counter3/weights/best.pt"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.076  # adjust confidence threshold here
IOU_THRESHOLD = 0.4  # non-max suppression IoU

# load model
model = YOLO(MODEL_PATH)

def run_inference(source):
    source_path = Path(source)
    if not source_path.exists():
        print(f"[!] Source '{source}' not found")
        return

    results = model.predict(source=str(source_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, show=False)  # set show=True to see live

    counts = {}
    img = cv2.imread(str(source_path))

    for r in results:
        boxes = r.boxes
        for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            if conf < CONF_THRESHOLD:
                continue
            cls_name = model.names[int(cls_id)]
            counts[cls_name] = counts.get(cls_name, 0) + 1
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # print counts
    print("[✓] Colony counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # save processed image
    out_path = OUTPUT_DIR / source_path.name
    cv2.imwrite(str(out_path), img)
    print(f"[✓] Processed image saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main.py path/to/image_or_folder")
        sys.exit(1)
    run_inference(sys.argv[1])
