# src/core/trainer.py
from ultralytics import YOLO
from pathlib import Path

# paths
DATA_YAML = Path("data/yolo/dataset.yaml")  # new dataset yaml
MODEL_PATH = "yolov8n.pt"                    # small model for faster training
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

def train():
    if not DATA_YAML.exists():
        print(f"[!] Dataset YAML not found at {DATA_YAML}")
        return

    print("[*] Starting YOLO training...")
    model = YOLO(MODEL_PATH)
    model.train(
        data=str(DATA_YAML),  # pass path, not dict
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="colony_counter"
    )
    print("[✓] Training finished!")

if __name__ == "__main__":
    train()
