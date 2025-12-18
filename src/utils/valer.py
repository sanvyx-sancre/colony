from pathlib import Path
import shutil

# paths
LABEL_TRAIN_DIR = Path("data/yolo/labels/train")
LABEL_VAL_DIR = Path("data/yolo/labels/val")
LABEL_VAL_DIR.mkdir(parents=True, exist_ok=True)

# sp images that are already in val
sp_images = [
    "sp01_img11.jpg", "sp04_img02.jpg", "sp06_img04.jpg", "sp06_img05.jpg",
    "sp07_img10.jpg", "sp07_img11.jpg", "sp11_img01.jpg", "sp11_img02.jpg",
    "sp14_img18.jpg", "sp19_img08.jpg", "sp19_img09.jpg", "sp20_img01.jpg",
    "sp20_img02.jpg", "sp20_img03.jpg", "sp20_img04.jpg", "sp20_img05.jpg",
    "sp20_img06.jpg", "sp21_img30.jpg", "sp21_img31.jpg", "sp23_img06.jpg",
    "sp23_img07.jpg", "sp23_img08.jpg"
]

for img_name in sp_images:
    label_file = LABEL_TRAIN_DIR / f"{Path(img_name).stem}.txt"
    if label_file.exists():
        shutil.move(label_file, LABEL_VAL_DIR / f"{Path(img_name).stem}.txt")
        print(f"[+] moved label for {img_name}")
    else:
        print(f"[!] label not found for {img_name}")
