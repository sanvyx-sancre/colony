import os
import shutil

SAMPLES_DIR = "data/samples"
IMAGES_DIR = "data/images"
LABELS_DIR = "data/labels_raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".json"


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    files = os.listdir(SAMPLES_DIR)

    img_count = 0
    json_count = 0

    for fname in files:
        src = os.path.join(SAMPLES_DIR, fname)
        if not os.path.isfile(src):
            continue

        name, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext in IMAGE_EXTS:
            dst = os.path.join(IMAGES_DIR, fname)
            shutil.move(src, dst)
            img_count += 1

        elif ext == LABEL_EXT:
            dst = os.path.join(LABELS_DIR, fname)
            shutil.move(src, dst)
            json_count += 1

    print(f"[+] moved {img_count} images")
    print(f"[+] moved {json_count} json labels")


if __name__ == "__main__":
    main()
