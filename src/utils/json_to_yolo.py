import json
from pathlib import Path
from PIL import Image
import yaml
from sklearn.model_selection import train_test_split
import shutil

# ------------------ paths ------------------
IMG_DIR = Path("data/images")             # original images
ANN_DIR = Path("data/labels_raw")         # original JSONs
YOLO_DIR = Path("data/yolo")              # final YOLO folder
YOLO_LABELS_DIR = YOLO_DIR / "labels_tmp" # temporary label storage

# train/val folders
IMG_TRAIN = YOLO_DIR / "images/train"
IMG_VAL   = YOLO_DIR / "images/val"
LBL_TRAIN = YOLO_DIR / "labels/train"
LBL_VAL   = YOLO_DIR / "labels/val"

NAMES_YAML_PATH = YOLO_DIR / "names.yaml"

for p in [YOLO_LABELS_DIR, IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------ helper funcs ------------------
def normalize_class(name: str) -> str:
    return name.lower().replace(".", "").replace("_", "").replace(" ", "")

def build_class_map(json_dir: Path) -> dict:
    classes = set()
    for jp in json_dir.glob("*.json"):
        with open(jp, "r") as f:
            data = json.load(f)
        for obj in data.get("labels", []):
            cls = normalize_class(obj.get("class", ""))
            classes.add(cls)
    return {cls: i for i, cls in enumerate(sorted(classes))}

def save_names_yaml(class_map: dict, out_file: Path):
    names = [None] * len(class_map)
    for cls, idx in class_map.items():
        names[idx] = cls
    names = [name.replace("_", " ").capitalize() for name in names]
    data = {"names": names, "nc": len(names)}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        yaml.dump(data, f)
    print(f"[+] saved names.yaml with {len(names)} classes to {out_file}")

def convert_one(json_path: Path, out_dir: Path, class_map: dict):
    base = json_path.stem

    # find matching image
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = IMG_DIR / f"{base}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        print(f"[!] image not found for {base}")
        return

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    with open(json_path, "r") as f:
        data = json.load(f)

    labels = data.get("labels", [])
    out_lines = []

    for obj in labels:
        cls_raw = obj.get("class", "")
        cls = normalize_class(cls_raw)
        cls_id = class_map.get(cls)
        if cls_id is None:
            print(f"[!] skipping unknown class '{cls_raw}' in {base}")
            continue

        x = obj["x"] / img_w
        y = obj["y"] / img_h
        w = obj["width"] / img_w
        h = obj["height"] / img_h

        out_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    out_file = out_dir / f"{base}.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(out_lines))

    return img_path  # return image path for later split

# ------------------ main workflow ------------------
def main(test_size=0.2, random_seed=42):
    # 1️⃣ detect classes
    class_map = build_class_map(ANN_DIR)
    print(f"[+] detected classes: {class_map}")

    # 2️⃣ convert all JSONs → YOLO labels
    json_files = sorted(ANN_DIR.glob("*.json"))
    base_names = []
    for jp in json_files:
        img_path = convert_one(jp, YOLO_LABELS_DIR, class_map)
        if img_path:
            base_names.append(jp.stem)

    # 3️⃣ split into train/val
    train_bases, val_bases = train_test_split(base_names, test_size=test_size, random_state=random_seed)

    for base_list, img_dest, lbl_dest in [(train_bases, IMG_TRAIN, LBL_TRAIN), (val_bases, IMG_VAL, LBL_VAL)]:
        for base in base_list:
            # copy image
            for ext in [".jpg", ".png", ".jpeg"]:
                img_file = IMG_DIR / f"{base}{ext}"
                if img_file.exists():
                    shutil.copy(img_file, img_dest / img_file.name)
            # copy label
            lbl_file = YOLO_LABELS_DIR / f"{base}.txt"
            if lbl_file.exists():
                shutil.copy(lbl_file, lbl_dest / lbl_file.name)

    # 4️⃣ generate names.yaml
    save_names_yaml(class_map, NAMES_YAML_PATH)

    print(f"[✓] YOLO dataset prepared: {len(train_bases)} train, {len(val_bases)} val")

if __name__ == "__main__":
    main()
