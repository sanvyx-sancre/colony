import json
from pathlib import Path
from PIL import Image
import yaml

# paths
IMG_DIR = Path("data/images")
ANN_DIR = Path("data/labels_raw")
YOLO_LABELS_DIR = Path("data/labels")
NAMES_YAML_PATH = Path("data/yolo/names.yaml")

YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

def normalize_class(name: str) -> str:
    return name.lower().replace(".", "").replace("_", "").replace(" ", "")

def build_class_map(json_dir: Path) -> dict:
    """Scan all JSONs to build a dynamic class map."""
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
    # optional: format nicely
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

    print(f"[+] {base}: {len(out_lines)} objects")

def main():
    # build class map from all JSONs
    class_map = build_class_map(ANN_DIR)
    print(f"[+] detected classes: {class_map}")

    # convert all JSONs
    json_files = sorted(ANN_DIR.glob("*.json"))
    for jp in json_files:
        convert_one(jp, YOLO_LABELS_DIR, class_map)

    # generate names.yaml
    save_names_yaml(class_map, NAMES_YAML_PATH)

    print("[✓] conversion + names.yaml done")

if __name__ == "__main__":
    main()
