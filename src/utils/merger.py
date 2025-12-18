from pathlib import Path

# paths
FIGSHARE_LABEL_DIR = Path("data/sample/figshare/annot_YOLO")
OUT_LABEL_DIR = Path("data/yolo/labels/figshare")  # remapped labels
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# your unified classes
CLASS_MAP = {
    "bsubtilis": 0,
    "calbicans": 1,
    "contamination": 2,
    "ecoli": 3,
    "paeruginosa": 4,
    "saureus": 5,
}

# Figshare original class IDs → class names
FIGSHARE_CLASS_NAMES = {
    0: "B.subtilis",
    1: "C.albicans",
    2: "E.coli",
    3: "P.aeruginosa",
    4: "S.aureus",
    # add more if needed
}

def remap_label_file(txt_path: Path):
    lines_out = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = parts
            cls_id = int(cls_id)
            cls_name = FIGSHARE_CLASS_NAMES.get(cls_id)
            if cls_name is None or cls_name.lower() not in CLASS_MAP:
                continue  # skip classes you don't care about
            new_cls_id = CLASS_MAP[cls_name.lower()]
            lines_out.append(f"{new_cls_id} {x} {y} {w} {h}")
    return lines_out

# process all labels
for txt_file in FIGSHARE_LABEL_DIR.glob("*.txt"):
    new_lines = remap_label_file(txt_file)
    if not new_lines:
        continue  # skip empty labels

    out_txt = OUT_LABEL_DIR / txt_file.name
    with open(out_txt, "w") as f:
        f.write("\n".join(new_lines))

    print(f"[+] remapped {txt_file.name}: {len(new_lines)} objects")
