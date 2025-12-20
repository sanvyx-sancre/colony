# sentetik veri oluşturma için kod
import os
import cv2
import math
import random
import numpy as np
from tqdm import tqdm
import json

# ================= CONFIG =================
YOLO_ROOT = "data/yolo"
OUT_ROOT = "data/synthetic"

IMG_SIZE = 640
NUM_SYNTHETIC = 308
MAX_TRIES = 50

PLATE_RADIUS = int(IMG_SIZE * 0.45)

CLASS_NAMES = [
    'Bsubtilis',
    'Calbicans',
    'Contamination',
    'Ecoli',
    'Paeruginosa',
    'Saureus'
]

# realistic scale ranges per species
SCALE_RANGES = {
    0: (0.4, 0.7),   # Bsubtilis
    1: (0.6, 1.1),   # Calbicans (bigger)
    2: (0.5, 1.3),   # Contamination (chaotic)
    3: (0.3, 0.6),   # Ecoli
    4: (0.4, 0.7),   # Paeruginosa
    5: (0.4, 0.8),   # Saureus
}

# optional mapping file: {"20": 2, "21": 0, "22": 1, ...}
CLASS_MAP_PATH = os.path.join(YOLO_ROOT, "class_map.json")

# =========================================

IMG_DIR = os.path.join(YOLO_ROOT, "images", "train")
LBL_DIR = os.path.join(YOLO_ROOT, "labels", "train")

OUT_IMG = os.path.join(OUT_ROOT, "images")
OUT_LBL = os.path.join(OUT_ROOT, "labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# load optional mapping
class_map = {}
if os.path.exists(CLASS_MAP_PATH):
    try:
        with open(CLASS_MAP_PATH, "r") as mf:
            class_map = json.load(mf)
            # normalize keys to strings and values to ints
            class_map = {str(k): int(v) for k, v in class_map.items()}
            print("loaded class map with", len(class_map), "entries from", CLASS_MAP_PATH)
    except Exception as e:
        print("warning: failed to load class mapping:", e)

# -------- load real crops --------
print("extracting real colony crops...")
crops = {}
token_counts = {}

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    if img is None:
        continue

    h, w, _ = img.shape
    lbl_path = os.path.join(LBL_DIR, img_name.rsplit(".", 1)[0] + ".txt")

    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[0]
            token_counts[token] = token_counts.get(token, 0) + 1

            if len(parts) < 5:
                print(f"warning: malformed label line {lbl_path}:{line_no} -> {line.strip()!r}")
                continue

            # try numeric class first, else map via class_map or class name
            cls = None
            try:
                cls_candidate = int(float(token))
                # if numeric and within range, accept
                if 0 <= cls_candidate < len(CLASS_NAMES):
                    cls = cls_candidate
                # else try to remap using class_map
                elif token in class_map:
                    cls = int(class_map[token])
                    print(f"info: remapped token {token!r} -> class {cls} using class_map")
                elif str(cls_candidate) in class_map:
                    cls = int(class_map[str(cls_candidate)])
                    print(f"info: remapped token {token!r} -> class {cls} using class_map")
                else:
                    print(f"warning: skipping label with invalid class {cls_candidate} in {lbl_path}:{line_no}")
                    continue
            except ValueError:
                # token wasn't numeric; try mapping by token string (e.g., class name)
                if token in class_map:
                    cls = int(class_map[token])
                    print(f"info: remapped token {token!r} -> class {cls} using class_map")
                else:
                    # try matching token as a class name (case-insensitive)
                    lower_names = [n.lower() for n in CLASS_NAMES]
                    if token.lower() in lower_names:
                        cls = lower_names.index(token.lower())
                        print(f"info: interpreted token {token!r} as class name -> {cls}")
                    else:
                        print(f"warning: invalid class token in {lbl_path}:{line_no} -> {token!r}")
                        continue

            if cls is None or cls < 0 or cls >= len(CLASS_NAMES):
                print(f"warning: skipping label with invalid class {cls} in {lbl_path}:{line_no}")
                continue

            try:
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                print(f"warning: malformed bbox numbers in {lbl_path}:{line_no} -> {parts[1:5]}")
                continue

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size > 0:
                crops.setdefault(cls, []).append(crop)

print({(CLASS_NAMES[k] if 0 <= k < len(CLASS_NAMES) else f"cls_{k}"): len(v) for k, v in crops.items()})

if not crops:
    # write token counts to a file to help create a mapping
    report_path = os.path.join(YOLO_ROOT, "class_token_counts.txt")
    with open(report_path, "w") as rf:
        for tok, cnt in sorted(token_counts.items(), key=lambda x: -x[1]):
            rf.write(f"{tok}\t{cnt}\n")
    print("error: no valid crops found. Wrote token counts to:", report_path)
    print("Either create a class_map.json mapping old tokens to 0..5, or fix your labels. Example class_map.json:")
    example = { "20": 2, "21": 0, "22": 1 }  # edit based on token counts
    print(json.dumps(example, indent=2))
    raise SystemExit(1)

# -------- helpers --------
def inside_plate(x, y, r):
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    return math.hypot(x - cx, y - cy) + r < PLATE_RADIUS

def soft_alpha(h, w):
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_d = min(cx, cy)
    alpha = np.clip(1 - (dist / max_d), 0, 1)
    alpha = cv2.GaussianBlur(alpha, (21,21), 0)
    return alpha

def paste(bg, fg, x, y, alpha):
    h, w = fg.shape[:2]
    roi = bg[y:y+h, x:x+w].astype(np.float32)
    fg_f = fg.astype(np.float32)
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    for c in range(3):
        roi[:,:,c] = roi[:,:,c] * (1 - a[:,:,0]) + fg_f[:,:,c] * a[:,:,0]
    bg[y:y+h, x:x+w] = np.clip(roi, 0, 255).astype(np.uint8)

# -------- generate --------
print("generating synthetic plates...")

for idx in tqdm(range(NUM_SYNTHETIC)):
    # pick a random background image (pick only images)
    bg_candidates = [n for n in os.listdir(IMG_DIR) if n.lower().endswith((".jpg", ".png"))]
    bg = cv2.imread(os.path.join(IMG_DIR, random.choice(bg_candidates)))
    bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

    placed = []
    labels = []

    colony_count = random.randint(50, 150)

    for _ in range(colony_count):
        cls = random.choice(list(crops.keys()))
        crop = random.choice(crops[cls])

        scale_range = SCALE_RANGES.get(cls, (0.5, 1.0))
        scale = random.uniform(*scale_range)
        crop = cv2.resize(crop, None, fx=scale, fy=scale)

        h, w = crop.shape[:2]
        r = max(h, w) // 2

        alpha = soft_alpha(h, w)

        for _ in range(MAX_TRIES):
            x = random.randint(0, IMG_SIZE - w)
            y = random.randint(0, IMG_SIZE - h)

            if not inside_plate(x + w//2, y + h//2, r):
                continue

            paste(bg, crop, x, y, alpha)

            xc = (x + w / 2) / IMG_SIZE
            yc = (y + h / 2) / IMG_SIZE
            bw = w / IMG_SIZE
            bh = h / IMG_SIZE

            labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            break

    img_name = f"synthetic_{idx:05d}.jpg"
    cv2.imwrite(os.path.join(OUT_IMG, img_name), bg)

    with open(os.path.join(OUT_LBL, img_name.replace(".jpg", ".txt")), "w") as f:
        f.write("\n".join(labels))

print("done.")