import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
YOLO_ROOT = "data/yolo"
OUT_ROOT = "data/synthetic"

IMG_SIZE = 640
NUM_SYNTHETIC = 50
COLONY_RANGE = (40, 220)
PAD = 0.1

# =========================================

IMG_DIR = os.path.join(YOLO_ROOT, "images", "train")
LBL_DIR = os.path.join(YOLO_ROOT, "labels", "train")

OUT_IMG = os.path.join(OUT_ROOT, "images")
OUT_LBL = os.path.join(OUT_ROOT, "labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# -------- extract crops automatically --------
print("extracting crops...")
crops = {}

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.rsplit(".", 1)[0] + ".txt")

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(lbl_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            bw *= w
            bh *= h
            xc *= w
            yc *= h

            pad_w = bw * PAD
            pad_h = bh * PAD

            x1 = int(max(xc - bw/2 - pad_w, 0))
            y1 = int(max(yc - bh/2 - pad_h, 0))
            x2 = int(min(xc + bw/2 + pad_w, w))
            y2 = int(min(yc + bh/2 + pad_h, h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cls = int(cls)
            crops.setdefault(cls, []).append(crop)

print(f"found crops per class: {[len(v) for v in crops.values()]}")

# -------- background sampler --------
def random_background():
    img = cv2.imread(os.path.join(IMG_DIR, random.choice(os.listdir(IMG_DIR))))
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# -------- degradation --------
def degrade(img):
    if random.random() < 0.7:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.4:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

# -------- generate synthetic plates --------
print("generating synthetic images...")
for idx in tqdm(range(NUM_SYNTHETIC)):
    bg = random_background()
    labels = []

    n_colonies = random.randint(*COLONY_RANGE)

    for _ in range(n_colonies):
        cls = random.choice(list(crops.keys()))
        crop = random.choice(crops[cls])

        scale = random.uniform(0.4, 1.3)
        crop = cv2.resize(crop, None, fx=scale, fy=scale)

        ch, cw, _ = crop.shape
        if ch >= IMG_SIZE or cw >= IMG_SIZE:
            continue

        x = random.randint(0, IMG_SIZE - cw)
        y = random.randint(0, IMG_SIZE - ch)

        bg[y:y+ch, x:x+cw] = crop

        xc = (x + cw/2) / IMG_SIZE
        yc = (y + ch/2) / IMG_SIZE
        bw = cw / IMG_SIZE
        bh = ch / IMG_SIZE

        labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    bg = degrade(bg)

    img_name = f"synthetic_{idx:05d}.jpg"
    cv2.imwrite(os.path.join(OUT_IMG, img_name), bg)

    with open(os.path.join(OUT_LBL, img_name.replace(".jpg", ".txt")), "w") as f:
        f.write("\n".join(labels))

print("done.")
