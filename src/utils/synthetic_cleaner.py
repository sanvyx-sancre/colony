import os

IMG_DIR = "data/synthetic/images"
LBL_DIR = "data/synthetic/labels"

image_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png"))
}

orphan_labels = []

for lbl in os.listdir(LBL_DIR):
    if not lbl.endswith(".txt"):
        continue

    name = os.path.splitext(lbl)[0]
    if name not in image_basenames:
        orphan_labels.append(lbl)

print(f"found {len(orphan_labels)} orphan labels:")
for lbl in orphan_labels:
    print(lbl)

for lbl in orphan_labels:
    os.remove(os.path.join(LBL_DIR, lbl))

print("orphan labels deleted.")
