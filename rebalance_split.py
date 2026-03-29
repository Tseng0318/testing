"""
Restore, balance, and re-split CORROSION and NOCORROSION.

Source: /Users/kuoweitseng/Desktop/ENDG511_project/rust_dataset/
Dest:   /Users/kuoweitseng/Desktop/testing/rust_dataset/

Steps:
  1. Pool train + test from source for both classes
  2. Downsample CORROSION to match NOCORROSION (663 each)
  3. Re-split 80/20 → 530 train / 133 test per class
  4. Excess CORROSION → rust_dataset/discarded/CORROSION/
"""

import os
import random
import shutil

random.seed(42)

SRC  = "/Users/kuoweitseng/Desktop/ENDG511_project/rust_dataset"
DEST = "/Users/kuoweitseng/Desktop/testing/rust_dataset"
CLASSES = ["CORROSION", "NOCORROSION"]
SPLIT = 0.8

# ── 1. Pool all images from source ───────────────────────────────────────────
pools = {}
for cls in CLASSES:
    images = []
    for split in ["train", "test"]:
        folder = os.path.join(SRC, split, cls)
        images += [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    pools[cls] = images
    print(f"{cls}: {len(images)} total")

# ── 2. Downsample to smallest class ──────────────────────────────────────────
target_n = min(len(v) for v in pools.values())
print(f"\nBalancing to {target_n} images per class")

discarded = []
for cls in CLASSES:
    random.shuffle(pools[cls])
    excess = pools[cls][target_n:]
    pools[cls] = pools[cls][:target_n]
    if excess:
        discarded.append((cls, excess))
        print(f"  {cls}: discarding {len(excess)} images")

# ── 3. Split 80/20 ────────────────────────────────────────────────────────────
n_train = int(target_n * SPLIT)
n_test  = target_n - n_train
print(f"\nSplit: {n_train} train / {n_test} test per class")

splits = {}
for cls in CLASSES:
    splits[cls] = {
        "train": pools[cls][:n_train],
        "test":  pools[cls][n_train:],
    }

# ── 4. Copy to destination ────────────────────────────────────────────────────
for cls in CLASSES:
    for split in ["train", "test"]:
        dest_dir = os.path.join(DEST, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        # Clear any existing files
        for f in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, f))
        # Copy selected images
        for src in splits[cls][split]:
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))

# Copy discarded images
for cls, paths in discarded:
    dest_dir = os.path.join(DEST, "discarded", cls)
    os.makedirs(dest_dir, exist_ok=True)
    for src in paths:
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
    print(f"  Discarded {len(paths)} → discarded/{cls}/")

# ── 5. Verify ─────────────────────────────────────────────────────────────────
print("\nFinal counts:")
for cls in CLASSES:
    for split in ["train", "test"]:
        n = len(os.listdir(os.path.join(DEST, split, cls)))
        print(f"  {split}/{cls}: {n}")
