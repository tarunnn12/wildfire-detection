import random
import shutil
from pathlib import Path

random.seed(42)

SOURCE = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_raw")
DEST = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_20")

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
FRACTION = 0.20

for out_split in SPLIT_MAP.values():
    (DEST / "images" / out_split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / out_split).mkdir(parents=True, exist_ok=True)

def copy_subset(src_split, out_split, fraction):
    src_img_dir = SOURCE / src_split / "images"
    src_lbl_dir = SOURCE / src_split / "labels"

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(src_img_dir.glob(ext))
    images = sorted(images)
    random.shuffle(images)
    selected = images[:int(len(images) * fraction)]

    out_img_dir = DEST / "images" / out_split
    out_lbl_dir = DEST / "labels" / out_split

    count = 0
    for img in selected:
        lbl = src_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, out_img_dir / img.name)
        shutil.copy2(lbl, out_lbl_dir / lbl.name)
        count += 1

    print(f"{src_split} -> {out_split}: copied {count} / {len(images)}")

for src_split, out_split in SPLIT_MAP.items():
    copy_subset(src_split, out_split, FRACTION)

print("\nDone. Output:", DEST)