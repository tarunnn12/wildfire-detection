from pathlib import Path
from collections import Counter

DATASET = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_raw")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = {0: "smoke", 1: "fire"}

def audit_split(split):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(image_dir.glob(ext))

    issues = 0
    class_counts = Counter()
    empty_labels = 0

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            issues += 1
            continue
        lines = label_path.read_text().strip().splitlines()
        if len(lines) == 0:
            empty_labels += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                issues += 1
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues += 1
                continue
            if cls not in CLASS_NAMES:
                issues += 1
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues += 1
                continue
            if w * h < 0.00005:
                issues += 1
                continue
            class_counts[cls] += 1

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {len(image_files)}")
    print(f"Empty labels: {empty_labels}")
    for cls_id, count in class_counts.items():
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {count}")
    print(f"Issues: {issues}")

for split in SPLITS:
    audit_split(split)