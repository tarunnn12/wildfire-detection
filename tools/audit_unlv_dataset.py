from pathlib import Path
from PIL import Image
from collections import Counter

DATASET = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\unlv_raw")
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = {0: "smoke", 1: "fire"}

def audit_split(split):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(image_dir.glob(ext))

    issues = []
    class_counts = Counter()
    empty_labels = 0

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            issues.append(f"Missing label: {img_path.name}")
            continue

        lines = label_path.read_text().strip().splitlines()

        if len(lines) == 0:
            empty_labels += 1
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"Bad format: {label_path.name} line {line_no}")
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                issues.append(f"Non-numeric: {label_path.name} line {line_no}")
                continue
            if cls not in CLASS_NAMES:
                issues.append(f"Invalid class {cls}: {label_path.name}")
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues.append(f"Invalid bbox: {label_path.name} line {line_no}")
                continue
            if w * h < 0.00005:
                issues.append(f"Tiny box: {label_path.name} line {line_no}")
            class_counts[cls] += 1

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {len(image_files)}")
    print(f"Empty labels: {empty_labels}")
    for cls_id, count in class_counts.items():
        print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {count}")
    print(f"Issues found: {len(issues)}")
    for issue in issues[:20]:
        print(f"  {issue}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")

for split in SPLITS:
    audit_split(split)