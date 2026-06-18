from pathlib import Path

DATASET = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\unlv_raw")
SPLITS = ["train", "valid", "test"]
REMAP = {0: 1, 1: 0}

def remap_label_file(label_path):
    lines = label_path.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(float(parts[0]))
        if old_cls not in REMAP:
            print(f"Unknown class {old_cls} in {label_path}")
            continue
        parts[0] = str(REMAP[old_cls])
        new_lines.append(" ".join(parts))
    label_path.write_text("\n".join(new_lines))

total = 0
for split in SPLITS:
    label_dir = DATASET / split / "labels"
    if not label_dir.exists():
        print(f"Skipping missing: {label_dir}")
        continue
    files = list(label_dir.glob("*.txt"))
    print(f"{split}: remapping {len(files)} files")
    for f in files:
        remap_label_file(f)
    total += len(files)

print(f"Done. Total remapped: {total}")