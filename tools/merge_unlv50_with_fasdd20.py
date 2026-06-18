import shutil
from pathlib import Path

UNLV50 = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\merged_unlv50")
FASDD  = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_20")
OUT    = Path(r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_unlv50")

SPLITS = ["train", "val", "test"]

for split in SPLITS:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_files(src_img, src_lbl, out_img, out_lbl, prefix):
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(src_img.glob(ext))
    count = 0
    for img in images:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        new_stem = f"{prefix}_{img.stem}"
        shutil.copy2(img, out_img / (new_stem + img.suffix))
        shutil.copy2(lbl, out_lbl / (new_stem + ".txt"))
        count += 1
    return count

for split in SPLITS:
    n1 = copy_files(
        UNLV50 / "images" / split, UNLV50 / "labels" / split,
        OUT / "images" / split, OUT / "labels" / split, "base"
    )
    n2 = copy_files(
        FASDD / "images" / split, FASDD / "labels" / split,
        OUT / "images" / split, OUT / "labels" / split, "fasdd"
    )
    print(f"{split}: base={n1}, fasdd={n2}, total={n1+n2}")

print("\nDone. Output:", OUT)