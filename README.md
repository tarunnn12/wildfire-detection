# Wildfire Detection System

A real-time fire and smoke detection system built with YOLOv8m and deployed via a Flask web application. Point any camera at fire or smoke footage and the system detects and classifies it live in the browser with bounding boxes and confidence scores.

## Live Demo

Run `python app/app.py` and open `http://localhost:5000`. The app streams your webcam feed, overlays detection boxes in real time, and shows session statistics (frames analysed, fire/smoke counts, last alert time).

## Final Model

| Property | Value |
|----------|-------|
| File | `weights/best.pt` (same as `weights/best_fasdd_unlv50.pt`) |
| Architecture | YOLOv8m |
| Input size | 640 × 640 |
| Classes | `smoke` (0), `fire` (1) |
| mAP50 | 0.810 |
| Precision | 0.815 |
| Recall | 0.734 |
| Live confidence | Consistently 85–94% on real fire/smoke footage |

This is the model the app loads by default and the one used for all demos.

## How This Model Was Built

The project went through several iterations of dataset improvement. Each step is preserved in this repo so the process is fully reproducible and auditable.

### 1. Initial dataset
Started with the **D-Fire** dataset (~21,000 images) and an **Indoor Fire & Smoke** dataset (~5,000 images), merged into one YOLO-format dataset with classes `0=smoke, 1=fire`.

### 2. Baseline training
Trained YOLOv8m from COCO-pretrained weights at 640×640 resolution. Result: **mAP50 = 0.822** on the original validation split (see `training/train.py` for this baseline script).

### 3. Adding real wildfire footage (UNLV dataset)
Added the UNLV Wildfire Detection dataset (~3,695 images of real drone/ground footage from wildfires including the 2023 Maui fires). This dataset used reversed class IDs (`0=fire, 1=smoke`), so labels were remapped to match the project standard before merging.

This produced `best_unlv50.pt` — evaluated on a harder, more diverse validation set that included this new wildfire footage, the original model scored 0.743 mAP50, while the UNLV-trained model scored **0.803 mAP50**, a +0.060 improvement, with fire recall improving by +0.061.

### 4. Adding FASDD (large-scale fire/smoke dataset)
Added a 20% audited subset (~9,500 images) of **FASDD** (Flame And Smoke Detection Dataset), a large, heterogeneous research dataset known for covering complex and varied fire scenes. This dataset also used reversed class IDs and was remapped before merging.

Trained fresh on the combined dataset (~28,600 train images). This produced the final model, `best_fasdd_unlv50.pt`, which improved over the UNLV-only model on **both** the new harder validation set (+0.12 mAP50) and the original validation set (0.803 → **0.810** mAP50) — confirming it generalized better without forgetting earlier learning.

### Dataset quality control
Every dataset was audited before use with `tools/audit_*.py` scripts, which check for:
- Missing or empty label files
- Invalid class IDs
- Out-of-range or malformed bounding boxes
- Extremely tiny boxes (likely annotation noise)

Across the full pipeline, fewer than 0.2% of annotations were found invalid and removed (`tools/fix_dataset.py`).

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA (recommended) — tested on RTX 5070 Laptop GPU
- Webcam or camera device

## Installation

```bash
git clone https://github.com/tarunnn12/wildfire-detection.git
cd wildfire-detection

conda create -n wildfire python=3.11 -y
conda activate wildfire

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python flask numpy Pillow
```

Verify GPU (optional):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running the App

```bash
python app/app.py
```
Open `http://localhost:5000`, allow camera access, and point the camera at fire or smoke (a video on a phone screen works well for testing).

## Project Structure

wildfire-detection/

│

├── app/

│   ├── app.py              # Flask app, video stream, session stats

│   ├── detector.py         # YOLOv8m inference wrapper

│   ├── templates/index.html

│   └── static/

│       ├── style.css       # Dark theme UI

│       └── main.js         # Live polling and UI updates

│

├── training/

│   ├── train.py                        # First baseline training script (D-Fire + Indoor)

│   └── train_fasdd_unlv50_fresh.py     # Final training script (UNLV + FASDD)

│

├── tools/                              # Dataset engineering pipeline

│   ├── audit_yolo_dataset.py           # Validates label files for the base dataset

│   ├── audit_unlv_dataset.py           # Validates UNLV dataset before merging

│   ├── audit_fasdd_dataset.py          # Validates FASDD dataset before merging

│   ├── remap_unlv_labels.py            # Fixes UNLV's reversed class IDs

│   ├── remap_fasdd_labels.py           # Fixes FASDD's reversed class IDs

│   ├── create_fasdd_subset.py          # Creates a random 20% audited subset of FASDD

│   ├── merge_unlv50_with_fasdd20.py    # Merges UNLV + FASDD subset into final dataset

│   ├── evaluate_models.py              # Compares two models on a validation set

│   └── compare_final_models.py         # Full head-to-head comparison across val sets

│

├── dataset/                 # Training data (raw + processed; large folders, see .gitattributes)

├── weights/

│   ├── best.pt                  # Final model (default, loaded by the app)

│   └── best_fasdd_unlv50.pt     # Same model, kept under its descriptive name

│

├── requirements.txt

└── README.md

## Training Your Own Model

To retrain from scratch on your own data, prepare a YOLO-format dataset with two classes (`0=smoke, 1=fire`), point a `data.yaml` at it, and run:

```bash
python training/train_fasdd_unlv50_fresh.py
```

Key training settings used for the final model:
- Base: YOLOv8m, COCO-pretrained
- Epochs: 80 (early stopping, patience=20)
- Image size: 640
- Batch size: 8
- Augmentation: mosaic, HSV shift, horizontal flip

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch 2.11 + CUDA 12.8
- Flask 3.0 + OpenCV
- HTML / CSS / JavaScript (no frontend framework)

## Dataset Sources

- [D-Fire Dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)
- [Indoor Fire & Smoke Detection](https://www.kaggle.com/datasets/sinchanashivanand/indoor-fire-and-smoke-detection-with-yolov8)
- [UNLV Wildfire Detection with Bounding Boxes](https://universe.roboflow.com/unlv-c6san/wildfire-detection-with-bounding-boxes)
- [FASDD (Flame And Smoke Detection Dataset)](https://universe.roboflow.com/forestfiresmoke/fasdd_cv-dx83j)
