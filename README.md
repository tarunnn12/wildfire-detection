# Wildfire Detection System

A real-time fire and smoke detection system built with YOLOv8m and deployed via a Flask web application. Point any camera at fire or smoke footage and the system detects and classifies threats live in the browser.

## Demo

The system runs in your browser at `http://localhost:5000` and shows:
- Live annotated camera feed with bounding boxes
- Fire and smoke detection with confidence scores
- Session statistics and alert history
- Real-time FPS counter

## Model

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m |
| Input size | 640 × 640 |
| Classes | fire, smoke |
| mAP50 | 0.822 |
| Precision | 0.831 |
| Recall | 0.737 |
| Training images | 26,771 |
| Training dataset | D-Fire + Indoor Fire & Smoke Detection |

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA (recommended) or CPU
- Webcam or camera device

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/tarunnn12/wildfire-detection.git
cd wildfire-detection
```

### 2. Create conda environment
```bash
conda create -n wildfire python=3.11 -y
conda activate wildfire
```

### 3. Install PyTorch with CUDA (skip if CPU only)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install dependencies
```bash
pip install ultralytics opencv-python flask numpy Pillow
```

### 5. Verify GPU (optional)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running the App

```bash
conda activate wildfire
python app/app.py
```

Open `http://localhost:5000` in your browser. Allow camera access when prompted.

## Testing

Point your camera at any of the following:
- Fire images or videos on a phone screen
- Smoke footage
- Plain background (should show no detections)

## Project Structure
wildfire-detection/
│
├── app/
│   ├── app.py              # Flask application and video stream
│   ├── detector.py         # YOLOv8m inference wrapper
│   ├── templates/
│   │   └── index.html      # Main UI
│   └── static/
│       ├── style.css       # Dark theme stylesheet
│       └── main.js         # Live polling and UI updates
│
├── training/
│   └── train.py            # YOLOv8m training script
│
├── weights/
│   └── best_v7.pt          # Trained model weights
│
├── requirements.txt
└── README.md

## Training Your Own Model

To retrain on your own dataset:

1. Prepare dataset in YOLO format with two classes: `smoke` (0), `fire` (1)
2. Update `dataset.yaml` with your dataset paths
3. Run training:

```bash
python training/train.py
```

Training configuration used:
- Base model: YOLOv8m pretrained on COCO
- Epochs: 80 (early stopping patience=15)
- Image size: 640
- Batch size: 8
- Augmentation: mosaic, HSV shift, horizontal flip, mixup

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch 2.11 + CUDA 12.8
- Flask 3.0
- OpenCV
- HTML / CSS / JavaScript

## Dataset Sources

- [D-Fire Dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) — 21,000+ images
- [Indoor Fire & Smoke Detection](https://www.kaggle.com/datasets/sinchanashivanand/indoor-fire-and-smoke-detection-with-yolov8) — 5,000+ images