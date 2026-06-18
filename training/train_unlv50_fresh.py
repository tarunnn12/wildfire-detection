# training/train_unlv50_fresh_yolov8m.py

import torch
from ultralytics import YOLO


DATA_YAML = r"C:\prsnl data\vscode\projects\wildfire\dataset\merged_unlv50.yaml"
PROJECT_DIR = r"C:\prsnl data\vscode\projects\wildfire\runs"


if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Fresh training from COCO-pretrained YOLOv8m
    # Do NOT load best_v7.pt here.
    model = YOLO("yolov8m.pt")

    model.train(
        data=DATA_YAML,

        # Main training setup
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=20,

        # Save location
        project=PROJECT_DIR,
        name="unlv50_fresh_yolov8m",
        exist_ok=True,

        # Optimisation
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,

        # Augmentation
        mosaic=0.5,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,

        # Validation/output
        val=True,
        plots=True,
        save=True,
    )