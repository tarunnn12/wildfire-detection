import torch
from ultralytics import YOLO

if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    model = YOLO("yolov8m.pt")

    model.train(
        data=r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_unlv50.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=20,
        project=r"C:\prsnl data\vscode\projects\wildfire\runs",
        name="fasdd_unlv50_fresh",
        exist_ok=True,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        mosaic=0.5,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,
        val=True,
        plots=True,
        save=True,
    )