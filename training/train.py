import torch
print(f"Using: {torch.cuda.get_device_name(0)}")
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    model.train(
        data=r'C:\prsnl data\vscode\projects\wildfire\dataset\merged.yaml',
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        patience=15,
        workers=0,
        project=r'C:\prsnl data\vscode\projects\wildfire\runs',
        name='aerialiq_v7',
        exist_ok=True,
        cos_lr=True,
        val=True,
        augment=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
    )