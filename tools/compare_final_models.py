from ultralytics import YOLO

if __name__ == '__main__':
    print("\n--- best_unlv50.pt on fasdd_unlv50 val set ---")
    m1 = YOLO(r"C:\prsnl data\vscode\projects\wildfire\weights\best_unlv50.pt")
    r1 = m1.val(
        data=r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_unlv50.yaml",
        imgsz=640, workers=0, name="eval_unlv50_on_fasdd"
    )

    print("\n--- best_fasdd_unlv50.pt on fasdd_unlv50 val set ---")
    m2 = YOLO(r"C:\prsnl data\vscode\projects\wildfire\weights\best_fasdd_unlv50.pt")
    r2 = m2.val(
        data=r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_unlv50.yaml",
        imgsz=640, workers=0, name="eval_fasdd_on_fasdd"
    )

    print("\n========= COMPARISON (on fasdd_unlv50 val set) =========")
    print(f"{'Metric':<15} {'unlv50':>10} {'fasdd_unlv50':>14} {'diff':>10}")
    print(f"{'mAP50':<15} {r1.box.map50:>10.4f} {r2.box.map50:>14.4f} {r2.box.map50-r1.box.map50:>+10.4f}")
    print(f"{'Precision':<15} {r1.box.mp:>10.4f} {r2.box.mp:>14.4f} {r2.box.mp-r1.box.mp:>+10.4f}")
    print(f"{'Recall':<15} {r1.box.mr:>10.4f} {r2.box.mr:>14.4f} {r2.box.mr-r1.box.mr:>+10.4f}")

    print("\n--- Now testing both on the ORIGINAL merged_unlv50 val set ---")
    print("\n--- best_unlv50.pt on merged_unlv50 val set ---")
    r3 = m1.val(
        data=r"C:\prsnl data\vscode\projects\wildfire\dataset\merged_unlv50.yaml",
        imgsz=640, workers=0, name="eval_unlv50_on_original"
    )
    print("\n--- best_fasdd_unlv50.pt on merged_unlv50 val set ---")
    r4 = m2.val(
        data=r"C:\prsnl data\vscode\projects\wildfire\dataset\merged_unlv50.yaml",
        imgsz=640, workers=0, name="eval_fasdd_on_original"
    )

    print("\n========= COMPARISON (on original merged_unlv50 val set) =========")
    print(f"{'Metric':<15} {'unlv50':>10} {'fasdd_unlv50':>14} {'diff':>10}")
    print(f"{'mAP50':<15} {r3.box.map50:>10.4f} {r4.box.map50:>14.4f} {r4.box.map50-r3.box.map50:>+10.4f}")