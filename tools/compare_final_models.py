from ultralytics import YOLO
import json
from pathlib import Path

OUTPUT_JSON = r"C:\prsnl data\vscode\projects\wildfire\runs\comparison_results.json"

MODEL_A_PATH = r"C:\prsnl data\vscode\projects\wildfire\weights\best_unlv50.pt"
MODEL_B_PATH = r"C:\prsnl data\vscode\projects\wildfire\weights\best_fasdd_unlv50.pt"
MODEL_A_NAME = "unlv50"
MODEL_B_NAME = "fasdd_unlv50"

HARD_VAL_YAML = r"C:\prsnl data\vscode\projects\wildfire\dataset\fasdd_unlv50.yaml"
ORIGINAL_VAL_YAML = r"C:\prsnl data\vscode\projects\wildfire\dataset\merged_unlv50.yaml"

def extract_metrics(results):
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "Precision": float(results.box.mp),
        "Recall": float(results.box.mr),
    }
    per_class = {}
    if hasattr(results.box, "maps") and results.names:
        for idx, name in results.names.items():
            try:
                per_class[name] = float(results.box.maps[idx])
            except (IndexError, TypeError):
                pass
    metrics["per_class_mAP50"] = per_class
    return metrics

if __name__ == "__main__":
    output = {MODEL_A_NAME: {}, MODEL_B_NAME: {}}

    print(f"\n--- {MODEL_A_NAME} on hard val set ---")
    model_a = YOLO(MODEL_A_PATH)
    r = model_a.val(data=HARD_VAL_YAML, imgsz=640, workers=0, name=f"eval_{MODEL_A_NAME}_hard")
    output[MODEL_A_NAME]["hard_val"] = extract_metrics(r)

    print(f"\n--- {MODEL_B_NAME} on hard val set ---")
    model_b = YOLO(MODEL_B_PATH)
    r = model_b.val(data=HARD_VAL_YAML, imgsz=640, workers=0, name=f"eval_{MODEL_B_NAME}_hard")
    output[MODEL_B_NAME]["hard_val"] = extract_metrics(r)

    print(f"\n--- {MODEL_A_NAME} on original val set ---")
    r = model_a.val(data=ORIGINAL_VAL_YAML, imgsz=640, workers=0, name=f"eval_{MODEL_A_NAME}_original")
    output[MODEL_A_NAME]["original_val"] = extract_metrics(r)

    print(f"\n--- {MODEL_B_NAME} on original val set ---")
    r = model_b.val(data=ORIGINAL_VAL_YAML, imgsz=640, workers=0, name=f"eval_{MODEL_B_NAME}_original")
    output[MODEL_B_NAME]["original_val"] = extract_metrics(r)

    output["_meta"] = {
        "model_a_name": MODEL_A_NAME,
        "model_b_name": MODEL_B_NAME,
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved comparison results to: {OUTPUT_JSON}")
    print("\n========= SUMMARY =========")
    for split in ["hard_val", "original_val"]:
        print(f"\n{split}:")
        print(f"  {MODEL_A_NAME}: mAP50={output[MODEL_A_NAME][split]['mAP50']:.4f}")
        print(f"  {MODEL_B_NAME}: mAP50={output[MODEL_B_NAME][split]['mAP50']:.4f}")