import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

INPUT_JSON = r"C:\prsnl data\vscode\projects\wildfire\runs\comparison_results.json"
OUTPUT_PATH = r"C:\prsnl data\vscode\projects\wildfire\runs\comparison_chart.png"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

MODEL_A = data["_meta"]["model_a_name"]
MODEL_B = data["_meta"]["model_b_name"]

COLOR_A = "#888780"
COLOR_B = "#D85A30"
TEXT_DARK = "#2C2C2A"

fig = plt.figure(figsize=(14, 10), facecolor="white")
fig.suptitle(f"Model comparison — {MODEL_A} vs {MODEL_B}", fontsize=16, fontweight="bold", color=TEXT_DARK, y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, top=0.90, bottom=0.08)

def bar_panel(ax, title, metrics_a, metrics_b, labels):
    x = np.arange(len(labels))
    width = 0.35
    vals_a = [metrics_a.get(l, 0) for l in labels]
    vals_b = [metrics_b.get(l, 0) for l in labels]

    bars1 = ax.bar(x - width/2, vals_a, width, label=MODEL_A, color=COLOR_A)
    bars2 = ax.bar(x + width/2, vals_b, width, label=MODEL_B, color=COLOR_B)

    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_DARK, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=TEXT_DARK)

ax1 = fig.add_subplot(gs[0, 0])
bar_panel(ax1, "On harder validation set",
          data[MODEL_A]["hard_val"], data[MODEL_B]["hard_val"],
          ["mAP50", "Precision", "Recall"])

ax2 = fig.add_subplot(gs[0, 1])
bar_panel(ax2, "On original validation set",
          data[MODEL_A]["original_val"], data[MODEL_B]["original_val"],
          ["mAP50", "Precision", "Recall"])

ax3 = fig.add_subplot(gs[1, 0])
per_class_a = data[MODEL_A]["original_val"].get("per_class_mAP50", {})
per_class_b = data[MODEL_B]["original_val"].get("per_class_mAP50", {})
class_labels = list(per_class_a.keys()) if per_class_a else ["smoke", "fire"]
bar_panel(ax3, "Per-class mAP50 (original val set)",
          per_class_a, per_class_b, class_labels)

ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
diff_hard = data[MODEL_B]["hard_val"]["mAP50"] - data[MODEL_A]["hard_val"]["mAP50"]
diff_orig = data[MODEL_B]["original_val"]["mAP50"] - data[MODEL_A]["original_val"]["mAP50"]

table_data = [
    ["Metric", MODEL_A, MODEL_B, "Diff"],
    ["mAP50 (hard val)", f"{data[MODEL_A]['hard_val']['mAP50']:.4f}", f"{data[MODEL_B]['hard_val']['mAP50']:.4f}", f"{diff_hard:+.4f}"],
    ["mAP50 (original val)", f"{data[MODEL_A]['original_val']['mAP50']:.4f}", f"{data[MODEL_B]['original_val']['mAP50']:.4f}", f"{diff_orig:+.4f}"],
    ["Precision (original)", f"{data[MODEL_A]['original_val']['Precision']:.4f}", f"{data[MODEL_B]['original_val']['Precision']:.4f}", ""],
    ["Recall (original)", f"{data[MODEL_A]['original_val']['Recall']:.4f}", f"{data[MODEL_B]['original_val']['Recall']:.4f}", ""],
]

table = ax4.table(cellText=table_data, loc="center", cellLoc="center", bbox=[0, 0.1, 1, 0.85])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#D3D1C7")
    if row == 0:
        cell.set_text_props(fontweight="bold")

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT_PATH}")