import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import json
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

BASE_DIR   = Path(__file__).resolve().parent
json_path  = BASE_DIR / "results_summary_all.json"
output_dir = BASE_DIR
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD
# =========================================================

with open(json_path, "r") as f:
    data = json.load(f)

models_data = data["models"]

# =========================================================
# CONFIG
# =========================================================

model_order = [
    "1D_ViT",
    "cnn_bilstm_transformer",
    "xresnet1d",
    "inception1d",
    "hubert_ecg",
    "resnet18_2d",
    "bilstm1d",
]

datasets = ["PTB-XL fold10", "CPSC2018", "Georgia"]

dataset_labels = {
    "PTB-XL fold10": "PTB-XL",
    "CPSC2018":      "CPSC 2018",
    "Georgia":       "Georgia",
}

colors = {
    "PTB-XL fold10": "#4C72B0",
    "CPSC2018":      "#55A868",
    "Georgia":       "#DD8452",
}

model_labels = {k: models_data[k]["label"] for k in model_order}

# =========================================================
# DATA
# =========================================================

f1_data  = {d: [models_data[m][d]["f1_macro"]   for m in model_order] for d in datasets}
auc_data = {d: [models_data[m][d]["auroc_macro"] for m in model_order] for d in datasets}

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(16, 7))

x      = np.arange(len(model_order))
width  = 0.22
offsets = [-width, 0, width]

# =========================================================
# BARS — Macro F1
# =========================================================

for i, d in enumerate(datasets):
    bars = ax.bar(
        x + offsets[i],
        f1_data[d],
        width=width,
        color=colors[d],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    for b, v in zip(bars, f1_data[d]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.012,
            f"{v:.2f}",
            ha="center", va="bottom",
            fontsize=7, fontweight="bold",
        )

# =========================================================
# DOTS — Macro AUROC
# =========================================================

for i, d in enumerate(datasets):
    xs = x + offsets[i]
    ys = auc_data[d]
    ax.scatter(
        xs, ys,
        s=90,
        color=colors[d],
        edgecolors="black",
        linewidth=0.6,
        zorder=5,
        marker="D",
    )
    for px, py in zip(xs, ys):
        ax.text(
            px, py + 0.013,
            f"{py:.2f}",
            ha="center", va="bottom",
            fontsize=7, fontweight="bold",
        )

# =========================================================
# AXIS
# =========================================================

ax.set_xticks(x)
ax.set_xticklabels(
    [model_labels[m] for m in model_order],
    fontsize=10, rotation=10, ha="right",
)
ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
ax.set_xlabel("Model", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "ECG Benchmark — Macro F1 (bars) and Macro AUROC (◆)\nacross Three Datasets",
    fontsize=14, fontweight="bold", pad=14,
)

# =========================================================
# LEGEND
# =========================================================

bar_handles = [
    plt.Rectangle((0, 0), 1, 1,
                  facecolor=colors[d], edgecolor="black", linewidth=0.5,
                  label=f"F1 — {dataset_labels[d]}")
    for d in datasets
]
dot_handles = [
    Line2D([0], [0],
           marker="D", color=colors[d],
           markersize=7, linestyle="None",
           markeredgecolor="black",
           label=f"AUC — {dataset_labels[d]}")
    for d in datasets
]
ax.legend(
    handles=bar_handles + dot_handles,
    ncol=2, fontsize=8.5,
    loc="upper left",
    framealpha=0.9, edgecolor="lightgray",
)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()
png_path = output_dir / "benchmark_bar.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print("Saved:", png_path)
