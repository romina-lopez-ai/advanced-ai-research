import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import json
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

BASE_DIR   = Path(__file__).resolve().parent
json_path  = BASE_DIR / "results_summary_all.json"
output_dir = BASE_DIR
output_dir.mkdir(parents=True, exist_ok=True)

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

dataset_markers = {
    "PTB-XL fold10": "*",
    "CPSC2018":      "^",
    "Georgia":       "o",
}

dataset_sizes_plot = {
    "PTB-XL fold10": 300,
    "CPSC2018":      130,
    "Georgia":       110,
}

model_colors = {
    "1D_ViT":                  "#2A5BA8",   # deeper blue
    "cnn_bilstm_transformer":  "#B82C30",   # deeper red
    "xresnet1d":               "#2E8B4A",   # deeper green
    "inception1d":             "#EE77B8",   # light pink, distinct from red
    "hubert_ecg":              "#7040A8",   # deeper purple
    "resnet18_2d":             "#444444",   # dark gray
    "bilstm1d":                "#0B8FA6",   # deeper cyan
}

# Annotation offsets from PTB-XL point (in points)
label_offsets_pt = {
    "cnn_bilstm_transformer": (-58,  +26),
    "bilstm1d":               (+38,  +20),
    "1D_ViT":                 (+44,  -22),
    "inception1d":            (-60,  +22),
    "xresnet1d":              (-88,  +12),
    "hubert_ecg":             (+46,   +4),
    "resnet18_2d":            (-55,  +22),
}

# Visual jitter for near-identical CPSC points
visual_jitter = {
    ("cnn_bilstm_transformer", "CPSC2018"): (0.0, +0.007),
    ("bilstm1d",               "CPSC2018"): (0.0, -0.005),
}

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(11, 9))

# =========================================================
# SCATTER PLOT
# =========================================================

for model_key in model_order:
    mdata = models_data[model_key]
    color = model_colors[model_key]

    xs_raw = [mdata[d]["auroc_macro"] for d in datasets]
    ys_raw = [mdata[d]["f1_macro"]    for d in datasets]

    xs_plot = [xs_raw[i] + visual_jitter.get((model_key, d), (0, 0))[0]
               for i, d in enumerate(datasets)]
    ys_plot = [ys_raw[i] + visual_jitter.get((model_key, d), (0, 0))[1]
               for i, d in enumerate(datasets)]

    # Dashed connecting line
    ax.plot(xs_raw, ys_raw,
            color=color, linewidth=1.5, alpha=0.55, linestyle="--", zorder=1)

    # Scatter points
    for d, xv, yv in zip(datasets, xs_plot, ys_plot):
        ax.scatter(xv, yv,
                   marker=dataset_markers[d],
                   s=dataset_sizes_plot[d],
                   color=color,
                   edgecolors="black",
                   linewidth=0.5,
                   zorder=3)

    # Annotation near PTB-XL point
    ptb_x = mdata["PTB-XL fold10"]["auroc_macro"]
    ptb_y = mdata["PTB-XL fold10"]["f1_macro"]
    ox, oy = label_offsets_pt[model_key]
    ax.annotate(
        mdata["label"],
        xy=(ptb_x, ptb_y),
        xytext=(ox, oy),
        textcoords="offset points",
        fontsize=10, color=color, fontweight="bold",
        ha="center", va="center", zorder=4,
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.80),
    )

# =========================================================
# GOLD RINGS — best model per dataset (avg of AUROC & F1)
# =========================================================

for d in datasets:
    avg_vals = {m: (models_data[m][d]["auroc_macro"] + models_data[m][d]["f1_macro"]) / 2
                for m in model_order}
    best_m = max(avg_vals, key=avg_vals.get)
    xv = models_data[best_m][d]["auroc_macro"]
    yv = models_data[best_m][d]["f1_macro"]
    ax.scatter(xv, yv,
               marker=dataset_markers[d],
               s=dataset_sizes_plot[d] * 2.8,
               facecolors="none",
               edgecolors="gold",
               linewidth=2.5,
               zorder=2)

# =========================================================
# AXIS FORMATTING
# =========================================================

ax.set_xlabel("Macro AUROC", fontsize=13, fontweight="bold", labelpad=8)
ax.set_ylabel("Macro F1",    fontsize=13, fontweight="bold", labelpad=8)
ax.set_xlim(0.695, 0.975)
ax.set_ylim(0.34,  0.82)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "ECG Benchmark — Macro AUROC vs Macro F1\nacross Three Datasets",
    fontsize=14, fontweight="bold", pad=14,
)

# =========================================================
# GOLD RING LEGEND (bottom-right, small)
# =========================================================

ring_handle = Line2D(
    [0], [0], marker="o", color="w",
    markerfacecolor="none", markeredgecolor="gold",
    markeredgewidth=1.8, markersize=10,
    label="Best model per dataset",
)
leg_ring = ax.legend(
    handles=[ring_handle],
    loc="lower right", fontsize=11,
    labelcolor="black",
    framealpha=0.92, edgecolor="lightgray",
)
ax.add_artist(leg_ring)

# =========================================================
# LEGEND TABLE OVERLAY (top-left, compact, inside plot)
# =========================================================

# Box dimensions in axes-fraction coordinates
# lh sized so box hugs the last row (space_below = C - 0.022, C = 0.042 → ~0.020 pad)
lx0, ly0 = 0.01,  0.688
lw,  lh  = 0.282, 0.292

# White background
ax.add_patch(FancyBboxPatch(
    (lx0, ly0), lw, lh,
    boxstyle="square,pad=0.003",
    facecolor="white", edgecolor="lightgray", lw=0.8,
    transform=ax.transAxes, zorder=8,
))

# Column x-positions (axes fraction) — tighter spacing between marker columns
cx_name = lx0 + 0.012
cx_ptb  = lx0 + lw * 0.57
cx_cpsc = lx0 + lw * 0.71
cx_geo  = lx0 + lw * 0.85

n_rows  = len(model_order)
y_hdr   = ly0 + lh - 0.022
row_h_l = (lh - 0.042) / n_rows   # box hugs content, ~0.020 pad below last row

# Header row
ax.text(cx_name, y_hdr, "Model",
        fontsize=9, fontweight="bold", va="center", color="black",
        transform=ax.transAxes, zorder=9)
ax.text(cx_ptb,  y_hdr, "PTB-XL ",
        fontsize=7, fontweight="bold", va="center", ha="center", color="black",
        transform=ax.transAxes, zorder=9)
ax.text(cx_cpsc, y_hdr, "  CPSC ",
        fontsize=7, fontweight="bold", va="center", ha="center", color="black",
        transform=ax.transAxes, zorder=9)
ax.text(cx_geo,  y_hdr, "    Georgia ",
        fontsize=7, fontweight="bold", va="center", ha="center", color="black",
        transform=ax.transAxes, zorder=9)

# Separator line
sep_y = y_hdr - row_h_l * 0.58
ax.add_line(Line2D(
    [lx0 + 0.004, lx0 + lw - 0.004], [sep_y, sep_y],
    color="gray", lw=0.5, alpha=0.5,
    transform=ax.transAxes, zorder=9,
))

# Model rows
for i, k in enumerate(model_order):
    color = model_colors[k]
    y_row = y_hdr - (i + 1) * row_h_l

    ax.text(cx_name, y_row, models_data[k]["label"],
            fontsize=9, va="center", color=color, fontweight="bold",
            transform=ax.transAxes, zorder=9)
    ax.plot([cx_ptb],  [y_row], marker="*", ms=13,  color=color, mec="black", mew=0.4,
            linestyle="None", transform=ax.transAxes, zorder=10, clip_on=False)
    ax.plot([cx_cpsc], [y_row], marker="^", ms=9,   color=color, mec="black", mew=0.4,
            linestyle="None", transform=ax.transAxes, zorder=10, clip_on=False)
    ax.plot([cx_geo],  [y_row], marker="o", ms=9,   color=color, mec="black", mew=0.4,
            linestyle="None", transform=ax.transAxes, zorder=10, clip_on=False)

# =========================================================
# SAVE
# =========================================================

plt.tight_layout()
plt.savefig(output_dir / "benchmark_all.png", dpi=500, bbox_inches="tight")
print("Saved:", output_dir / "benchmark_all.png")
