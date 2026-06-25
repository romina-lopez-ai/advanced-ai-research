"""
COMP6011 Task 3 — Explainability Comparison: Attention vs Grad x Input vs SHAP
Student: Santiago Boxiga
Date: 2026-05-17

Side-by-side comparison of three XAI methods on multi-label ECG samples.
- Attention: class-agnostic (ViT CLS token, shown once)
- Gradient x Input: class-specific, full sample-level resolution
- SHAP: class-specific, patch-level resolution
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from vit_ecg_experiment import (
    CONFIG, DEVICE, ViTECG, bandpass_filter, CLASS_TO_IDX
)

SAMPLE_CASES = [
    {"ecg_id": 685, "true_labels": ["LBBB", "1dAVb"], "desc": "multi-label"},
    {"ecg_id": 7598, "true_labels": ["1dAVb", "RBBB"], "desc": "multi-label"},
]
OUTPUT_DIR = os.path.join(CONFIG["output_dir"], "explainability")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def load_model():
    model = ViTECG(
        num_leads=CONFIG["num_leads"],
        signal_length=CONFIG["signal_length"],
        patch_size=CONFIG["patch_size"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)
    model_path = os.path.join(CONFIG["output_dir"], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def load_ecg_sample(ecg_id):
    db = pd.read_csv(os.path.join(CONFIG["ptbxl_path"], "ptbxl_database.csv"),
                     index_col="ecg_id")
    row = db.loc[ecg_id]
    record_path = os.path.join(CONFIG["ptbxl_path"], row.filename_hr)
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal.T.astype(np.float64)

    sig = np.nan_to_num(sig, nan=0.0)
    sig = bandpass_filter(sig, fs=500)
    for lead in range(12):
        std = sig[lead].std()
        if std > 1e-6:
            sig[lead] = (sig[lead] - sig[lead].mean()) / std
        else:
            sig[lead] = 0.0
    return sig.astype(np.float32)


# ── Method 1: Attention (class-agnostic) ────────────────────────────────────

def get_attention_map(model, signal_tensor):
    with torch.no_grad():
        logits, attn_maps = model(signal_tensor, return_attention=True)
        probs = torch.softmax(logits, dim=1)

    last_attn = attn_maps[-1][0].cpu().numpy()
    if last_attn.ndim == 3:
        last_attn = last_attn.mean(axis=0)
    cls_attn = last_attn[0, 1:]

    attn_upsampled = np.repeat(cls_attn, CONFIG["patch_size"])
    if attn_upsampled.max() > 0:
        attn_upsampled = attn_upsampled / attn_upsampled.max()
    return attn_upsampled, probs


# ── Method 2: Gradient x Input (full resolution, class-specific) ────────────

def get_grad_x_input(model, signal_tensor, target_class):
    model.zero_grad()
    inp = signal_tensor.clone().detach().requires_grad_(True)

    logits = model(inp)
    logits[0, target_class].backward()

    grad = inp.grad[0].cpu().numpy()        # (12, 5000)
    signal_np = inp.detach().cpu().numpy()[0]
    saliency = np.abs(grad * signal_np)     # (12, 5000)

    # Average across leads for a single-channel heatmap
    saliency_avg = saliency.mean(axis=0)    # (5000,)
    if saliency_avg.max() > 0:
        saliency_avg = saliency_avg / saliency_avg.max()
    return saliency_avg, saliency


# ── Method 3: SHAP (patch-level, class-specific) ────────────────────────────

def get_shap_values(model, signal_tensor, target_class, n_samples=200):
    num_patches = CONFIG["signal_length"] // CONFIG["patch_size"]
    signal_np = signal_tensor.cpu().numpy()[0]
    baseline = np.zeros_like(signal_np)

    def predict_fn(masks):
        batch = []
        for mask in masks:
            x = baseline.copy()
            for p in range(num_patches):
                if mask[p] == 1:
                    s = p * CONFIG["patch_size"]
                    e = s + CONFIG["patch_size"]
                    x[:, s:e] = signal_np[:, s:e]
            batch.append(x)
        batch_tensor = torch.FloatTensor(np.array(batch)).to(DEVICE)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs[:, target_class].cpu().numpy()

    shap_values = np.zeros(num_patches)
    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        perm = rng.permutation(num_patches)
        mask_with = np.zeros(num_patches)
        mask_without = np.zeros(num_patches)

        for idx in perm:
            mask_with[:] = mask_without[:]
            mask_with[idx] = 1
            val_with = predict_fn(mask_with[np.newaxis, :])[0]
            val_without = predict_fn(mask_without[np.newaxis, :])[0]
            shap_values[idx] += (val_with - val_without)
            mask_without[idx] = 1

    shap_values /= n_samples
    shap_abs = np.abs(shap_values)
    if shap_abs.max() > 0:
        shap_abs = shap_abs / shap_abs.max()

    shap_upsampled = np.repeat(shap_abs, CONFIG["patch_size"])
    return shap_upsampled


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _overlay_heatmap(ax, signal_1d, heatmap, cmap_name):
    """Render signal with continuous colour-mapped heatmap overlay."""
    t = np.arange(len(signal_1d))
    ax.plot(t, signal_1d, color="black", linewidth=0.5, alpha=0.6, zorder=2)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=1)

    for i in range(len(t) - 1):
        val = heatmap[i]
        if val > 0.05:
            ax.axvspan(t[i], t[i + 1], color=cmap(norm(val)), alpha=val * 0.7, zorder=1)

    ax.set_xlim(0, len(signal_1d))


def plot_combined(signal, attention, grad_per_class, shap_per_class,
                  pred_class, pred_conf, true_labels, ecg_id):
    """
    Layout:
      Row 0: Attention (one column spanning all, since it's class-agnostic)
      Row 1..N: Grad x Input per target class
      Row N+1..2N: SHAP per target class
    Total rows = 1 + 2*n_classes, one column, Lead II.
    """
    n_classes = len(true_labels)
    n_rows = 1 + 2 * n_classes
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), sharex=True)

    lead_idx = 1  # Lead II
    row = 0

    # Attention (class-agnostic, single row)
    _overlay_heatmap(axes[row], signal[lead_idx], attention, "Reds")
    axes[row].set_ylabel("Attention\n(CLS token)", fontsize=9, fontweight="bold")
    axes[row].set_title("Class-agnostic", fontsize=10, style="italic")
    row += 1

    # Grad x Input per class
    for target_label in true_labels:
        _overlay_heatmap(axes[row], signal[lead_idx],
                         grad_per_class[target_label], "YlOrRd")
        axes[row].set_ylabel(f"Grad x Input\n({target_label})",
                             fontsize=9, fontweight="bold")
        row += 1

    # SHAP per class
    for target_label in true_labels:
        _overlay_heatmap(axes[row], signal[lead_idx],
                         shap_per_class[target_label], "BuPu")
        axes[row].set_ylabel(f"SHAP\n({target_label})",
                             fontsize=9, fontweight="bold")
        row += 1

    axes[-1].set_xlabel("Sample (500 Hz)", fontsize=10)
    true_str = " + ".join(true_labels)
    fig.suptitle(
        f"ECG #{ecg_id} (Lead II) | True: {true_str} | "
        f"Predicted: {pred_class} ({pred_conf:.1%})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"combined_ecg{ecg_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_leads(signal, attention, grad_per_class, grad_full_per_class,
                   shap_per_class, pred_class, pred_conf, true_labels, ecg_id):
    """12-lead figure: columns = [Attention, Grad(class1), Grad(class2), SHAP(class1), SHAP(class2)]."""
    n_classes = len(true_labels)
    n_cols = 1 + 2 * n_classes
    fig, axes = plt.subplots(12, n_cols, figsize=(5 * n_cols, 20), sharex=True)

    col_labels = ["Attention"]
    col_cmaps = ["Reds"]
    col_heatmaps = [attention]
    col_per_lead = [None]  # None means use the averaged heatmap for all leads

    for lbl in true_labels:
        col_labels.append(f"Grad x Input\n({lbl})")
        col_cmaps.append("YlOrRd")
        col_heatmaps.append(None)
        col_per_lead.append(grad_full_per_class[lbl])  # (12, 5000) per-lead saliency

    for lbl in true_labels:
        col_labels.append(f"SHAP ({lbl})")
        col_cmaps.append("BuPu")
        col_heatmaps.append(shap_per_class[lbl])
        col_per_lead.append(None)

    for col in range(n_cols):
        for row in range(12):
            ax = axes[row, col]

            # Use per-lead saliency if available, otherwise use averaged
            if col_per_lead[col] is not None:
                lead_heatmap = col_per_lead[col][row]
                if lead_heatmap.max() > 0:
                    lead_heatmap = lead_heatmap / lead_heatmap.max()
            else:
                lead_heatmap = col_heatmaps[col]

            _overlay_heatmap(ax, signal[row], lead_heatmap, col_cmaps[col])

            if col == 0:
                ax.set_ylabel(LEAD_NAMES[row], rotation=0, labelpad=20, fontsize=9)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=5)

    true_str = " + ".join(true_labels)
    fig.suptitle(
        f"ECG #{ecg_id} | True: {true_str} | Predicted: {pred_class} ({pred_conf:.1%})\n"
        f"Continuous heatmap intensity = attribution strength",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"all_leads_ecg{ecg_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model()

    for case in SAMPLE_CASES:
        ecg_id = case["ecg_id"]
        true_labels = case["true_labels"]
        print(f"\n{'='*60}")
        print(f"Processing ECG #{ecg_id} ({case['desc']}): {true_labels}")
        print(f"{'='*60}")

        signal = load_ecg_sample(ecg_id)
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)

        attn_map, probs = get_attention_map(model, signal_tensor)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = CONFIG["classes"][pred_idx]
        pred_conf = probs[0, pred_idx].item()
        print(f"  Prediction: {pred_class} ({pred_conf:.1%})")
        classes = CONFIG["classes"]
        prob_strs = [f"{classes[i]}={probs[0,i]:.3f}" for i in range(7)]
        print(f"  All probs: {prob_strs}")

        grad_per_class = {}
        grad_full_per_class = {}
        shap_per_class = {}

        for target_label in true_labels:
            target_idx = CLASS_TO_IDX[target_label]
            print(f"\n  --- Targeting class: {target_label} (idx={target_idx}) ---")

            sig_t = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)
            grad_avg, grad_full = get_grad_x_input(model, sig_t, target_idx)
            grad_per_class[target_label] = grad_avg
            grad_full_per_class[target_label] = grad_full
            print(f"  Grad x Input computed (5000-sample resolution)")

            shap_per_class[target_label] = get_shap_values(
                model, signal_tensor, target_idx, n_samples=200
            )
            print(f"  SHAP computed (patch-level)")

        plot_combined(signal, attn_map, grad_per_class, shap_per_class,
                      pred_class, pred_conf, true_labels, ecg_id)

        plot_all_leads(signal, attn_map, grad_per_class, grad_full_per_class,
                       shap_per_class, pred_class, pred_conf, true_labels, ecg_id)

    print(f"\nAll explainability figures saved to {OUTPUT_DIR}/")
