"""
HuBERT-ECG Small — Fine-tuning for 7-class ECG Classification
Foundation model : HuBERT-ECG Small (30.5 M params, hidden_size=512, CC BY-NC 4.0)
Input            : 12-lead ECG → bandpass (0.05-47 Hz) → downsample 500→100 Hz
                   → min-max per lead [-1,1] → flatten (12×1000) → (12000,)
Split            : PTB-XL folds 1-8 train | fold 9 val (threshold opt) | fold 10 test
Eval             : PTB-XL fold 10, Georgia (PhysioNet 2020), CPSC 2018

═══ KAGGLE SETUP ════════════════════════════════════════════════════════════════
Step 1 — Accelerator : GPU → T4 x1
Step 2 — Internet    : Settings → Internet → On (needed to download model)
Step 3 — Add datasets (+Add Data):
            bjoernjostein/ptbxl-electrocardiography-database
            physionet/georgia-12lead-ecg-challenge-database
            bjoernjostein/china-physiological-signal-challenge-in-2018
Step 4 — In a SEPARATE cell at the top of the notebook, run:
            !pip install -q git+https://github.com/Edoar-do/HuBERT-ECG.git wfdb
Step 5 — Run this script (or paste its contents into subsequent cells)

Expected runtime: ~1.5-2 h on Kaggle T4 GPU (including data loading)
Output directory: /kaggle/working/results_hubert_ecg_small/
════════════════════════════════════════════════════════════════════════════════
"""

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "experiment_name": "hubert_ecg_small_ptbxl_benchmark",
    "model":           "HuBERT-ECG-Small",
    "hf_model_id":     "Edoardo-BS/hubert-ecg-small",
    "hidden_size":     512,          # HuBERT-ECG Small hidden dim
    "src_fs":          500,          # raw PTB-XL sampling rate
    "model_fs":        100,          # HuBERT-ECG operates at 100 Hz
    "signal_length":   5000,         # 10 s at 500 Hz (raw)
    "model_len":       1000,         # 10 s at 100 Hz (after downsample)
    "num_leads":       12,
    "num_classes":     7,
    "classes":         ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "OTHERS"],
    "batch_size":      32,
    "epochs":          50,
    "backbone_lr":     1e-5,         # low LR for pretrained backbone
    "head_lr":         1e-3,         # higher LR for new classification head
    "weight_decay":    1e-4,
    "warmup_epochs":   3,
    "early_stopping_patience": 15,
    "confidence_threshold":    0.60,
    "dropout":         0.1,
    # ── Kaggle paths ─────────────────────────────────────────────────────────
    "ptbxl_path":  "/kaggle/input/ptbxl-electrocardiography-database",
    "georgia_path":"/kaggle/input/georgia-12lead-ecg-challenge-database",
    "cpsc_path":   "/kaggle/input/china-physiological-signal-challenge-in-2018",
    "out_dir":     "/kaggle/working/results_hubert_ecg_small",
    "cache_dir":   "/kaggle/working/cache_hubert_ecg",
}

import os
import ast
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, multilabel_confusion_matrix,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, sosfiltfilt, resample
from scipy.io import loadmat
from tqdm import tqdm

os.makedirs(CONFIG["out_dir"],   exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
NUM_WORKERS = 0 if sys.platform == "win32" else 2
print(f"Experiment : {CONFIG['experiment_name']}")
print(f"Device     : {DEVICE}")
print(f"Seed       : {SEED}")
print(f"PyTorch    : {torch.__version__}")
print(f"Output     : {CONFIG['out_dir']}")


# =============================================================================
# Label mappings  (identical to xresnet1d / inception1d scripts)
# =============================================================================
CLASS_ORDER  = CONFIG["classes"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}

SCP_TO_CLASS = {
    "NORM": "NORM",
    "AFIB": "AFIB",
    "AFLT": "AFLT",
    "1AVB": "1dAVb",
    "RBBB": "RBBB",  "CRBBB": "RBBB",  "IRBBB": "RBBB",
    "LBBB": "LBBB",  "CLBBB": "LBBB",  "ILBBB": "LBBB",
}

SNOMED_TO_CLASS = {
    "426783006": "NORM",
    "164889003": "AFIB",
    "164890007": "AFLT",
    "270492004": "1dAVb",
    "59118001":  "RBBB",
    "713427006": "RBBB",
    "713426002": "RBBB",
    "164909002": "LBBB",
    "251146004": "LBBB",
    "445118002": "LBBB",
    "284470004": "OTHERS",  "17338001":  "OTHERS",  "427172004": "OTHERS",
    "164884008": "OTHERS",  "429622005": "OTHERS",  "164931005": "OTHERS",
    "164930006": "OTHERS",  "164934002": "OTHERS",  "59931005":  "OTHERS",
    "164947007": "OTHERS",  "111975006": "OTHERS",  "698252002": "OTHERS",
    "426648003": "OTHERS",  "39732003":  "OTHERS",  "47665007":  "OTHERS",
    "251200008": "OTHERS",  "55827005":  "OTHERS",  "164873001": "OTHERS",
    "89792004":  "OTHERS",  "6374002":   "OTHERS",  "233917008": "OTHERS",
    "195042002": "OTHERS",  "54016002":  "OTHERS",  "27885002":  "OTHERS",
    "195060002": "OTHERS",  "426761007": "OTHERS",  "713422000": "OTHERS",
    "426995002": "OTHERS",  "10370003":  "OTHERS",  "164912004": "OTHERS",
    "17366009":  "OTHERS",  "67198005":  "OTHERS",  "426177001": "OTHERS",
    "427084000": "OTHERS",  "164865005": "OTHERS",  "57054005":  "OTHERS",
    "164917005": "OTHERS",  "11157007":  "OTHERS",  "428750005": "OTHERS",
    "425623009": "OTHERS",  "427393009": "OTHERS",  "425419005": "OTHERS",
    "67741000119109": "OTHERS",
}


def _scp_to_multilabel(scp_codes_dict):
    present = {c for c, lh in scp_codes_dict.items() if lh >= 100.0}
    y = np.zeros(7, dtype=np.float32)
    has_named = False
    for code in present:
        cls = SCP_TO_CLASS.get(code)
        if cls:
            y[CLASS_TO_IDX[cls]] = 1
            has_named = True
    if (present - set(SCP_TO_CLASS.keys())) or not has_named:
        y[CLASS_TO_IDX["OTHERS"]] = 1
    return y


def _snomed_multilabel_from_hea(hea_path):
    y = np.zeros(7, dtype=np.float32)
    try:
        with open(hea_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    codes_str = line.split(":", 1)[1].strip()
                    has_mapped = has_unmapped = False
                    for code in codes_str.split(","):
                        code = code.strip()
                        cls = SNOMED_TO_CLASS.get(code)
                        if cls:
                            y[CLASS_TO_IDX[cls]] = 1
                            has_mapped = True
                        else:
                            has_unmapped = True
                    if has_unmapped or not has_mapped:
                        y[CLASS_TO_IDX["OTHERS"]] = 1
    except Exception:
        pass
    return y


# =============================================================================
# Preprocessing — HuBERT-ECG specific pipeline
#   bandpass (0.05-47 Hz) → downsample to 100 Hz → min-max per lead [-1,1]
#   → flatten (12, 1000) → (12000,)
# =============================================================================
def _bandpass_hubert(signal, fs=500):
    sos = butter(4, [0.05, 47.0], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def _preprocess_for_hubert(sig_raw, src_fs=500, target_fs=100):
    """sig_raw: (12, T) float64 at src_fs. Returns (12000,) float32."""
    sig = np.nan_to_num(sig_raw, nan=0.0)
    sig = _bandpass_hubert(sig, fs=src_fs)
    n_target = int(sig.shape[-1] * target_fs / src_fs)
    sig = resample(sig, n_target, axis=-1)          # (12, 1000)
    for i in range(sig.shape[0]):
        mn, mx = sig[i].min(), sig[i].max()
        sig[i] = 2.0 * (sig[i] - mn) / (mx - mn + 1e-8) - 1.0
    return sig.reshape(-1).astype(np.float32)       # (12000,)


# =============================================================================
# Data Loading  (caches separate from xresnet1d — different preprocessing)
# =============================================================================
def _find_ptbxl_root(base):
    if os.path.exists(os.path.join(base, "ptbxl_database.csv")):
        return base
    for d in os.listdir(base):
        candidate = os.path.join(base, d)
        if os.path.isdir(candidate) and \
                os.path.exists(os.path.join(candidate, "ptbxl_database.csv")):
            return candidate
    raise FileNotFoundError(f"ptbxl_database.csv not found under {base}")


def load_ptbxl(data_path=None):
    if data_path is None:
        data_path = CONFIG["ptbxl_path"]
    data_path = _find_ptbxl_root(data_path)
    cache_path = os.path.join(CONFIG["cache_dir"], "ptbxl_hubert_100hz.npz")

    if os.path.exists(cache_path):
        print(f"  PTB-XL: loading from cache ({cache_path})")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels_ml"], cached["folds"]

    print("  PTB-XL: building HuBERT cache (bandpass+100Hz+minmax) — ~20-40 min…")
    db = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    signals, labels_ml, folds = [], [], []
    skipped = 0
    for _, row in tqdm(db.iterrows(), total=len(db), desc="PTB-XL"):
        record_path = os.path.join(data_path, row["filename_hr"])
        try:
            rec = wfdb.rdrecord(record_path)
            sig = rec.p_signal.T.astype(np.float64)  # (12, 5000)
            if sig.shape != (12, 5000):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        signals.append(_preprocess_for_hubert(sig))
        labels_ml.append(_scp_to_multilabel(row["scp_codes"]))
        folds.append(int(row["strat_fold"]))

    signals_arr = np.stack(signals)
    labels_arr  = np.stack(labels_ml)
    folds_arr   = np.array(folds, dtype=np.int32)
    print(f"  PTB-XL: {len(signals_arr)} records ({skipped} skipped). Saving cache…")
    np.savez_compressed(cache_path, signals=signals_arr, labels_ml=labels_arr, folds=folds_arr)
    return signals_arr, labels_arr, folds_arr


def load_georgia(data_path=None):
    if data_path is None:
        data_path = CONFIG["georgia_path"]
    cache_path = os.path.join(CONFIG["cache_dir"], "georgia_hubert_100hz.npz")

    if os.path.exists(cache_path):
        print(f"  Georgia: loading from cache ({cache_path})")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels_ml"]

    print("  Georgia: building HuBERT cache…")
    hea_files = sorted(glob.glob(os.path.join(data_path, "**", "*.hea"), recursive=True))
    signals, labels_ml = [], []
    skipped = 0
    for hea_path in tqdm(hea_files, desc="Georgia"):
        mat_path = hea_path.replace(".hea", ".mat")
        if not os.path.exists(mat_path):
            skipped += 1
            continue
        try:
            sig = loadmat(mat_path)["val"].astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if sig.shape[0] != 12:
            skipped += 1
            continue
        if sig.shape[1] < 5000:
            sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
        elif sig.shape[1] > 5000:
            sig = sig[:, :5000]
        signals.append(_preprocess_for_hubert(sig))
        labels_ml.append(_snomed_multilabel_from_hea(hea_path))

    signals_arr = np.stack(signals)
    labels_arr  = np.stack(labels_ml)
    print(f"  Georgia: {len(signals_arr)} records ({skipped} skipped). Saving cache…")
    np.savez_compressed(cache_path, signals=signals_arr, labels_ml=labels_arr)
    return signals_arr, labels_arr


def load_cpsc(data_path=None):
    if data_path is None:
        data_path = CONFIG["cpsc_path"]
    cache_path = os.path.join(CONFIG["cache_dir"], "cpsc_hubert_100hz.npz")

    if os.path.exists(cache_path):
        print(f"  CPSC: loading from cache ({cache_path})")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels_ml"]

    print("  CPSC: building HuBERT cache…")
    hea_files = sorted(glob.glob(os.path.join(data_path, "**", "*.hea"), recursive=True))
    signals, labels_ml = [], []
    skipped = 0
    for hea_path in tqdm(hea_files, desc="CPSC 2018"):
        mat_path = hea_path.replace(".hea", ".mat")
        if not os.path.exists(mat_path):
            skipped += 1
            continue
        try:
            sig = loadmat(mat_path)["val"].astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if sig.shape[0] != 12:
            skipped += 1
            continue
        if sig.shape[1] < 5000:
            sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
        elif sig.shape[1] > 5000:
            sig = sig[:, :5000]
        signals.append(_preprocess_for_hubert(sig))
        labels_ml.append(_snomed_multilabel_from_hea(hea_path))

    signals_arr = np.stack(signals)
    labels_arr  = np.stack(labels_ml)
    print(f"  CPSC: {len(signals_arr)} records ({skipped} skipped). Saving cache…")
    np.savez_compressed(cache_path, signals=signals_arr, labels_ml=labels_arr)
    return signals_arr, labels_arr


# =============================================================================
# Dataset
# =============================================================================
class ECGDataset(Dataset):
    def __init__(self, signals, labels_ml, augment=False):
        self.signals = torch.from_numpy(signals)   # (N, 12000) float32
        self.labels  = torch.from_numpy(labels_ml).float()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.signals[idx].clone()    # (12000,)
        if self.augment:
            sig2d = sig.reshape(12, CONFIG["model_len"])    # (12, 1000)
            sig2d = sig2d + torch.randn_like(sig2d) * 0.05
            shift = torch.randint(-50, 51, (1,)).item()     # ±0.5 s at 100 Hz
            sig2d = torch.roll(sig2d, shift, dims=-1)
            sig = sig2d.reshape(-1)
        return sig, self.labels[idx]


# =============================================================================
# Model — pretrained HuBERT-ECG backbone + linear classification head
# =============================================================================
class HuBERTECGClassifier(nn.Module):
    def __init__(self, hf_model_id, hidden_size=512, num_classes=7, dropout=0.1):
        super().__init__()
        try:
            import hubert_ecg  # registers custom AutoModel types
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(hf_model_id)
        except ImportError:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_values):
        # input_values: (B, 12000)
        out    = self.backbone(input_values=input_values)
        hidden = out.last_hidden_state          # (B, T_reduced, 512)
        hidden = hidden.mean(dim=1)             # mean-pool over time → (B, 512)
        return self.classifier(self.dropout(hidden))   # (B, 7) logits


# =============================================================================
# Threshold optimization
# =============================================================================
def optimize_thresholds(y_true_ml, y_probs, class_names):
    thresholds = np.arange(0.05, 0.80, 0.01)
    best = {}
    for j, cls in enumerate(class_names):
        if y_true_ml[:, j].sum() == 0:
            best[cls] = 0.5
            continue
        best_f1, best_t = 0.0, 0.5
        col_true = y_true_ml[:, j]
        col_prob = y_probs[:, j]
        for t in thresholds:
            pred = (col_prob >= t).astype(int)
            tp = ((pred == 1) & (col_true == 1)).sum()
            fp = ((pred == 1) & (col_true == 0)).sum()
            fn = ((pred == 0) & (col_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[cls] = round(float(best_t), 2)
    return best


# =============================================================================
# Training
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, scheduler, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for signals, labels in tqdm(loader, desc="  Train", leave=False):
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(signals)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_preds.append((torch.sigmoid(logits.float()) >= 0.5).int().cpu().numpy())
        all_labels.append(labels.cpu().numpy().astype(int))
    scheduler.step()
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)


def validate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            with autocast():
                logits = model(signals)
            total_loss += criterion(logits.float(), labels).item()
            all_preds.append((torch.sigmoid(logits.float()) >= 0.5).int().cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(int))
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro", zero_division=0)


# =============================================================================
# Evaluation
# =============================================================================
def plot_confusion_matrices(y_true_ml, y_pred, dataset_name, classes):
    mcm = multilabel_confusion_matrix(y_true_ml, y_pred)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for j, cls in enumerate(classes):
        ax = axes[j]
        tn, fp, fn, tp = mcm[j].ravel()
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["True Neg", "True Pos"])
        ax.set_title(f"{cls}  (TP={tp} FN={fn})", fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(2):
            for k in range(2):
                ax.text(k, i, f"{cm[i,k]:,}", ha="center", va="center", fontsize=11,
                        color="white" if cm[i, k] > thresh else "black")
    axes[-1].axis("off")
    plt.suptitle(f"Confusion Matrices — {CONFIG['model']} — {dataset_name}", fontsize=13)
    plt.tight_layout()
    fname = f"confusion_matrix_{dataset_name.lower().replace(' ','_')}.png"
    plt.savefig(os.path.join(CONFIG["out_dir"], fname), dpi=150, bbox_inches="tight")
    plt.close()


def get_probs(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for signals, _ in loader:
            with autocast():
                logits = model(signals.to(DEVICE))
            all_probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate(model, loader, dataset_name, y_true_ml, thresholds, flag_threshold=0.60):
    probs   = get_probs(model, loader)
    n       = len(probs)
    classes = CLASS_ORDER

    y_pred = np.zeros_like(y_true_ml, dtype=int)
    for j, cls in enumerate(classes):
        y_pred[:, j] = (probs[:, j] >= thresholds[cls]).astype(int)

    auroc_per, prauc_per = [], []
    for j in range(len(classes)):
        if y_true_ml[:, j].sum() == 0:
            auroc_per.append(float("nan"))
            prauc_per.append(float("nan"))
        else:
            try:
                auroc_per.append(roc_auc_score(y_true_ml[:, j], probs[:, j]))
            except ValueError:
                auroc_per.append(float("nan"))
            try:
                prauc_per.append(average_precision_score(y_true_ml[:, j], probs[:, j]))
            except ValueError:
                prauc_per.append(float("nan"))

    valid_aurocs = [v for v in auroc_per if not np.isnan(v)]
    valid_praucs = [v for v in prauc_per if not np.isnan(v)]
    auroc_macro  = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    prauc_macro  = float(np.mean(valid_praucs)) if valid_praucs else float("nan")

    f1_macro = f1_score(y_true_ml, y_pred, average="macro", zero_division=0)
    f1_per   = f1_score(y_true_ml, y_pred, average=None,    zero_division=0)

    y_pred_05 = (probs >= 0.5).astype(int)
    f1_at_05  = f1_score(y_true_ml, y_pred_05, average="macro", zero_division=0)

    mcm = multilabel_confusion_matrix(y_true_ml, y_pred)
    sensitivity, specificity = [], []
    for j in range(len(classes)):
        tn, fp, fn, tp = mcm[j].ravel()
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else float("nan"))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))

    max_prob  = probs.max(axis=1)
    flagged   = max_prob < flag_threshold
    flag_rate = flagged.sum() / n * 100

    print(f"\n{'='*74}")
    print(f"  {dataset_name}   N={n}   (multilabel, per-class thresholds)")
    print(f"{'='*74}")
    print(f"  Macro AUROC  : {auroc_macro:.4f}")
    print(f"  Macro PR-AUC : {prauc_macro:.4f}  (Precision-Recall, better for rare classes)")
    print(f"  Macro F1     : {f1_macro:.4f}  (per-class optimal thresholds)")
    print(f"  Macro F1@0.5 : {f1_at_05:.4f}  (fixed threshold 0.5, no tuning)")
    print(f"  Low-conf flag (<{flag_threshold:.2f}): {flag_rate:.1f}%  ({int(flagged.sum())}/{n})")
    print(f"\n  {'Class':<8} {'Thresh':>7} {'AUROC':>7} {'PR-AUC':>8} {'F1':>7} "
          f"{'Sens':>7} {'Spec':>7} {'Support':>8}")
    print(f"  {'-'*70}")
    for j, cls in enumerate(classes):
        support   = int(y_true_ml[:, j].sum())
        auroc_str = f"{auroc_per[j]:7.4f}" if not np.isnan(auroc_per[j]) else "    nan"
        prauc_str = f"{prauc_per[j]:8.4f}" if not np.isnan(prauc_per[j]) else "     nan"
        print(f"  {cls:<8} {thresholds[cls]:7.2f} {auroc_str} {prauc_str} {f1_per[j]:7.4f} "
              f"{sensitivity[j]:7.4f} {specificity[j]:7.4f} {support:8d}")
    print(f"{'='*74}")

    per_class = {}
    for j, cls in enumerate(classes):
        tn, fp, fn, tp = mcm[j].ravel()
        per_class[cls] = {
            "threshold":    thresholds[cls],
            "auroc":        round(float(auroc_per[j]), 4) if not np.isnan(auroc_per[j]) else None,
            "prauc":        round(float(prauc_per[j]), 4) if not np.isnan(prauc_per[j]) else None,
            "f1":           round(float(f1_per[j]), 4),
            "sensitivity":  round(float(sensitivity[j]), 4) if not np.isnan(sensitivity[j]) else None,
            "specificity":  round(float(specificity[j]), 4) if not np.isnan(specificity[j]) else None,
            "support":      int(y_true_ml[:, j].sum()),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    result = {
        "dataset":          dataset_name,
        "model":            CONFIG["model"],
        "evaluation":       "multilabel_per_class_threshold",
        "macro_auroc":      round(float(auroc_macro), 4),
        "macro_prauc":      round(float(prauc_macro), 4),
        "macro_f1":         round(float(f1_macro), 4),
        "macro_f1_at_05":   round(float(f1_at_05), 4),
        "thresholds":       thresholds,
        "per_class":        per_class,
        "low_confidence_flagging": {
            "threshold":        flag_threshold,
            "flagged_count":    int(flagged.sum()),
            "total_count":      n,
            "flagged_rate_pct": round(float(flag_rate), 2),
        },
    }

    plot_confusion_matrices(y_true_ml, y_pred, dataset_name, classes)

    json_path = os.path.join(CONFIG["out_dir"],
                             f"results_{dataset_name.lower().replace(' ','_')}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")
    return result


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # ── Load PTB-XL ──────────────────────────────────────────────────────────
    print("\n=== Loading PTB-XL (→ 100 Hz for HuBERT) ===")
    t0 = time.time()
    signals, labels_ml, folds = load_ptbxl()
    print(f"  {len(signals)} records in {time.time()-t0:.1f}s  shape={signals.shape}")

    train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
    val_mask   = folds == 9
    test_mask  = folds == 10

    train_labels = labels_ml[train_mask]
    val_labels   = labels_ml[val_mask]
    test_labels  = labels_ml[test_mask]

    print("  Class distribution (train, multilabel):")
    for i, cls in enumerate(CLASS_ORDER):
        print(f"    {cls}: {int(train_labels[:, i].sum())}")

    train_ds = ECGDataset(signals[train_mask], train_labels, augment=True)
    val_ds   = ECGDataset(signals[val_mask],   val_labels)
    test_ds  = ECGDataset(signals[test_mask],  test_labels)
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── pos_weight for class imbalance ───────────────────────────────────────
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.FloatTensor(neg_counts / (pos_counts + 1e-6)).to(DEVICE)
    print(f"  pos_weight: {pos_weight.cpu().numpy().round(2)}")

    # ── Build model ──────────────────────────────────────────────────────────
    print(f"\n=== Loading HuBERT-ECG Small from {CONFIG['hf_model_id']} ===")
    model = HuBERTECGClassifier(
        hf_model_id=CONFIG["hf_model_id"],
        hidden_size=CONFIG["hidden_size"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Total parameters    : {n_params:,}")
    print(f"  Backbone parameters : {n_backbone:,}  (pretrained, lower LR)")
    print(f"  Head parameters     : {n_params - n_backbone:,}  (random init, higher LR)")

    # ── Optimizer: different LRs for backbone vs head ────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),
         "lr": CONFIG["backbone_lr"]},
        {"params": list(model.dropout.parameters()) + list(model.classifier.parameters()),
         "lr": CONFIG["head_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    warmup_epochs = CONFIG["warmup_epochs"]
    total_epochs  = CONFIG["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()

    model_path = os.path.join(CONFIG["out_dir"], "best_model.pt")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n=== Fine-tuning ({total_epochs} epochs max, early stopping "
          f"patience={CONFIG['early_stopping_patience']}) ===")
    best_val_f1  = 0.0
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(total_epochs):
        t_ep = time.time()
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
        vl_loss, vl_f1 = validate(model, val_loader, criterion)
        elapsed = time.time() - t_ep

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train Loss {tr_loss:.4f} F1 {tr_f1:.4f} | "
              f"Val Loss {vl_loss:.4f} F1 {vl_f1:.4f} | {elapsed:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            patience_ctr = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Best val F1 {best_val_f1:.4f}  saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (patience={CONFIG['early_stopping_patience']})")
                break

    # Training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Training Curves — {CONFIG['model']}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["out_dir"], "training_curves.png"), dpi=150)
    plt.close()

    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"\nBest model loaded from {model_path}")

    # ── Threshold optimization on val fold 9 ─────────────────────────────────
    print("\n=== Optimizing thresholds on validation set (fold 9) ===")
    val_probs  = get_probs(model, val_loader)
    thresholds = optimize_thresholds(val_labels, val_probs, CLASS_ORDER)
    print(f"  Optimized thresholds: {thresholds}")

    # ── Evaluate PTB-XL fold 10 ───────────────────────────────────────────────
    print("\n=== PTB-XL Test (fold 10) ===")
    ptbxl_results = evaluate(model, test_loader, "PTB-XL", test_labels, thresholds)

    # ── Evaluate Georgia ─────────────────────────────────────────────────────
    print("\n=== Loading Georgia ===")
    t0 = time.time()
    georgia_signals, georgia_labels_ml = load_georgia()
    print(f"  {len(georgia_signals)} records in {time.time()-t0:.1f}s")
    georgia_loader = DataLoader(
        ECGDataset(georgia_signals, georgia_labels_ml),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("\n=== Georgia Evaluation ===")
    georgia_results = evaluate(model, georgia_loader, "Georgia", georgia_labels_ml, thresholds)

    # ── Evaluate CPSC 2018 ────────────────────────────────────────────────────
    print("\n=== Loading CPSC 2018 ===")
    t0 = time.time()
    cpsc_signals, cpsc_labels_ml = load_cpsc()
    print(f"  {len(cpsc_signals)} records in {time.time()-t0:.1f}s")
    cpsc_loader = DataLoader(
        ECGDataset(cpsc_signals, cpsc_labels_ml),
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("\n=== CPSC 2018 Evaluation ===")
    cpsc_results = evaluate(model, cpsc_loader, "CPSC2018", cpsc_labels_ml, thresholds)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Experiment Complete ===")
    print(f"  PTB-XL  AUROC={ptbxl_results['macro_auroc']:.4f}  "
          f"PR-AUC={ptbxl_results['macro_prauc']:.4f}  F1={ptbxl_results['macro_f1']:.4f}")
    print(f"  Georgia AUROC={georgia_results['macro_auroc']:.4f}  "
          f"PR-AUC={georgia_results['macro_prauc']:.4f}  F1={georgia_results['macro_f1']:.4f}")
    print(f"  CPSC    AUROC={cpsc_results['macro_auroc']:.4f}  "
          f"PR-AUC={cpsc_results['macro_prauc']:.4f}  F1={cpsc_results['macro_f1']:.4f}")

    summary = {
        "experiment":  CONFIG["experiment_name"],
        "model":       CONFIG["model"],
        "hf_model_id": CONFIG["hf_model_id"],
        "threshold":   "per_class_optimized",
        "optimized_thresholds": thresholds,
        "flag_threshold": CONFIG["confidence_threshold"],
        "architecture": {
            "backbone":      CONFIG["hf_model_id"],
            "hidden_size":   CONFIG["hidden_size"],
            "model_fs":      CONFIG["model_fs"],
            "input_length":  CONFIG["num_leads"] * CONFIG["model_len"],
            "total_params":  n_params,
            "backbone_params": n_backbone,
        },
        "training": {
            "backbone_lr":  CONFIG["backbone_lr"],
            "head_lr":      CONFIG["head_lr"],
            "weight_decay": CONFIG["weight_decay"],
            "batch_size":   CONFIG["batch_size"],
            "best_val_f1":  round(float(best_val_f1), 4),
        },
        "models": {
            CONFIG["model"]: {
                "PTB-XL fold10": {
                    "n":             ptbxl_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   ptbxl_results["macro_auroc"],
                    "prauc_macro":   ptbxl_results["macro_prauc"],
                    "f1_macro":      ptbxl_results["macro_f1"],
                    "f1_at_05":      ptbxl_results["macro_f1_at_05"],
                    "flag_rate_pct": ptbxl_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    ptbxl_results["low_confidence_flagging"]["flagged_count"],
                },
                "Georgia": {
                    "n":             georgia_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   georgia_results["macro_auroc"],
                    "prauc_macro":   georgia_results["macro_prauc"],
                    "f1_macro":      georgia_results["macro_f1"],
                    "f1_at_05":      georgia_results["macro_f1_at_05"],
                    "flag_rate_pct": georgia_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    georgia_results["low_confidence_flagging"]["flagged_count"],
                },
                "CPSC2018": {
                    "n":             cpsc_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   cpsc_results["macro_auroc"],
                    "prauc_macro":   cpsc_results["macro_prauc"],
                    "f1_macro":      cpsc_results["macro_f1"],
                    "f1_at_05":      cpsc_results["macro_f1_at_05"],
                    "flag_rate_pct": cpsc_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    cpsc_results["low_confidence_flagging"]["flagged_count"],
                },
            }
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_path = os.path.join(CONFIG["out_dir"], "results_summary_hubert_ecg_small.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
