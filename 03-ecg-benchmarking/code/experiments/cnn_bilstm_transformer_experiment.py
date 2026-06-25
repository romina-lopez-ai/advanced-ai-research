"""
COMP6011 Research Task 3 -- CNN-BiLSTM-Transformer for ECG Classification
Student: Santiago Boxiga
Date: 2026-05-18
Description: Hybrid CNN-BiLSTM-Transformer trained on PTB-XL, evaluated on PTB-XL, Georgia, CPSC.
  CNN extracts local morphology (QRS, ST segments),
  BiLSTM captures rhythm dependencies (e.g. AFIB),
  Transformer attends to diagnostically relevant segments.
"""

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "experiment_name": "cnn_bilstm_transformer_ecg_ptbxl",
    "model": "CNN_BiLSTM_Transformer",
    "dataset": "PTB-XL",
    "sampling_rate": 500,
    "signal_length": 5000,
    "num_leads": 12,
    "num_classes": 7,
    "classes": ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "OTHERS"],
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 5e-4,
    "weight_decay": 3e-2,
    "confidence_threshold": 0.60,
    "warmup_epochs": 5,
    # CNN
    "cnn_channels": [48, 96, 192],
    "cnn_kernel_size": 7,
    # BiLSTM
    "lstm_hidden": 96,
    "lstm_layers": 1,
    "lstm_dropout": 0.0,
    # Transformer
    "tf_heads": 4,
    "tf_layers": 2,
    "tf_dropout": 0.3,
    # Paths
    "ptbxl_path": "data/ptbxl",
    "georgia_path": "data/georgia",
    "cpsc_path": "data/cpsc",
    "output_dir": "code/experiments/results_cnn_bilstm_transformer",
    "skip_training": False,
}

import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    multilabel_confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat
import time
import json
from tqdm import tqdm

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print(f"Experiment: {CONFIG['experiment_name']}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# =============================================================================
# SCP-ECG to 7-class mapping (PTB-XL)
# =============================================================================
SCP_TO_CLASS = {
    "NORM": "NORM",
    "AFIB": "AFIB",
    "AFLT": "AFLT",
    "1AVB": "1dAVb",
    "RBBB": "RBBB",
    "CRBBB": "RBBB",
    "IRBBB": "RBBB",
    "LBBB": "LBBB",
    "CLBBB": "LBBB",
    "ILBBB": "LBBB",
}

CLASS_TO_IDX = {c: i for i, c in enumerate(CONFIG["classes"])}

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
    "284470004": "OTHERS",
    "17338001":  "OTHERS",
    "427172004": "OTHERS",
    "164884008": "OTHERS",
    "429622005": "OTHERS",
    "164931005": "OTHERS",
    "164930006": "OTHERS",
    "164934002": "OTHERS",
    "59931005":  "OTHERS",
    "164947007": "OTHERS",
    "111975006": "OTHERS",
    "698252002": "OTHERS",
    "426648003": "OTHERS",
    "39732003":  "OTHERS",
    "47665007":  "OTHERS",
    "251200008": "OTHERS",
    "55827005":  "OTHERS",
    "164873001": "OTHERS",
    "89792004":  "OTHERS",
    "6374002":   "OTHERS",
    "233917008": "OTHERS",
    "195042002": "OTHERS",
    "54016002":  "OTHERS",
    "27885002":  "OTHERS",
    "195060002": "OTHERS",
    "426761007": "OTHERS",
    "713422000": "OTHERS",
    "426995002": "OTHERS",
    "10370003":  "OTHERS",
    "164912004": "OTHERS",
    "17366009":  "OTHERS",
    "67198005":  "OTHERS",
    "426177001": "OTHERS",
    "427084000": "OTHERS",
    "164865005": "OTHERS",
    "57054005":  "OTHERS",
    "164917005": "OTHERS",
    "11157007":  "OTHERS",
    "428750005": "OTHERS",
    "425623009": "OTHERS",
    "427393009": "OTHERS",
    "425419005": "OTHERS",
    "67741000119109": "OTHERS",
}


# =============================================================================
# Multilabel ground truth builders
# =============================================================================
def _scp_to_multilabel(scp_codes_dict):
    present = {c for c, lh in scp_codes_dict.items() if lh >= 100.0}
    y = np.zeros(7, dtype=np.float32)
    has_named = False
    for code in present:
        cls = SCP_TO_CLASS.get(code)
        if cls:
            y[CLASS_TO_IDX[cls]] = 1
            has_named = True
    others_codes = present - set(SCP_TO_CLASS.keys())
    if others_codes or not has_named:
        y[CLASS_TO_IDX["OTHERS"]] = 1
    return y


def ptbxl_multilabel_from_csv(data_path, folds=None):
    cache_path = os.path.join(data_path, "ptbxl_multilabel_cached.npz")
    cached_sl = np.load(os.path.join(data_path, "ptbxl_cached.npz"))
    cached_folds = cached_sl["folds"]
    cached_labels = cached_sl["labels"]

    if os.path.exists(cache_path):
        ml = np.load(cache_path)
        labels_ml = ml["labels_ml"]
        if len(labels_ml) == len(cached_folds):
            if folds is not None:
                mask = np.isin(cached_folds, folds)
                return labels_ml[mask]
            return labels_ml

    db = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    all_ml, all_sl, all_folds = [], [], []
    for _, row in db.iterrows():
        codes = row.scp_codes
        ml_vec = _scp_to_multilabel(codes)
        mapped = set()
        for code, conf in codes.items():
            if conf >= 100.0 and code in SCP_TO_CLASS:
                mapped.add(SCP_TO_CLASS[code])
        if not mapped:
            sl = CLASS_TO_IDX["OTHERS"]
        else:
            sl = CLASS_TO_IDX["OTHERS"]
            for p in ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "NORM"]:
                if p in mapped:
                    sl = CLASS_TO_IDX[p]
                    break
        all_ml.append(ml_vec)
        all_sl.append(sl)
        all_folds.append(row.strat_fold)

    all_ml = np.stack(all_ml)
    all_sl = np.array(all_sl)
    all_folds_arr = np.array(all_folds)

    labels_ml = np.zeros((len(cached_folds), 7), dtype=np.float32)
    cache_idx = 0
    for i in range(len(all_ml)):
        if cache_idx >= len(cached_folds):
            break
        if all_folds_arr[i] == cached_folds[cache_idx] and all_sl[i] == cached_labels[cache_idx]:
            labels_ml[cache_idx] = all_ml[i]
            cache_idx += 1

    if cache_idx != len(cached_folds):
        print(f"  [warn] multilabel alignment matched {cache_idx}/{len(cached_folds)} records")

    np.savez(cache_path, labels_ml=labels_ml)

    if folds is not None:
        mask = np.isin(cached_folds, folds)
        return labels_ml[mask]
    return labels_ml


def snomed_multilabel_from_hea(hea_path):
    y = np.zeros(7, dtype=np.float32)
    try:
        with open(hea_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    codes_str = line.split(":", 1)[1].strip()
                    has_any_mapping = False
                    has_unmapped = False
                    for code in codes_str.split(","):
                        code = code.strip()
                        cls = SNOMED_TO_CLASS.get(code)
                        if cls:
                            y[CLASS_TO_IDX[cls]] = 1
                            has_any_mapping = True
                        else:
                            has_unmapped = True
                    if has_unmapped or not has_any_mapping:
                        y[CLASS_TO_IDX["OTHERS"]] = 1
    except Exception:
        pass
    return y


def georgia_multilabel(data_path):
    cache_path = os.path.join(data_path, "georgia_multilabel_cached.npz")
    cached_sl = np.load(os.path.join(data_path, "georgia_cached.npz"))
    n_cached = len(cached_sl["labels"])

    if os.path.exists(cache_path):
        ml = np.load(cache_path)
        if len(ml["labels_ml"]) == n_cached:
            return ml["labels_ml"]

    db = pd.read_csv(os.path.join(data_path, "georgia_database.csv"))
    all_ml, all_sl = [], []
    for _, row in db.iterrows():
        ecg_id = row.ecg_id
        num = int(ecg_id[1:])
        group = (num - 1) // 999 + 1
        hea_path = os.path.join(data_path, f"g{group}", ecg_id + ".hea")
        mat_path = os.path.join(data_path, f"g{group}", ecg_id + ".mat")
        if not os.path.exists(mat_path):
            continue
        try:
            sig = loadmat(mat_path)["val"]
            if sig.shape[0] != 12:
                continue
        except Exception:
            continue
        ml_vec = snomed_multilabel_from_hea(hea_path)
        all_ml.append(ml_vec)
        all_sl.append(CLASS_TO_IDX[row.primary_class])

    all_ml = np.stack(all_ml)
    all_sl = np.array(all_sl)
    cached_labels = cached_sl["labels"]

    labels_ml = np.zeros((n_cached, 7), dtype=np.float32)
    cache_idx = 0
    for i in range(len(all_ml)):
        if cache_idx >= n_cached:
            break
        if all_sl[i] == cached_labels[cache_idx]:
            labels_ml[cache_idx] = all_ml[i]
            cache_idx += 1

    if cache_idx != n_cached:
        print(f"  [warn] Georgia multilabel alignment matched {cache_idx}/{n_cached}")

    np.savez(cache_path, labels_ml=labels_ml)
    return labels_ml


def cpsc_multilabel(data_path):
    cache_path = os.path.join(data_path, "cpsc_multilabel_cached.npz")
    cached_sl = np.load(os.path.join(data_path, "cpsc_cached.npz"))
    n_cached = len(cached_sl["labels"])

    if os.path.exists(cache_path):
        ml = np.load(cache_path)
        if len(ml["labels_ml"]) == n_cached:
            return ml["labels_ml"]

    hea_files = sorted([f for f in os.listdir(data_path) if f.endswith(".hea")])
    all_ml, all_sl = [], []
    for hea_file in hea_files:
        rec_id = hea_file.replace(".hea", "")
        record_path = os.path.join(data_path, rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal.T
        except Exception:
            continue
        if sig.shape[0] != 12:
            continue
        ml_vec = snomed_multilabel_from_hea(os.path.join(data_path, hea_file))
        all_ml.append(ml_vec)

        snomed_codes = []
        with open(os.path.join(data_path, hea_file), "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    codes_str = line.split(":", 1)[1].strip()
                    snomed_codes = [c.strip() for c in codes_str.split(",")]
        mapped = set()
        for code in snomed_codes:
            if code in SNOMED_TO_CLASS:
                mapped.add(SNOMED_TO_CLASS[code])
        if not mapped:
            sl = CLASS_TO_IDX["OTHERS"]
        else:
            sl = CLASS_TO_IDX["OTHERS"]
            for p in ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "NORM"]:
                if p in mapped:
                    sl = CLASS_TO_IDX[p]
                    break
        all_sl.append(sl)

    all_ml = np.stack(all_ml)
    all_sl = np.array(all_sl)
    cached_labels = cached_sl["labels"]

    labels_ml = np.zeros((n_cached, 7), dtype=np.float32)
    cache_idx = 0
    for i in range(len(all_ml)):
        if cache_idx >= n_cached:
            break
        if all_sl[i] == cached_labels[cache_idx]:
            labels_ml[cache_idx] = all_ml[i]
            cache_idx += 1

    if cache_idx != n_cached:
        print(f"  [warn] CPSC multilabel alignment matched {cache_idx}/{n_cached}")

    np.savez(cache_path, labels_ml=labels_ml)
    return labels_ml


# =============================================================================
# Per-class threshold optimization
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
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[cls] = round(float(best_t), 2)
    return best


# =============================================================================
# Data Loading
# =============================================================================
def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500, order=3):
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def load_ptbxl(data_path, sampling_rate=500):
    cache_path = os.path.join(data_path, "ptbxl_cached.npz")
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels"], cached["folds"]

    print("No cache found, loading from raw files (this takes ~20-40 min on first run)...")
    db = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    signals, labels, folds = [], [], []
    total = len(db)

    for idx, row in tqdm(db.iterrows(), total=total, desc="Loading PTB-XL"):
        mapped = set()
        for code, conf in row.scp_codes.items():
            if conf >= 100.0 and code in SCP_TO_CLASS:
                mapped.add(SCP_TO_CLASS[code])
        if not mapped:
            label = "OTHERS"
        else:
            priority = ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "NORM"]
            label = "OTHERS"
            for p in priority:
                if p in mapped:
                    label = p
                    break

        record_path = os.path.join(data_path, row.filename_hr)
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal.T
            if sig.shape != (12, 5000):
                continue
        except Exception:
            continue

        sig = np.nan_to_num(sig, nan=0.0)
        sig = bandpass_filter(sig, fs=sampling_rate)
        for lead in range(12):
            std = sig[lead].std()
            if std > 1e-6:
                sig[lead] = (sig[lead] - sig[lead].mean()) / std
            else:
                sig[lead] = 0.0

        signals.append(sig.astype(np.float32))
        labels.append(CLASS_TO_IDX[label])
        folds.append(row.strat_fold)

    signals = np.array(signals)
    labels = np.array(labels)
    folds = np.array(folds)

    print(f"Saving cache to {cache_path}...")
    np.savez(cache_path, signals=signals, labels=labels, folds=folds)
    return signals, labels, folds


def load_georgia(data_path):
    cache_path = os.path.join(data_path, "georgia_cached.npz")
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels"]

    print("No cache found, loading from raw files...")
    db = pd.read_csv(os.path.join(data_path, "georgia_database.csv"))

    signals, labels = [], []
    total = len(db)

    for _, row in tqdm(db.iterrows(), total=total, desc="Loading Georgia"):
        ecg_id = row.ecg_id
        num = int(ecg_id[1:])
        group = (num - 1) // 999 + 1
        record_path = os.path.join(data_path, f"g{group}", ecg_id)

        try:
            mat = loadmat(record_path + ".mat")
            sig = mat["val"].astype(np.float64)
        except Exception:
            continue
        if sig.shape[0] != 12:
            continue

        if sig.shape[1] < 5000:
            sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
        elif sig.shape[1] > 5000:
            sig = sig[:, :5000]

        sig = np.nan_to_num(sig, nan=0.0)
        sig = bandpass_filter(sig, fs=500)
        for lead in range(12):
            std = sig[lead].std()
            if std > 1e-6:
                sig[lead] = (sig[lead] - sig[lead].mean()) / std
            else:
                sig[lead] = 0.0

        label_str = row.primary_class
        labels.append(CLASS_TO_IDX[label_str])
        signals.append(sig.astype(np.float32))

    signals = np.array(signals)
    labels = np.array(labels)

    print(f"Saving cache to {cache_path}...")
    np.savez(cache_path, signals=signals, labels=labels)
    return signals, labels


def load_cpsc(data_path):
    cache_path = os.path.join(data_path, "cpsc_cached.npz")
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = np.load(cache_path)
        return cached["signals"], cached["labels"]

    print("No cache found, loading from raw files...")
    hea_files = sorted([f for f in os.listdir(data_path) if f.endswith(".hea")])
    total = len(hea_files)

    signals = np.empty((total, 12, 5000), dtype=np.float32)
    labels = np.empty(total, dtype=np.int64)
    count = 0

    for hea_file in tqdm(hea_files, total=total, desc="Loading CPSC"):
        rec_id = hea_file.replace(".hea", "")
        record_path = os.path.join(data_path, rec_id)

        snomed_codes = []
        with open(os.path.join(data_path, hea_file), "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    codes_str = line.split(":", 1)[1].strip()
                    snomed_codes = [c.strip() for c in codes_str.split(",")]

        mapped = set()
        for code in snomed_codes:
            if code in SNOMED_TO_CLASS:
                mapped.add(SNOMED_TO_CLASS[code])
        if not mapped:
            label = "OTHERS"
        else:
            priority = ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "NORM"]
            label = "OTHERS"
            for p in priority:
                if p in mapped:
                    label = p
                    break

        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal.T
        except Exception:
            continue
        if sig.shape[0] != 12:
            continue

        if sig.shape[1] < 5000:
            sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
        elif sig.shape[1] > 5000:
            sig = sig[:, :5000]

        sig = np.nan_to_num(sig, nan=0.0)
        sig = bandpass_filter(sig, fs=500)
        for lead in range(12):
            std = sig[lead].std()
            if std > 1e-6:
                sig[lead] = (sig[lead] - sig[lead].mean()) / std
            else:
                sig[lead] = 0.0

        signals[count] = sig
        labels[count] = CLASS_TO_IDX[label]
        count += 1

    signals = signals[:count]
    labels = labels[:count]

    print(f"Saving cache to {cache_path}...")
    np.savez(cache_path, signals=signals, labels=labels)
    return signals, labels


class ECGDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = torch.from_numpy(signals)
        if labels.ndim == 2:
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        if self.augment:
            sig = sig + torch.randn_like(sig) * 0.05
            shift = torch.randint(-250, 250, (1,)).item()
            sig = torch.roll(sig, shift, dims=-1)
            scale = 0.85 + torch.rand(1).item() * 0.30
            sig = sig * scale
            if torch.rand(1).item() < 0.15:
                drop_lead = torch.randint(0, 12, (1,)).item()
                sig[drop_lead] = 0.0
        return sig, self.labels[idx]


# =============================================================================
# CNN-BiLSTM-Transformer Model
# =============================================================================
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNNBiLSTMTransformer(nn.Module):

    def __init__(self, num_leads=12, num_classes=7,
                 cnn_channels=None, cnn_kernel_size=7,
                 lstm_hidden=96, lstm_layers=1, lstm_dropout=0.0,
                 tf_heads=4, tf_layers=2, tf_dropout=0.3):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [48, 96, 192]

        cnn_blocks = []
        in_ch = num_leads
        for out_ch in cnn_channels:
            cnn_blocks.append(CNNBlock(in_ch, out_ch, cnn_kernel_size, pool_size=2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)
        self.cnn_drop = nn.Dropout(0.2)
        self.cnn_out_dim = cnn_channels[-1]

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        embed_dim = lstm_hidden * 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 626, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(tf_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=tf_heads,
            dim_feedforward=embed_dim * 4,
            dropout=tf_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(tf_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn_drop(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)       # (B, 625, 256)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 626, 256)

        seq_len = x.shape[1]
        x = self.pos_drop(x + self.pos_embed[:, :seq_len, :])
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])  # CLS token
        return logits


# =============================================================================
# Training (multilabel: BCEWithLogitsLoss)
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for signals, labels in tqdm(dataloader, desc="Training", leave=False):
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).int()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy().astype(int))

    if scheduler:
        scheduler.step()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(dataloader), macro_f1


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            logits = model(signals)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy().astype(int))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(dataloader), macro_f1


# =============================================================================
# Evaluation (multilabel with per-class optimized thresholds)
# =============================================================================
def get_probs(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for signals, _ in dataloader:
            signals = signals.to(DEVICE)
            logits = model(signals)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate(model, dataloader, dataset_name, y_true_ml, thresholds,
             flag_threshold=0.60):
    all_probs = get_probs(model, dataloader)
    n = len(all_probs)
    classes = CONFIG["classes"]
    n_cls = len(classes)

    y_pred = np.zeros_like(y_true_ml, dtype=int)
    for j, cls in enumerate(classes):
        y_pred[:, j] = (all_probs[:, j] >= thresholds[cls]).astype(int)

    auroc_per = []
    for j in range(n_cls):
        if y_true_ml[:, j].sum() == 0:
            auroc_per.append(float("nan"))
        else:
            try:
                auroc_per.append(roc_auc_score(y_true_ml[:, j], all_probs[:, j]))
            except ValueError:
                auroc_per.append(float("nan"))
    valid_aurocs = [v for v in auroc_per if not np.isnan(v)]
    auroc_macro = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    f1_macro = f1_score(y_true_ml, y_pred, average="macro", zero_division=0)
    f1_per = f1_score(y_true_ml, y_pred, average=None, zero_division=0)

    mcm = multilabel_confusion_matrix(y_true_ml, y_pred)
    sensitivity, specificity = [], []
    for j in range(n_cls):
        tn, fp, fn, tp = mcm[j].ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        sensitivity.append(sens)
        specificity.append(spec)

    max_prob = all_probs.max(axis=1)
    flagged = max_prob < flag_threshold
    flag_rate = flagged.sum() / n * 100

    sep = "-" * 74
    print(f"\n{'='*74}")
    print(f"  Dataset : {dataset_name}   N={n}   (multilabel eval)")
    print(f"{'='*74}")
    print(f"  Macro AUROC : {auroc_macro:.4f}")
    print(f"  Macro F1    : {f1_macro:.4f}")
    print(f"  Low-conf flag rate (max prob < {flag_threshold:.2f}) : "
          f"{flag_rate:.1f}%  ({int(flagged.sum())} / {n})")
    print(f"\n  {'Class':<8} {'Thresh':>7} {'AUROC':>7} {'F1':>7} "
          f"{'Sens':>7} {'Spec':>7} {'Support':>8}")
    print(f"  {sep}")
    for j, cls in enumerate(classes):
        support = int(y_true_ml[:, j].sum())
        print(f"  {cls:<8} {thresholds[cls]:>7.2f} "
              f"{auroc_per[j]:>7.4f} {f1_per[j]:>7.4f} "
              f"{sensitivity[j]:>7.4f} {specificity[j]:>7.4f} {support:>8}")
    print(f"{'='*74}")

    per_class = {}
    for j, cls in enumerate(classes):
        tn, fp, fn, tp = mcm[j].ravel()
        per_class[cls] = {
            "threshold": thresholds[cls],
            "auroc": round(float(auroc_per[j]), 4) if not np.isnan(auroc_per[j]) else None,
            "f1": round(float(f1_per[j]), 4),
            "sensitivity": round(float(sensitivity[j]), 4) if not np.isnan(sensitivity[j]) else None,
            "specificity": round(float(specificity[j]), 4) if not np.isnan(specificity[j]) else None,
            "support": int(y_true_ml[:, j].sum()),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    results_json = {
        "dataset": dataset_name,
        "model": CONFIG["model"],
        "evaluation": "multilabel_per_class_threshold",
        "macro_auroc": round(float(auroc_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "thresholds": thresholds,
        "per_class": per_class,
        "low_confidence_flagging": {
            "threshold": flag_threshold,
            "flagged_count": int(flagged.sum()),
            "total_count": n,
            "flagged_rate_pct": round(float(flag_rate), 2),
        },
    }

    json_path = os.path.join(
        CONFIG["output_dir"],
        f"results_{dataset_name.lower().replace(' ', '_')}.json",
    )
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results saved to {json_path}")

    return results_json


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n=== Loading PTB-XL ===")
    t0 = time.time()
    signals, labels, folds = load_ptbxl(CONFIG["ptbxl_path"])
    print(f"Loaded {len(signals)} records in {time.time()-t0:.1f}s")

    print("\n=== Building multilabel ground truth (PTB-XL) ===")
    labels_ml_all = ptbxl_multilabel_from_csv(CONFIG["ptbxl_path"])

    train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
    val_mask = folds == 9
    test_mask = folds == 10

    train_labels_ml = labels_ml_all[train_mask]
    val_labels_ml = labels_ml_all[val_mask]
    test_labels_ml = labels_ml_all[test_mask]

    print("Class distribution (train, multilabel):")
    for i, cls in enumerate(CONFIG["classes"]):
        print(f"  {cls}: {int(train_labels_ml[:, i].sum())}")

    train_ds = ECGDataset(signals[train_mask], train_labels_ml, augment=True)
    val_ds = ECGDataset(signals[val_mask], val_labels_ml)
    test_ds = ECGDataset(signals[test_mask], test_labels_ml)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    pos_counts = train_labels_ml.sum(axis=0)
    neg_counts = len(train_labels_ml) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = torch.FloatTensor(pos_weight).to(DEVICE)
    print(f"BCE pos_weight: {pos_weight.cpu().numpy().round(2)}")

    model = CNNBiLSTMTransformer(
        num_leads=CONFIG["num_leads"],
        num_classes=CONFIG["num_classes"],
        cnn_channels=CONFIG["cnn_channels"],
        cnn_kernel_size=CONFIG["cnn_kernel_size"],
        lstm_hidden=CONFIG["lstm_hidden"],
        lstm_layers=CONFIG["lstm_layers"],
        lstm_dropout=CONFIG["lstm_dropout"],
        tf_heads=CONFIG["tf_heads"],
        tf_layers=CONFIG["tf_layers"],
        tf_dropout=CONFIG["tf_dropout"],
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    warmup_epochs = CONFIG.get("warmup_epochs", 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, CONFIG["epochs"] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model_path = os.path.join(CONFIG["output_dir"], "best_model_multilabel.pt")

    if CONFIG["skip_training"] and os.path.exists(model_path):
        print("\n=== Skipping training (skip_training=True, loading saved model) ===")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        if os.path.exists(model_path):
            print(f"\n=== Resuming from {model_path} at LR={CONFIG['learning_rate']} ===")
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("\n=== Training from scratch (multilabel BCE) ===")
        best_val_f1 = 0
        patience = 25
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(CONFIG["epochs"]):
            t_start = time.time()
            train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, scheduler)
            val_loss, val_f1 = validate(model, val_loader, criterion)
            elapsed = time.time() - t_start

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
                  f"{elapsed:.1f}s")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curves (CNN-BiLSTM-Transformer, Multilabel BCE)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
        plt.close()

        model.load_state_dict(torch.load(model_path, weights_only=True))

    print(f"  Val multilabel shape:  {val_labels_ml.shape}")
    print(f"  Test multilabel shape: {test_labels_ml.shape}")

    print("\n=== Optimizing per-class thresholds on validation set ===")
    val_probs = get_probs(model, val_loader)
    thresholds = optimize_thresholds(val_labels_ml, val_probs, CONFIG["classes"])
    print(f"  Optimized thresholds: {thresholds}")

    print("\n=== Evaluating on PTB-XL Test Set (Fold 10) ===")
    ptbxl_results = evaluate(model, test_loader, "PTB-XL", test_labels_ml, thresholds)

    print("\n=== Loading Georgia Dataset ===")
    t0 = time.time()
    georgia_signals, georgia_labels = load_georgia(CONFIG["georgia_path"])
    print(f"Loaded {len(georgia_signals)} records in {time.time()-t0:.1f}s")

    georgia_ds = ECGDataset(georgia_signals, georgia_labels)
    georgia_loader = DataLoader(georgia_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print("\n=== Building multilabel ground truth (Georgia) ===")
    georgia_labels_ml = georgia_multilabel(CONFIG["georgia_path"])
    print(f"  Georgia multilabel shape: {georgia_labels_ml.shape}")

    print("\n=== Evaluating on Georgia ===")
    georgia_results = evaluate(model, georgia_loader, "Georgia", georgia_labels_ml, thresholds)

    print("\n=== Loading CPSC 2018 Dataset ===")
    t0 = time.time()
    cpsc_signals, cpsc_labels = load_cpsc(CONFIG["cpsc_path"])
    print(f"Loaded {len(cpsc_signals)} records in {time.time()-t0:.1f}s")

    cpsc_ds = ECGDataset(cpsc_signals, cpsc_labels)
    cpsc_loader = DataLoader(cpsc_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print("\n=== Building multilabel ground truth (CPSC) ===")
    cpsc_labels_ml = cpsc_multilabel(CONFIG["cpsc_path"])
    print(f"  CPSC multilabel shape: {cpsc_labels_ml.shape}")

    print("\n=== Evaluating on CPSC 2018 ===")
    cpsc_results = evaluate(model, cpsc_loader, "CPSC2018", cpsc_labels_ml, thresholds)

    print("\n=== Experiment Complete ===")
    print(f"Optimized thresholds: {thresholds}")
    print(f"PTB-XL  Macro AUROC: {ptbxl_results['macro_auroc']}  Macro F1: {ptbxl_results['macro_f1']}")
    print(f"Georgia Macro AUROC: {georgia_results['macro_auroc']}  Macro F1: {georgia_results['macro_f1']}")
    print(f"CPSC    Macro AUROC: {cpsc_results['macro_auroc']}  Macro F1: {cpsc_results['macro_f1']}")
    print(f"\nResults saved to {CONFIG['output_dir']}/")

    summary = {
        "threshold": "per_class_optimized",
        "optimized_thresholds": thresholds,
        "flag_threshold": CONFIG["confidence_threshold"],
        "models": {
            CONFIG["model"]: {
                "PTB-XL fold10": {
                    "n": ptbxl_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro": ptbxl_results["macro_auroc"],
                    "f1_macro": ptbxl_results["macro_f1"],
                    "flag_rate_pct": ptbxl_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count": ptbxl_results["low_confidence_flagging"]["flagged_count"],
                },
                "Georgia": {
                    "n": georgia_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro": georgia_results["macro_auroc"],
                    "f1_macro": georgia_results["macro_f1"],
                    "flag_rate_pct": georgia_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count": georgia_results["low_confidence_flagging"]["flagged_count"],
                },
                "CPSC2018": {
                    "n": cpsc_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro": cpsc_results["macro_auroc"],
                    "f1_macro": cpsc_results["macro_f1"],
                    "flag_rate_pct": cpsc_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count": cpsc_results["low_confidence_flagging"]["flagged_count"],
                },
            }
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = os.path.join(CONFIG["output_dir"], "results_summary_cnn_bilstm_transformer.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
