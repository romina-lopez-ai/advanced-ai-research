"""
COMP6011 Research Task 3 -- BiLSTM1D ECG Classifier (Trained from scratch)
Date: 2026-05-20

Input:
    Raw 12-lead ECG signal (12, 1000) at 100 Hz low-res.
    Model input shape: (batch, 12, 1000).

Training protocol:
    PTB-XL folds 1-8  → train
    PTB-XL fold 9     → validation (threshold optimisation)
    PTB-XL fold 10    → test
    Georgia, CPSC     → external evaluation

Design choices:
    - Bidirectional LSTM with adaptive concat pooling (avg + max + last timestep).
    - Multilabel BCEWithLogitsLoss + pos_weight for class imbalance.
    - Per-class threshold search on val set (maximises per-class F1).
    - Bandpass 0.5-45 Hz at 100 Hz; per-lead z-score normalisation.
    - Cosine annealing with linear warmup; early stopping on val macro-F1.
    - Mixed precision (AMP) training for speed.
    - Reuses PTB-XL/Georgia/CPSC cache files if already built by resnet18_2d_experiment.py.
"""

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    "experiment_name": "bilstm1d_ecg_ptbxl",
    "model":           "BiLSTM1D_Trained",
    "num_classes":     7,
    "classes":         ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "OTHERS"],
    "sampling_rate":   100,       # low-res PTB-XL (filename_lr)
    "signal_length":   1000,      # samples at 100 Hz
    "num_leads":       12,
    # BiLSTM architecture
    "lstm_hidden":     128,       # hidden size per direction (×2 for bidirectional)
    "lstm_layers":     2,         # stacked LSTM layers
    "lstm_dropout":    0.3,       # dropout between LSTM layers (only if layers > 1)
    # Training
    "batch_size":      64,        # reduce to 32 if GPU OOM
    "epochs":          60,
    "learning_rate":   1e-3,
    "weight_decay":    1e-3,
    "warmup_epochs":   5,
    "patience":        20,
    "confidence_threshold": 0.60,
    # Mixed precision: True = ~1.5x faster on CUDA, no accuracy loss
    "use_amp":         True,
    # Paths (relative to project root)
    "ptbxl_path":   "data/raw/ptbxl",
    "georgia_path": "data/raw/georgia",
    "cpsc_path":    "data/raw/cpsc2018",
    "output_dir":   "code/experiments/results_bilstm1d_trained",
    # Set True to skip training and only evaluate a saved checkpoint
    "skip_training": False,
}

import os, ast, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, multilabel_confusion_matrix
from scipy.signal import butter, sosfiltfilt, resample_poly
import wfdb
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print(f"Experiment : {CONFIG['experiment_name']}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device     : {DEVICE}")
print(f"PyTorch    : {torch.__version__}")

CLASSES      = CONFIG["classes"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# =============================================================================
# Label mapping constants
# =============================================================================

SCP_TO_CLASS = {
    "NORM":  "NORM",
    "AFIB":  "AFIB",
    "AFLT":  "AFLT",
    "1AVB":  "1dAVb",
    "CRBBB": "RBBB",
    "IRBBB": "RBBB",
    "CLBBB": "LBBB",
    "ILBBB": "LBBB",
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

PRIORITY = ["AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "NORM"]


# =============================================================================
# Label mapping helpers
# =============================================================================

def scp_to_singlelabel(scp_codes_str: str):
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception as e:
        print(f"  [warn] cannot parse scp_codes: {e}")
        return None
    mapped = set()
    for code, conf in codes.items():
        if conf >= 100.0:
            cls = SCP_TO_CLASS.get(code)
            if cls:
                mapped.add(cls)
    if not mapped:
        return "OTHERS"
    for p in PRIORITY:
        if p in mapped:
            return p
    return "OTHERS"


def scp_to_multilabel(scp_codes_str: str) -> np.ndarray:
    y = np.zeros(7, dtype=np.float32)
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        y[CLASS_TO_IDX["OTHERS"]] = 1
        return y
    present = {c for c, lh in codes.items() if lh >= 100.0}
    has_named = False
    for code in present:
        cls = SCP_TO_CLASS.get(code)
        if cls:
            y[CLASS_TO_IDX[cls]] = 1
            has_named = True
    if not has_named or (present - set(SCP_TO_CLASS.keys())):
        y[CLASS_TO_IDX["OTHERS"]] = 1
    return y


def hea_to_multilabel(hea_path: str) -> np.ndarray:
    y = np.zeros(7, dtype=np.float32)
    try:
        with open(hea_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    codes_str = line.split(":", 1)[1].strip()
                    has_named = False
                    has_unmapped = False
                    for code in codes_str.split(","):
                        code = code.strip()
                        cls = SNOMED_TO_CLASS.get(code)
                        if cls:
                            y[CLASS_TO_IDX[cls]] = 1
                            has_named = True
                        else:
                            has_unmapped = True
                    if has_unmapped or not has_named:
                        y[CLASS_TO_IDX["OTHERS"]] = 1
    except Exception:
        pass
    return y


def hea_to_singlelabel(hea_path: str) -> str:
    mapped = set()
    try:
        with open(hea_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and "Dx" in line:
                    for code in line.split(":", 1)[1].strip().split(","):
                        cls = SNOMED_TO_CLASS.get(code.strip())
                        if cls:
                            mapped.add(cls)
    except Exception:
        pass
    if not mapped:
        return "OTHERS"
    for p in PRIORITY:
        if p in mapped:
            return p
    return "OTHERS"


# =============================================================================
# Signal preprocessing
# =============================================================================

def bandpass_filter(sig: np.ndarray, lowcut=0.5, highcut=45.0,
                    fs=100, order=3) -> np.ndarray:
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1).astype(np.float32)


def preprocess_signal(sig: np.ndarray, fs_in: int = 100):
    if sig.shape[0] != 12:
        return None
    if fs_in != 100:
        factor = int(round(fs_in / 100))
        sig = resample_poly(sig, up=1, down=factor, axis=1).astype(np.float32)
    if sig.shape[1] >= 1000:
        sig = sig[:, :1000]
    else:
        sig = np.pad(sig, ((0, 0), (0, 1000 - sig.shape[1]))).astype(np.float32)
    sig = np.nan_to_num(sig.astype(np.float32), nan=0.0)
    sig = bandpass_filter(sig, fs=100)
    for lead in range(12):
        std = sig[lead].std()
        if std > 1e-6:
            sig[lead] = (sig[lead] - sig[lead].mean()) / std
        else:
            sig[lead] = 0.0
    if np.isnan(sig).any() or np.isinf(sig).any():
        return None
    return sig.astype(np.float32)


# =============================================================================
# Data loading  (shares cache files with resnet18_2d_experiment.py)
# =============================================================================

def load_ptbxl(data_path: str):
    """Load PTB-XL low-res (100 Hz). Reuses cache from resnet18_2d if present."""
    cache = os.path.join(data_path, "ptbxl_lr_cached.npz")
    if os.path.exists(cache):
        print(f"  Loading PTB-XL from cache: {cache}")
        d = np.load(cache)
        return d["signals"], d["labels"], d["ml_labels"], d["folds"]

    print("  No cache — loading PTB-XL from raw files (first run ~3 min)...")
    db = pd.read_csv(os.path.join(data_path, "ptbxl_database.csv"))
    signals, labels, ml_labels, folds = [], [], [], []
    skipped = 0

    for _, row in tqdm(db.iterrows(), total=len(db), desc="  PTB-XL"):
        sl = scp_to_singlelabel(row["scp_codes"])
        if sl is None:
            skipped += 1
            continue
        ml = scp_to_multilabel(row["scp_codes"])
        rec_path = os.path.join(data_path, row["filename_lr"])
        try:
            sig, _ = wfdb.rdsamp(rec_path)
            sig = sig.T.astype(np.float32)
        except Exception:
            skipped += 1
            continue
        processed = preprocess_signal(sig, fs_in=100)
        if processed is None:
            skipped += 1
            continue
        signals.append(processed)
        labels.append(CLASS_TO_IDX[sl])
        ml_labels.append(ml)
        folds.append(int(row["strat_fold"]))

    signals   = np.stack(signals).astype(np.float32)
    labels    = np.array(labels, dtype=np.int64)
    ml_labels = np.stack(ml_labels).astype(np.float32)
    folds     = np.array(folds, dtype=np.int32)

    print(f"  Loaded {len(signals)} records  (skipped {skipped})")
    np.savez(cache, signals=signals, labels=labels, ml_labels=ml_labels, folds=folds)
    print(f"  Cache saved → {cache}")
    return signals, labels, ml_labels, folds


def load_georgia(data_path: str):
    """Load Georgia 12-Lead ECG (500 Hz → resampled to 100 Hz)."""
    cache = os.path.join(data_path, "georgia_lr_cached.npz")
    if os.path.exists(cache):
        print(f"  Loading Georgia from cache: {cache}")
        d = np.load(cache)
        return d["signals"], d["labels"], d["ml_labels"]

    print("  No cache — loading Georgia from raw files...")
    signals, labels, ml_labels = [], [], []
    skipped = 0

    for gdir in sorted(os.scandir(data_path), key=lambda e: e.name):
        if not gdir.is_dir():
            continue
        for hea_path in sorted(
            os.path.join(gdir.path, f)
            for f in os.listdir(gdir.path) if f.endswith(".hea")
        ):
            mat_path = hea_path.replace(".hea", ".mat")
            if not os.path.exists(mat_path):
                skipped += 1
                continue
            try:
                sig = loadmat(mat_path)["val"].astype(np.float32)
            except Exception:
                skipped += 1
                continue
            processed = preprocess_signal(sig, fs_in=500)
            if processed is None:
                skipped += 1
                continue
            signals.append(processed)
            labels.append(CLASS_TO_IDX[hea_to_singlelabel(hea_path)])
            ml_labels.append(hea_to_multilabel(hea_path))

    signals   = np.stack(signals).astype(np.float32)
    labels    = np.array(labels, dtype=np.int64)
    ml_labels = np.stack(ml_labels).astype(np.float32)

    print(f"  Loaded {len(signals)} records  (skipped {skipped})")
    np.savez(cache, signals=signals, labels=labels, ml_labels=ml_labels)
    print(f"  Cache saved → {cache}")
    return signals, labels, ml_labels


def load_cpsc(data_path: str):
    """Load CPSC 2018 (500 Hz → resampled to 100 Hz)."""
    cache = os.path.join(data_path, "cpsc_lr_cached.npz")
    if os.path.exists(cache):
        print(f"  Loading CPSC from cache: {cache}")
        d = np.load(cache)
        return d["signals"], d["labels"], d["ml_labels"]

    print("  No cache — loading CPSC from raw files...")
    signals, labels, ml_labels = [], [], []
    skipped = 0

    for gdir in sorted(os.scandir(data_path), key=lambda e: e.name):
        if not gdir.is_dir():
            continue
        for hea_file in sorted(f for f in os.listdir(gdir.path) if f.endswith(".hea")):
            hea_path = os.path.join(gdir.path, hea_file)
            rec_path = hea_path.replace(".hea", "")
            try:
                sig, fields = wfdb.rdsamp(rec_path)
                sig = sig.T.astype(np.float32)
                fs_in = fields.get("fs", 500)
            except Exception:
                skipped += 1
                continue
            processed = preprocess_signal(sig, fs_in=fs_in)
            if processed is None:
                skipped += 1
                continue
            signals.append(processed)
            labels.append(CLASS_TO_IDX[hea_to_singlelabel(hea_path)])
            ml_labels.append(hea_to_multilabel(hea_path))

    signals   = np.stack(signals).astype(np.float32)
    labels    = np.array(labels, dtype=np.int64)
    ml_labels = np.stack(ml_labels).astype(np.float32)

    print(f"  Loaded {len(signals)} records  (skipped {skipped})")
    np.savez(cache, signals=signals, labels=labels, ml_labels=ml_labels)
    print(f"  Cache saved → {cache}")
    return signals, labels, ml_labels


# =============================================================================
# Dataset  (returns raw signal tensor — no 2D image conversion)
# =============================================================================

class ECGSignalDataset(Dataset):
    """
    Wraps (N, 12, 1000) signal arrays.
    Returns (12, 1000) float32 tensor directly — no image conversion.
    """

    def __init__(self, signals: np.ndarray, ml_labels: np.ndarray,
                 augment: bool = False):
        self.signals   = signals
        self.ml_labels = torch.from_numpy(ml_labels).float()
        self.augment   = augment

    def __len__(self):
        return len(self.ml_labels)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()   # (12, 1000)

        if self.augment:
            sig += np.random.randn(*sig.shape).astype(np.float32) * 0.05
            sig *= (0.85 + np.random.rand() * 0.30)
            shift = np.random.randint(-100, 100)
            sig = np.roll(sig, shift, axis=-1)
            if np.random.rand() < 0.15:
                drop = np.random.randint(0, 12)
                sig[drop] = 0.0

        return torch.tensor(sig, dtype=torch.float32), self.ml_labels[idx]


# =============================================================================
# Model — BiLSTM1D with adaptive concat pooling
# =============================================================================

class BiLSTM1DModel(nn.Module):
    """
    Bidirectional LSTM for 12-lead ECG classification.

    Forward pass:
      (batch, 12, 1000)
        → permute → (batch, 1000, 12)
        → BiLSTM  → (batch, 1000, hidden*2)
        → concat pool [avg | max | last] → (batch, hidden*6)
        → BN → Dropout → Linear → GELU → Dropout → Linear(num_classes)
    """

    def __init__(self, num_leads: int = 12, hidden_size: int = 128,
                 num_layers: int = 2, lstm_dropout: float = 0.3,
                 num_classes: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_leads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_size * 2   # bidirectional output
        pool_dim = feat_dim * 3      # avg + max + last timestep

        self.head = nn.Sequential(
            nn.BatchNorm1d(pool_dim),
            nn.Dropout(0.5),
            nn.Linear(pool_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12, 1000)
        x = x.permute(0, 2, 1)              # (batch, 1000, 12)
        out, _ = self.lstm(x)               # (batch, 1000, hidden*2)
        avg    = out.mean(dim=1)            # (batch, hidden*2)
        mx     = out.max(dim=1).values      # (batch, hidden*2)
        last   = out[:, -1, :]              # (batch, hidden*2)
        pooled = torch.cat([avg, mx, last], dim=1)   # (batch, hidden*6)
        return self.head(pooled)


def build_bilstm(num_classes: int = 7) -> nn.Module:
    return BiLSTM1DModel(
        num_leads   = CONFIG["num_leads"],
        hidden_size = CONFIG["lstm_hidden"],
        num_layers  = CONFIG["lstm_layers"],
        lstm_dropout= CONFIG["lstm_dropout"],
        num_classes = num_classes,
    )


# =============================================================================
# Per-class threshold optimisation
# =============================================================================

def optimize_thresholds(y_true: np.ndarray, y_probs: np.ndarray,
                        class_names: list) -> dict:
    thresholds = np.arange(0.05, 0.80, 0.01)
    best = {}
    for j, cls in enumerate(class_names):
        if y_true[:, j].sum() == 0:
            best[cls] = 0.5
            continue
        best_f1, best_t = 0.0, 0.5
        col_true = y_true[:, j]
        col_prob = y_probs[:, j]
        for t in thresholds:
            pred = (col_prob >= t).astype(int)
            tp = int(((pred == 1) & (col_true == 1)).sum())
            fp = int(((pred == 1) & (col_true == 0)).sum())
            fn = int(((pred == 0) & (col_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[cls] = round(float(best_t), 2)
    return best


# =============================================================================
# Training helpers
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    for sigs, labels in tqdm(loader, desc="  train", leave=False):
        sigs, labels = sigs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(sigs)
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy().astype(int))

    if scheduler:
        scheduler.step()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for sigs, labels in loader:
        sigs, labels = sigs.to(DEVICE), labels.to(DEVICE)
        total_loss += criterion(model(sigs), labels).item()
        preds = (torch.sigmoid(model(sigs)) >= 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy().astype(int))

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1


@torch.no_grad()
def get_probs(model, loader) -> np.ndarray:
    model.eval()
    all_probs = []
    for sigs, _ in loader:
        sigs = sigs.to(DEVICE)
        all_probs.append(torch.sigmoid(model(sigs)).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# =============================================================================
# Evaluation  (identical JSON format to resnet18_2d_experiment.py)
# =============================================================================

def evaluate(model, loader, dataset_name: str, y_true_ml: np.ndarray,
             thresholds: dict, flag_threshold: float = 0.60) -> dict:
    all_probs = get_probs(model, loader)
    n     = len(all_probs)
    n_cls = len(CLASSES)

    y_pred = np.zeros((n, n_cls), dtype=int)
    for j, cls in enumerate(CLASSES):
        y_pred[:, j] = (all_probs[:, j] >= thresholds[cls]).astype(int)

    auroc_per = []
    for j in range(n_cls):
        if y_true_ml[:, j].sum() == 0:
            auroc_per.append(float("nan"))
        else:
            try:
                auroc_per.append(float(roc_auc_score(y_true_ml[:, j], all_probs[:, j])))
            except ValueError:
                auroc_per.append(float("nan"))
    valid       = [v for v in auroc_per if not math.isnan(v)]
    auroc_macro = float(np.mean(valid)) if valid else float("nan")

    f1_macro = float(f1_score(y_true_ml, y_pred, average="macro", zero_division=0))
    f1_per   = f1_score(y_true_ml, y_pred, average=None, zero_division=0)

    mcm = multilabel_confusion_matrix(y_true_ml, y_pred)
    sensitivity, specificity = [], []
    for j in range(n_cls):
        tn, fp, fn, tp = mcm[j].ravel()
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else float("nan"))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))

    max_prob   = all_probs.max(axis=1)
    flag_count = int((max_prob < flag_threshold).sum())
    flag_rate  = flag_count / n * 100

    print(f"\n{'='*74}")
    print(f"  Dataset : {dataset_name}   N={n}")
    print(f"{'='*74}")
    print(f"  Macro AUROC : {auroc_macro:.4f}")
    print(f"  Macro F1    : {f1_macro:.4f}")
    print(f"  Low-conf flag rate (<{flag_threshold:.2f}) : "
          f"{flag_rate:.1f}%  ({flag_count}/{n})")
    print(f"\n  {'Class':<8} {'Thresh':>7} {'AUROC':>7} {'F1':>7} "
          f"{'Sens':>7} {'Spec':>7} {'Supp':>6}")
    print(f"  {'-'*70}")
    for j, cls in enumerate(CLASSES):
        au = f"{auroc_per[j]:.4f}" if not math.isnan(auroc_per[j]) else "  N/A "
        print(f"  {cls:<8} {thresholds[cls]:>7.2f} {au:>7} "
              f"{f1_per[j]:>7.4f} {sensitivity[j]:>7.4f} "
              f"{specificity[j]:>7.4f} {int(y_true_ml[:,j].sum()):>6}")
    print(f"{'='*74}")

    per_class = {}
    for j, cls in enumerate(CLASSES):
        tn, fp, fn, tp = [int(x) for x in mcm[j].ravel()]
        per_class[cls] = {
            "threshold":   thresholds[cls],
            "auroc":       round(auroc_per[j], 6) if not math.isnan(auroc_per[j]) else None,
            "f1":          round(float(f1_per[j]), 6),
            "sensitivity": round(sensitivity[j], 6) if not math.isnan(sensitivity[j]) else None,
            "specificity": round(specificity[j], 6) if not math.isnan(specificity[j]) else None,
            "support":     tp + fn,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    result = {
        "dataset":    dataset_name,
        "model":      CONFIG["model"],
        "evaluation": "multilabel_per_class_threshold",
        "macro_auroc": round(auroc_macro, 6) if not math.isnan(auroc_macro) else None,
        "macro_f1":    round(f1_macro, 6),
        "thresholds":  thresholds,
        "per_class":   per_class,
        "low_confidence_flagging": {
            "threshold":      flag_threshold,
            "flagged_count":  flag_count,
            "total_count":    n,
            "flagged_rate_pct": round(flag_rate, 2),
        },
    }

    out_path = os.path.join(
        CONFIG["output_dir"],
        f"results_{dataset_name.lower().replace(' ', '_')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved → {out_path}")
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ── 1. Load PTB-XL ────────────────────────────────────────────────────────
    print("\n=== Loading PTB-XL (100 Hz low-res) ===")
    t0 = time.time()
    signals, sl_labels, ml_labels, folds = load_ptbxl(CONFIG["ptbxl_path"])
    print(f"  {len(signals)} total records  ({time.time()-t0:.1f}s)")

    train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
    val_mask   = folds == 9
    test_mask  = folds == 10

    train_ml = ml_labels[train_mask]
    val_ml   = ml_labels[val_mask]
    test_ml  = ml_labels[test_mask]

    print(f"  Split → Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")
    print("  Train class distribution (multilabel):")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {int(train_ml[:, i].sum())}")

    # ── 2. Datasets & DataLoaders ─────────────────────────────────────────────
    train_ds = ECGSignalDataset(signals[train_mask], train_ml, augment=True)
    val_ds   = ECGSignalDataset(signals[val_mask],   val_ml,   augment=False)
    test_ds  = ECGSignalDataset(signals[test_mask],  test_ml,  augment=False)

    nw = min(4, os.cpu_count() or 1)
    dl_kwargs = dict(batch_size=CONFIG["batch_size"], num_workers=nw,
                     pin_memory=True, persistent_workers=(nw > 0))
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\n=== Building BiLSTM1D ===")
    model = build_bilstm(num_classes=CONFIG["num_classes"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  hidden={CONFIG['lstm_hidden']}  layers={CONFIG['lstm_layers']}  "
          f"bidirectional=True")
    print(f"  Parameters: {n_params:,}")

    # ── 4. Loss with pos_weight ───────────────────────────────────────────────
    pos_counts = train_ml.sum(axis=0)
    neg_counts = len(train_ml) - pos_counts
    pos_weight = torch.FloatTensor(neg_counts / (pos_counts + 1e-6)).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight: {pos_weight.cpu().numpy().round(2)}")

    use_amp = CONFIG["use_amp"] and DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"  Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")

    # ── 5. Optimiser & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    warmup = CONFIG["warmup_epochs"]
    total  = CONFIG["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 6. Training ───────────────────────────────────────────────────────────
    model_path = os.path.join(CONFIG["output_dir"], "best_model.pt")

    if CONFIG["skip_training"] and os.path.exists(model_path):
        print("\n=== Skipping training (skip_training=True) ===")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        if os.path.exists(model_path):
            print(f"\n=== Resuming from {model_path} ===")
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("\n=== Training from scratch ===")

        best_val_f1      = 0.0
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(total):
            t_ep = time.time()
            tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer,
                                         criterion, scheduler, scaler)
            vl_loss, vl_f1 = validate(model, val_loader, criterion)
            elapsed = time.time() - t_ep

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            print(f"  Epoch {epoch+1:3d}/{total} | "
                  f"Train Loss {tr_loss:.4f} F1 {tr_f1:.4f} | "
                  f"Val Loss {vl_loss:.4f} F1 {vl_f1:.4f} | "
                  f"{elapsed:.1f}s")

            if vl_f1 > best_val_f1:
                best_val_f1      = vl_f1
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print(f"    ✓ Saved best model (val F1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}  "
                          f"(best val F1={best_val_f1:.4f})")
                    break

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses,   label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.set_title("BiLSTM1D Training Curves")
        ax.legend(); plt.tight_layout()
        curve_path = os.path.join(CONFIG["output_dir"], "training_curves.png")
        plt.savefig(curve_path, dpi=150); plt.close()
        print(f"  Training curves saved → {curve_path}")

        model.load_state_dict(torch.load(model_path, weights_only=True))

    # ── 7. Per-class threshold optimisation (on validation fold 9) ───────────
    print("\n=== Optimising per-class thresholds on fold 9 ===")
    val_probs  = get_probs(model, val_loader)
    thresholds = optimize_thresholds(val_ml, val_probs, CLASSES)
    print(f"  Optimised thresholds: {thresholds}")

    # ── 8. Evaluate on PTB-XL fold 10 ────────────────────────────────────────
    print("\n=== PTB-XL Test Set (Fold 10) ===")
    ptbxl_results = evaluate(model, test_loader, "PTB-XL fold10", test_ml, thresholds)

    # ── 9. Evaluate on Georgia ────────────────────────────────────────────────
    print("\n=== Georgia Dataset ===")
    t0 = time.time()
    geo_sig, _, geo_ml = load_georgia(CONFIG["georgia_path"])
    print(f"  {len(geo_sig)} records  ({time.time()-t0:.1f}s)")
    geo_ds     = ECGSignalDataset(geo_sig, geo_ml)
    geo_loader = DataLoader(geo_ds, shuffle=False, **dl_kwargs)
    georgia_results = evaluate(model, geo_loader, "Georgia", geo_ml, thresholds)

    # ── 10. Evaluate on CPSC 2018 ─────────────────────────────────────────────
    print("\n=== CPSC 2018 Dataset ===")
    t0 = time.time()
    cpsc_sig, _, cpsc_ml = load_cpsc(CONFIG["cpsc_path"])
    print(f"  {len(cpsc_sig)} records  ({time.time()-t0:.1f}s)")
    cpsc_ds     = ECGSignalDataset(cpsc_sig, cpsc_ml)
    cpsc_loader = DataLoader(cpsc_ds, shuffle=False, **dl_kwargs)
    cpsc_results = evaluate(model, cpsc_loader, "CPSC2018", cpsc_ml, thresholds)

    # ── 11. Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "threshold":            "per_class_optimized",
        "optimized_thresholds": thresholds,
        "flag_threshold":       CONFIG["confidence_threshold"],
        "models": {
            CONFIG["model"]: {
                "PTB-XL fold10": {
                    "n":             ptbxl_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   ptbxl_results["macro_auroc"],
                    "f1_macro":      ptbxl_results["macro_f1"],
                    "flag_rate_pct": ptbxl_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    ptbxl_results["low_confidence_flagging"]["flagged_count"],
                },
                "Georgia": {
                    "n":             georgia_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   georgia_results["macro_auroc"],
                    "f1_macro":      georgia_results["macro_f1"],
                    "flag_rate_pct": georgia_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    georgia_results["low_confidence_flagging"]["flagged_count"],
                },
                "CPSC2018": {
                    "n":             cpsc_results["low_confidence_flagging"]["total_count"],
                    "auroc_macro":   cpsc_results["macro_auroc"],
                    "f1_macro":      cpsc_results["macro_f1"],
                    "flag_rate_pct": cpsc_results["low_confidence_flagging"]["flagged_rate_pct"],
                    "flag_count":    cpsc_results["low_confidence_flagging"]["flagged_count"],
                },
            }
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_path = os.path.join(CONFIG["output_dir"], "results_summary_bilstm1d_trained.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── 12. Final print ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXPERIMENT COMPLETE")
    print("="*60)
    print(f"  PTB-XL  AUROC {ptbxl_results['macro_auroc']:.4f}  "
          f"F1 {ptbxl_results['macro_f1']:.4f}")
    print(f"  Georgia AUROC {georgia_results['macro_auroc']:.4f}  "
          f"F1 {georgia_results['macro_f1']:.4f}")
    print(f"  CPSC    AUROC {cpsc_results['macro_auroc']:.4f}  "
          f"F1 {cpsc_results['macro_f1']:.4f}")
    print(f"\n  All results saved → {CONFIG['output_dir']}/")
    print(f"  Summary           → {summary_path}")
