"""
BiLSTM inference script.
Checkpoint: output/lstm_all/checkpoints/best_model.ckpt
Input:  (batch, 12, 1000)  float32
Output: (batch, 71)        logits → sigmoid → map to 7 classes

Usage:
    python src/models/bilstm1d.py --data data/validation
    python src/models/bilstm1d.py --data data/raw/ptbxl --fold 10
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "output/lstm_all/checkpoints/best_model.ckpt"
SCALER = ROOT / "ecg_ptbxl_benchmarking/output/exp0/data/standard_scaler.pkl"

# ── PTB-XL 71-class order (from mlb.pkl) ────────────────────────────────────
CLASSES_71 = [
    '1AVB','2AVB','3AVB','ABQRS','AFIB','AFLT','ALMI','AMI','ANEUR','ASMI',
    'BIGU','CLBBB','CRBBB','DIG','EL','HVOLT','ILBBB','ILMI','IMI','INJAL',
    'INJAS','INJIL','INJIN','INJLA','INVT','IPLMI','IPMI','IRBBB','ISCAL',
    'ISCAN','ISCAS','ISCIL','ISCIN','ISCLA','ISC_','IVCD','LAFB','LAO/LAE',
    'LMI','LNGQT','LOWT','LPFB','LPR','LVH','LVOLT','NDT','NORM','NST_',
    'NT_','PAC','PACE','PMI','PRC(S)','PSVT','PVC','QWAVE','RAO/RAE','RVH',
    'SARRH','SBRAD','SEHYP','SR','STACH','STD_','STE_','SVARR','SVTAC',
    'TAB_','TRIGU','VCLVH','WPW',
]
IDX = {c: i for i, c in enumerate(CLASSES_71)}

# Map to our 7 classes: take max probability over each group
LABEL_MAP = {
    'NORM':   [IDX['NORM']],
    'AFIB':   [IDX['AFIB']],
    'AFLT':   [IDX['AFLT']],
    '1dAVb':  [IDX['1AVB']],
    'RBBB':   [IDX['CRBBB'], IDX['IRBBB']],
    'LBBB':   [IDX['CLBBB'], IDX['ILBBB']],
    'OTHERS': [i for i in range(71)
               if i not in {IDX['NORM'], IDX['AFIB'], IDX['AFLT'],
                            IDX['1AVB'], IDX['CRBBB'], IDX['IRBBB'],
                            IDX['CLBBB'], IDX['ILBBB']}],
}
CLASS_ORDER = ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB', 'OTHERS']


# ── Architecture ─────────────────────────────────────────────────────────────

class _Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class _AdaptiveConcatPoolRNN(nn.Module):
    """Placeholder module to preserve model.1[0] index — forward bypassed."""
    def forward(self, x):
        return x


class _BiLSTMModel(nn.Module):
    """
    Backbone: Sequential[Lambda, Lambda, nn.LSTM] → keys model.0.{0,1,2}.*
    Head:     Sequential[ConcatPool, Sequential[Dropout, Linear]] → keys model.1.*

    forward() bypasses Sequential.forward() for backbone and head to handle
    LSTM's tuple output and implement AdaptiveConcatPoolRNN manually.
    """
    def __init__(self, input_size=12, hidden_size=256, num_layers=2, num_classes=71):
        super().__init__()
        backbone = nn.Sequential(
            _Lambda(lambda x: x.permute(0, 2, 1)),  # (B,12,1000)→(B,1000,12)
            _Lambda(lambda x: x),                    # identity placeholder
            nn.LSTM(input_size, hidden_size, num_layers,
                    batch_first=True, bidirectional=True),
        )
        head = nn.Sequential(
            _AdaptiveConcatPoolRNN(),                # model.1[0] — no params
            nn.Sequential(                           # model.1[1]
                nn.Dropout(p=0.5),                   # model.1[1][0]
                nn.Linear(hidden_size * 6, num_classes),  # model.1[1][1]: 1536→71
            ),
        )
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        # x: (batch, 12, 1000)
        x = x.permute(0, 2, 1)                          # (batch, 1000, 12)
        out, (h_n, _) = self.model[0][2](x)             # LSTM: out (B,1000,512)
        avg  = out.mean(dim=1)                           # (batch, 512)
        mx   = out.max(dim=1).values                     # (batch, 512)
        last = out[:, -1, :]                             # (batch, 512) — last timestep
        pooled = torch.cat([avg, mx, last], dim=1)       # (batch, 1536)
        return self.model[1][1](pooled)                  # Dropout + Linear


# ── Load & map ───────────────────────────────────────────────────────────────

def load_model(ckpt_path=CKPT, device="cpu"):
    model = _BiLSTMModel()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    return model


def map_to_7(probs71: np.ndarray) -> np.ndarray:
    """(N, 71) → (N, 7)  using max pooling over class groups."""
    out = np.zeros((len(probs71), 7), dtype=np.float32)
    for j, cls in enumerate(CLASS_ORDER):
        out[:, j] = probs71[:, LABEL_MAP[cls]].max(axis=1)
    return out


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_scaler():
    with open(SCALER, "rb") as f:
        return pickle.load(f)


def load_npy_folder(folder: Path, scaler=None):
    """Load all .npy files from a folder of subfolders (validation/test format)."""
    records, names = [], []
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            npy = list(sub.glob("*.npy"))
            if npy:
                x = np.load(npy[0]).astype(np.float32)   # (12, 1000)
                if scaler:
                    x = ((x - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)
                records.append(x)
                names.append(sub.name)
    return np.stack(records), names  # (N, 12, 1000)


def load_ptbxl_fold(ptbxl_dir: Path, fold: int, scaler=None):
    """Load a specific PTB-XL fold using ptbxl_database.csv."""
    import pandas as pd
    import wfdb

    db = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    fold_df = db[db["strat_fold"] == fold]
    records, names = [], []
    for _, row in fold_df.iterrows():
        path = ptbxl_dir / row["filename_lr"]
        sig, _ = wfdb.rdsamp(str(path))          # (1000, 12)
        x = sig.T.astype(np.float32)             # (12, 1000)
        if scaler:
            x = ((x - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)
        records.append(x)
        names.append(row["filename_lr"])
    return np.stack(records), names


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, signals: np.ndarray, batch_size=64, device="cpu"):
    """signals: (N, 12, 1000)  →  probs71: (N, 71)"""
    all_probs = []
    for i in range(0, len(signals), batch_size):
        batch = torch.tensor(signals[i:i+batch_size]).to(device)
        logits = model(batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BiLSTM inference on ECG data")
    parser.add_argument("--data",   required=True, help="Path to data folder")
    parser.add_argument("--fold",   type=int, default=None,
                        help="PTB-XL fold number (if loading from raw ptbxl)")
    parser.add_argument("--ckpt",   default=str(CKPT))
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip standard scaler normalization")
    parser.add_argument("--out",    default=None,
                        help="Save predictions to .npy file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    data_path = Path(args.data)
    scaler = None if args.no_scale else _load_scaler()

    print(f"Loading model from {args.ckpt}")
    model = load_model(args.ckpt, device=device)

    print(f"Loading data from {data_path}")
    if args.fold is not None:
        signals, names = load_ptbxl_fold(data_path, args.fold, scaler)
    else:
        signals, names = load_npy_folder(data_path, scaler)

    print(f"  {len(signals)} recordings  shape {signals.shape}")

    print("Running inference...")
    probs71 = run_inference(model, signals, device=device)
    probs7  = map_to_7(probs71)

    print("\nPredictions (threshold 0.5):")
    print(f"{'Name':<30} " + "  ".join(f"{c:>6}" for c in CLASS_ORDER))
    print("-" * 80)
    for name, p in zip(names, probs7):
        preds = [("  Y  " if v >= 0.5 else "  -  ") for v in p]
        print(f"{name:<30} {''.join(preds)}")

    if args.out:
        out_path = Path(args.out)
        np.save(out_path, {"probs71": probs71, "probs7": probs7, "names": names,
                           "classes": CLASS_ORDER})
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
