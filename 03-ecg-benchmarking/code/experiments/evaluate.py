"""
Evaluation script for pretrained Fraunhofer models:
  xresnet1d | inception1d | resnet1d | bilstm1d
...

Computes:
  - Macro-averaged AUROC  (primary)
  - Macro-averaged F1     (secondary, threshold 0.5)
  - Per-class sensitivity, specificity, support
  - Confusion matrix
  - Low-confidence flagging rate (max prob < 0.60)

Usage:
    python src/eval/evaluate.py --model xresnet1d --dataset ptbxl
    python src/eval/evaluate.py --model xresnet1d --dataset cpsc
    python src/eval/evaluate.py --model xresnet1d --dataset georgia
    python src/eval/evaluate.py --model xresnet1d --dataset all
    python src/eval/evaluate.py --model all --dataset all
"""

import argparse
import ast
import importlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCALER_PATH = ROOT / "ecg_ptbxl_benchmarking/output/exp0/data/standard_scaler.pkl"

CLASS_ORDER = ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB', 'OTHERS']

# ── PTB-XL label extraction ───────────────────────────────────────────────────

# SCP codes that map directly to one of our 6 named classes
PTBXL_SCP_TO_CLASS = {
    'NORM':  'NORM',
    'AFIB':  'AFIB',
    'AFLT':  'AFLT',
    '1AVB':  '1dAVb',
    'CRBBB': 'RBBB',
    'IRBBB': 'RBBB',
    'CLBBB': 'LBBB',
    'ILBBB': 'LBBB',
}

def ptbxl_labels(scp_codes_str: str, threshold: float = 100.0) -> np.ndarray:
    """Parse scp_codes dict string → (7,) binary label vector."""
    codes = ast.literal_eval(scp_codes_str)
    present = {c for c, lh in codes.items() if lh >= threshold}
    y = np.zeros(7, dtype=np.float32)
    has_named = set()
    for code in present:
        cls = PTBXL_SCP_TO_CLASS.get(code)
        if cls:
            y[CLASS_ORDER.index(cls)] = 1
            has_named.add(cls)
    # OTHERS = 1 if any code is NOT in the named-class groups
    others_codes = present - set(PTBXL_SCP_TO_CLASS.keys())
    if others_codes:
        y[CLASS_ORDER.index('OTHERS')] = 1
    return y


# ── CPSC 2018 label extraction ────────────────────────────────────────────────

# SNOMED-CT codes used in PhysioNet Challenge 2020 for CPSC 2018 records
SNOMED_TO_CLASS = {
    # Named classes (CPSC + Georgia)
    '426783006': 'NORM',    # Normal sinus rhythm
    '164889003': 'AFIB',    # Atrial fibrillation
    '164890007': 'AFLT',    # Atrial flutter
    '270492004': '1dAVb',   # First-degree AV block
    '59118001':  'RBBB',    # Right bundle branch block
    '713427006': 'RBBB',    # Complete RBBB
    '713426002': 'RBBB',    # Incomplete RBBB
    '164909002': 'LBBB',    # Left bundle branch block
    '251146004': 'LBBB',    # Complete LBBB
    # CPSC OTHERS
    '284470004': 'OTHERS',  # PAC
    '17338001':  'OTHERS',  # PVC (alt code)
    '427172004': 'OTHERS',  # PVC
    '164884008': 'OTHERS',  # Ventricular premature beats (PVC variant)
    '429622005': 'OTHERS',  # ST depression
    '164931005': 'OTHERS',  # ST elevation
    '164930006': 'OTHERS',  # ST elevation (alt)
    # Georgia / PhysioNet Challenge 2020 OTHERS
    '164934002': 'OTHERS',  # T wave abnormal
    '59931005':  'OTHERS',  # Inverted T wave
    '164947007': 'OTHERS',  # Prolonged QT
    '111975006': 'OTHERS',  # Prolonged QT (alt)
    '698252002': 'OTHERS',  # Nonspecific IVCD
    '426648003': 'OTHERS',  # Left atrial enlargement
    '445118002': 'OTHERS',  # Left anterior fascicular block
    '39732003':  'OTHERS',  # Left axis deviation
    '47665007':  'OTHERS',  # Right axis deviation
    '251200008': 'OTHERS',  # Abnormal ECG NOS
    '55827005':  'OTHERS',  # LVH
    '164873001': 'OTHERS',  # LVH (alt)
    '89792004':  'OTHERS',  # Right ventricular hypertrophy
    '6374002':   'OTHERS',  # Bundle branch block NOS
    '233917008': 'OTHERS',  # AV block NOS
    '195042002': 'OTHERS',  # 2nd degree AV block
    '54016002':  'OTHERS',  # Mobitz II AV block
    '27885002':  'OTHERS',  # 3rd degree AV block
    '195060002': 'OTHERS',  # SVT
    '426761007': 'OTHERS',  # SVT (alt)
    '713422000': 'OTHERS',  # Atrial tachycardia
    '426995002': 'OTHERS',  # Junctional rhythm
    '10370003':  'OTHERS',  # Pacing rhythm
    '164912004': 'OTHERS',  # U wave abnormal
    '17366009':  'OTHERS',  # Atrial arrhythmia NOS
    '67198005':  'OTHERS',  # Sick sinus syndrome
    '426177001': 'OTHERS',  # Sinus bradycardia
    '427084000': 'OTHERS',  # Sinus tachycardia
    '164865005': 'OTHERS',  # Myocardial infarction NOS
    '57054005':  'OTHERS',  # Acute MI
    '164917005': 'OTHERS',  # Prolonged QRS
    '11157007':  'OTHERS',  # Ventricular bigeminy
    '428750005': 'OTHERS',  # Myocardial ischaemia
    '425623009': 'OTHERS',  # Low QRS voltages
    '427393009': 'OTHERS',  # Sinus arrhythmia
    '425419005': 'OTHERS',  # Atypical T wave abnormality
    '67741000119109': 'OTHERS',  # ST/T change NOS
}

def cpsc_labels(hea_path: Path) -> np.ndarray:
    """Parse .hea file Dx line → (7,) binary label vector."""
    y = np.zeros(7, dtype=np.float32)
    try:
        text = hea_path.read_text(errors='ignore')
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('#') and 'Dx' in line:
                # format: "# Dx: 270492004,164889003"
                codes_str = line.split(':', 1)[1].strip()
                for code in codes_str.split(','):
                    code = code.strip()
                    cls = SNOMED_TO_CLASS.get(code)
                    if cls:
                        y[CLASS_ORDER.index(cls)] = 1
    except Exception:
        pass
    return y


# ── Data loading ─────────────────────────────────────────────────────────────

def load_scaler():
    import pickle
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_ptbxl_fold10(apply_scale=True):
    """Returns signals (N,12,1000) float32 and labels (N,7) float32."""
    import wfdb
    ptbxl_dir = ROOT / "data/raw/ptbxl"
    db = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    fold10 = db[db["strat_fold"] == 10]

    scaler = load_scaler() if apply_scale else None
    signals, labels, names = [], [], []
    for _, row in fold10.iterrows():
        path = ptbxl_dir / row["filename_lr"]
        sig, _ = wfdb.rdsamp(str(path))          # (1000, 12)
        x = sig.T.astype(np.float32)             # (12, 1000)
        if scaler:
            x = ((x - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)
        signals.append(x)
        labels.append(ptbxl_labels(row["scp_codes"]))
        names.append(row["filename_lr"])

    return np.stack(signals), np.stack(labels), names


def load_cpsc2018(apply_scale=True):
    """Returns signals (N,12,1000) float32 and labels (N,7) float32.
    Resamples from 500 Hz to 100 Hz using polyphase filter.
    """
    import wfdb
    cpsc_dir = ROOT / "data/raw/cpsc2018"
    if not cpsc_dir.exists():
        print("  [skip] CPSC 2018 not found at", cpsc_dir)
        return None, None, None

    scaler = load_scaler() if apply_scale else None
    signals, labels, names = [], [], []

    for gdir in sorted(cpsc_dir.iterdir()):
        if not gdir.is_dir():
            continue
        for hea in sorted(gdir.glob("*.hea")):
            rec_path = hea.with_suffix("")       # drop .hea → wfdb record path
            try:
                sig, fields = wfdb.rdsamp(str(rec_path))   # (N_samples, 12)
            except Exception as e:
                print(f"  [warn] {rec_path.name}: {e}")
                continue

            x = sig.T.astype(np.float32)         # (12, N_samples)

            # Resample to 100 Hz (CPSC is 500 Hz)
            fs = fields.get("fs", 500)
            if fs != 100:
                down = int(round(fs / 100))
                x = resample_poly(x, up=1, down=down, axis=1).astype(np.float32)

            # Trim or pad to exactly 1000 samples
            if x.shape[1] >= 1000:
                x = x[:, :1000]
            else:
                x = np.pad(x, ((0, 0), (0, 1000 - x.shape[1])))

            if scaler:
                x = ((x - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)

            if np.isnan(x).any() or np.isinf(x).any():
                print(f"  [warn] {hea.stem}: NaN/Inf after scaling — skipped")
                continue

            signals.append(x)
            labels.append(cpsc_labels(hea))
            names.append(hea.stem)

    if not signals:
        print("  [skip] No CPSC records found. Is the download complete?")
        return None, None, None

    return np.stack(signals), np.stack(labels), names


def load_georgia(apply_scale=True):
    """Returns signals (N,12,1000) float32 and labels (N,7) float32.
    Georgia (PhysioNet Challenge 2020) is 500 Hz — resampled to 100 Hz.
    """
    import wfdb
    georgia_dir = ROOT / "data/raw/georgia"
    if not georgia_dir.exists():
        print("  [skip] Georgia not found at", georgia_dir)
        return None, None, None

    scaler = load_scaler() if apply_scale else None
    signals, labels, names = [], [], []

    for gdir in sorted(georgia_dir.iterdir()):
        if not gdir.is_dir():
            continue
        for hea in sorted(gdir.glob("*.hea")):
            rec_path = hea.with_suffix("")
            try:
                sig, fields = wfdb.rdsamp(str(rec_path))   # (N_samples, 12)
            except Exception as e:
                print(f"  [warn] {rec_path.name}: {e}")
                continue

            x = sig.T.astype(np.float32)         # (12, N_samples)

            # Resample to 100 Hz (Georgia is 500 Hz)
            fs = fields.get("fs", 500)
            if fs != 100:
                down = int(round(fs / 100))
                x = resample_poly(x, up=1, down=down, axis=1).astype(np.float32)

            # Trim or pad to exactly 1000 samples
            if x.shape[1] >= 1000:
                x = x[:, :1000]
            else:
                x = np.pad(x, ((0, 0), (0, 1000 - x.shape[1])))

            if scaler:
                x = ((x - scaler.mean_[0]) / scaler.scale_[0]).astype(np.float32)

            if np.isnan(x).any() or np.isinf(x).any():
                print(f"  [warn] {hea.stem}: NaN/Inf after scaling — skipped")
                continue

            signals.append(x)
            labels.append(cpsc_labels(hea))      # same SNOMED format as CPSC
            names.append(hea.stem)

    if not signals:
        print("  [skip] No Georgia records found. Is the download complete?")
        return None, None, None

    return np.stack(signals), np.stack(labels), names


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    dataset_name: str, threshold: float = 0.5,
                    flag_threshold: float = 0.60) -> dict:
    """
    y_true:  (N, 7) int/float binary labels
    y_score: (N, 7) sigmoid probabilities
    Returns dict of metric values and a formatted report string.
    """
    y_pred = (y_score >= threshold).astype(int)
    n = len(y_true)

    # ── AUROC (skip classes with no positive samples) ──
    valid_cols = [j for j in range(7) if y_true[:, j].sum() > 0]
    if len(valid_cols) == 0:
        auroc_macro = float('nan')
        auroc_per   = [float('nan')] * 7
    else:
        auroc_per = []
        for j in range(7):
            if y_true[:, j].sum() == 0:
                auroc_per.append(float('nan'))
            else:
                try:
                    auroc_per.append(roc_auc_score(y_true[:, j], y_score[:, j]))
                except Exception:
                    auroc_per.append(float('nan'))
        valid_vals = [v for v in auroc_per if not np.isnan(v)]
        auroc_macro = float(np.mean(valid_vals)) if valid_vals else float('nan')

    # ── Macro F1 ──
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None,    zero_division=0)

    # ── Per-class sensitivity & specificity ──
    mcm = multilabel_confusion_matrix(y_true, y_pred)   # (7, 2, 2)
    sensitivity, specificity = [], []
    for j in range(7):
        tn, fp, fn, tp = mcm[j].ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        sensitivity.append(sens)
        specificity.append(spec)

    # ── Low-confidence flagging ──
    max_prob = y_score.max(axis=1)                        # (N,)
    flag_rate = float((max_prob < flag_threshold).mean())

    # ── Format report ──
    sep = "-" * 74
    lines = [
        f"\n{'='*74}",
        f"  Dataset : {dataset_name}   N={n}",
        f"{'='*74}",
        f"  Macro AUROC : {auroc_macro:.4f}",
        f"  Macro F1    : {f1_macro:.4f}",
        f"  Low-conf flag rate (max prob < {flag_threshold:.2f}) : "
        f"{flag_rate*100:.1f}%  ({int(flag_rate*n)} / {n})",
        f"\n  {'Class':<8} {'AUROC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}",
        f"  {sep}",
    ]
    for j, cls in enumerate(CLASS_ORDER):
        lines.append(
            f"  {cls:<8} "
            f"{auroc_per[j]:>7.4f} "
            f"{f1_per[j]:>7.4f} "
            f"{sensitivity[j]:>7.4f} "
            f"{specificity[j]:>7.4f}"
        )

    # Aggregate confusion matrix
    # Flatten to binary: is the record correctly classified at all?
    any_correct = ((y_pred & y_true.astype(int)).sum(axis=1) > 0)
    any_predicted = (y_pred.sum(axis=1) > 0)
    lines += [
        f"\n  {sep}",
        f"  Records with >=1 correct class hit : "
        f"{any_correct.sum()} / {n}  ({any_correct.mean()*100:.1f}%)",
        f"  Records with no prediction         : "
        f"{(~any_predicted.astype(bool)).sum()} / {n}",
        f"{'='*74}",
    ]

    return {
        "n":           n,
        "auroc_macro": auroc_macro,
        "f1_macro":    f1_macro,
        "auroc_per":   auroc_per,
        "f1_per":      list(f1_per),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "flag_rate":   flag_rate,
        "flag_count":  int(flag_rate * n),
        "mcm":         mcm,
        "report":      "\n".join(lines),
    }


# ── JSON serialization ────────────────────────────────────────────────────────

def _jv(v):
    """Convert numpy scalar/array to JSON-safe Python type; NaN -> null."""
    if isinstance(v, np.ndarray):
        return [[_jv(cell) for cell in row] if hasattr(row, '__iter__') else _jv(row)
                for row in v]
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(round(v, 6))
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _detail_entry(metrics: dict) -> dict:
    per_class = {}
    for j, cls in enumerate(CLASS_ORDER):
        tn = int(metrics["mcm"][j][0][0])
        fp = int(metrics["mcm"][j][0][1])
        fn = int(metrics["mcm"][j][1][0])
        tp = int(metrics["mcm"][j][1][1])
        per_class[cls] = {
            "auroc":       _jv(metrics["auroc_per"][j]),
            "f1":          _jv(metrics["f1_per"][j]),
            "sensitivity": _jv(metrics["sensitivity"][j]),
            "specificity": _jv(metrics["specificity"][j]),
            "support":     tp + fn,   # total positive samples for this class
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    return {
        "n":              metrics["n"],
        "auroc_macro":    _jv(metrics["auroc_macro"]),
        "f1_macro":       _jv(metrics["f1_macro"]),
        "flag_rate":      _jv(metrics["flag_rate"]),
        "flag_count":     metrics["flag_count"],
        "per_class":      per_class,
    }


def _summary_entry(metrics: dict) -> dict:
    return {
        "n":             metrics["n"],
        "auroc_macro":   _jv(metrics["auroc_macro"]),
        "f1_macro":      _jv(metrics["f1_macro"]),
        "flag_rate_pct": round(float(metrics["flag_rate"]) * 100, 1),
        "flag_count":    metrics["flag_count"],
    }


def _load_or_empty(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"threshold": 0.5, "flag_threshold": 0.60, "models": {}}


def save_jsons(all_results: dict, out_dir: Path) -> None:
    """Upsert results into existing JSONs — only the ran model/dataset entries are touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path  = out_dir / "results_detail.json"
    summary_path = out_dir / "results_summary.json"

    detail  = _load_or_empty(detail_path)
    summary = _load_or_empty(summary_path)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    detail["generated"]  = ts
    summary["generated"] = ts

    for model_name, ds_map in all_results.items():
        detail["models"].setdefault(model_name,  {})
        summary["models"].setdefault(model_name, {})
        for ds_name, metrics in ds_map.items():
            detail["models"][model_name][ds_name]  = _detail_entry(metrics)
            summary["models"][model_name][ds_name] = _summary_entry(metrics)

    detail_path.write_text(json.dumps(detail,  indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  JSON saved -> {detail_path}")
    print(f"  JSON saved -> {summary_path}")


# ── Inference wrapper ─────────────────────────────────────────────────────────

def get_model(model_name: str):
    module = importlib.import_module(f"src.models.{model_name}")
    return module.load_model(), module.run_inference, module.map_to_7


MODEL_CHOICES = ["resnet1d", "bilstm1d", "xresnet1d", "inception1d"]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval(model_name: str, dataset: str, save_dir: Path | None = None):
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    t0 = time.time()
    model, infer_fn, map7_fn = get_model(model_name)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    results = {}

    def _run(signals, labels, names, ds_name, batch_size=64):
        print(f"  Running inference on {ds_name} ({len(signals)} records)...")
        t1 = time.time()
        probs71 = infer_fn(model, signals, batch_size=batch_size)
        probs7  = map7_fn(probs71)
        print(f"  Inference done in {time.time()-t1:.1f}s")
        metrics = compute_metrics(labels, probs7, ds_name)
        print(metrics["report"])
        results[ds_name] = metrics
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"{model_name}_{ds_name}_probs.npy",
                    {"probs71": probs71, "probs7": probs7,
                     "y_true": labels, "names": names})
            print(f"  Saved predictions -> {save_dir}/{model_name}_{ds_name}_probs.npy")

    if dataset in ("ptbxl", "all"):
        print("\n  Loading PTB-XL fold 10 ...")
        t1 = time.time()
        signals, labels, names = load_ptbxl_fold10()
        print(f"  Loaded {len(signals)} records in {time.time()-t1:.1f}s")
        _run(signals, labels, names, "PTB-XL fold10")

    if dataset in ("cpsc", "all"):
        print("\n  Loading CPSC 2018 ...")
        t1 = time.time()
        signals, labels, names = load_cpsc2018()
        if signals is not None:
            print(f"  Loaded {len(signals)} records in {time.time()-t1:.1f}s")
            _run(signals, labels, names, "CPSC2018", batch_size=32)

    if dataset in ("georgia", "all"):
        print("\n  Loading Georgia (PhysioNet Challenge 2020) ...")
        t1 = time.time()
        signals, labels, names = load_georgia()
        if signals is not None:
            print(f"  Loaded {len(signals)} records in {time.time()-t1:.1f}s")
            _run(signals, labels, names, "Georgia", batch_size=32)

    print(f"\n  Total wall time: {time.time()-t0:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="xresnet1d",
                        choices=MODEL_CHOICES + ["all"])
    parser.add_argument("--dataset",  default="ptbxl",
                        choices=["ptbxl", "cpsc", "georgia", "all"])
    parser.add_argument("--save",     default=None,
                        help="Directory to save .npy prediction files")
    parser.add_argument("--json-dir", default="results",
                        help="Directory to save JSON result files (default: results)")
    args = parser.parse_args()

    save_dir = Path(args.save) if args.save else None
    json_dir = Path(args.json_dir)

    all_results = {}
    if args.model == "all":
        for m in MODEL_CHOICES:
            all_results[m] = run_eval(m, args.dataset, save_dir)
    else:
        all_results[args.model] = run_eval(args.model, args.dataset, save_dir)

    save_jsons(all_results, json_dir)


if __name__ == "__main__":
    main()
