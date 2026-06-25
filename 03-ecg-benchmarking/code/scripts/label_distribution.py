"""
Computes class label distribution for PTB-XL fold 10, CPSC 2018, and Georgia.
Reads only metadata (no signal loading) — runs in seconds.

Output:
  results/label_distribution.json
  results/plots/class_distribution.png
  results/plots/others_breakdown.png

Usage:
    python src/eval/label_distribution.py
"""

import ast
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
PLOT_DIR  = ROOT / "results" / "plots"
OUT_JSON  = ROOT / "results" / "label_distribution.json"

CLASS_ORDER = ['NORM', 'AFIB', 'AFLT', '1dAVb', 'RBBB', 'LBBB', 'OTHERS']

# ── PTB-XL mappings ───────────────────────────────────────────────────────────

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

# ── SNOMED mappings (CPSC 2018 + Georgia / PhysioNet Challenge 2020) ──────────

SNOMED_TO_CLASS = {
    # Named classes
    '426783006': 'NORM',
    '164889003': 'AFIB',
    '164890007': 'AFLT',
    '270492004': '1dAVb',
    '59118001':  'RBBB',
    '713427006': 'RBBB',
    '713426002': 'RBBB',
    '164909002': 'LBBB',
    '251146004': 'LBBB',
    # CPSC OTHERS
    '284470004': 'OTHERS',  # PAC
    '17338001':  'OTHERS',  # PVC
    '427172004': 'OTHERS',  # PVC (alt)
    '164884008': 'OTHERS',  # Ventricular premature beats
    '429622005': 'OTHERS',  # ST depression
    '164931005': 'OTHERS',  # ST elevation
    '164930006': 'OTHERS',  # ST elevation (alt)
    # Georgia / Challenge 2020 OTHERS
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
    '55827005':  'OTHERS',  # Left ventricular hypertrophy
    '164873001': 'OTHERS',  # LVH (alt)
    '89792004':  'OTHERS',  # Right ventricular hypertrophy
    '6374002':   'OTHERS',  # Bundle branch block NOS
    '233917008': 'OTHERS',  # AV block NOS
    '195042002': 'OTHERS',  # 2nd degree AV block
    '54016002':  'OTHERS',  # Mobitz II AV block
    '27885002':  'OTHERS',  # 3rd degree AV block
    '195060002': 'OTHERS',  # Supraventricular tachycardia
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
    '57054005':  'OTHERS',  # Acute myocardial infarction
    '164917005': 'OTHERS',  # Prolonged QRS
    '11157007':  'OTHERS',  # Ventricular bigeminy
    '164912004': 'OTHERS',  # U wave change
}

SNOMED_NAMES = {
    '426783006': 'Normal sinus rhythm',
    '164889003': 'Atrial fibrillation',
    '164890007': 'Atrial flutter',
    '270492004': '1st-degree AV block',
    '59118001':  'RBBB',
    '713427006': 'Complete RBBB',
    '713426002': 'Incomplete RBBB',
    '164909002': 'LBBB',
    '251146004': 'Complete LBBB',
    '284470004': 'PAC',
    '17338001':  'PVC',
    '427172004': 'PVC (alt)',
    '164884008': 'Ventricular premature beats',
    '429622005': 'ST depression',
    '164931005': 'ST elevation',
    '164930006': 'ST elevation (alt)',
    '164934002': 'T wave abnormal',
    '59931005':  'Inverted T wave',
    '164947007': 'Prolonged QT interval',
    '111975006': 'Prolonged QT interval (alt)',
    '698252002': 'Nonspecific IVCD',
    '426648003': 'Left atrial enlargement',
    '445118002': 'Left anterior fascicular block',
    '39732003':  'Left axis deviation',
    '47665007':  'Right axis deviation',
    '251200008': 'Abnormal ECG NOS',
    '55827005':  'LVH',
    '164873001': 'LVH (alt)',
    '89792004':  'Right ventricular hypertrophy',
    '6374002':   'Bundle branch block NOS',
    '233917008': 'AV block NOS',
    '195042002': '2nd degree AV block',
    '54016002':  'Mobitz II AV block',
    '27885002':  '3rd degree AV block',
    '195060002': 'SVT',
    '426761007': 'SVT (alt)',
    '713422000': 'Atrial tachycardia',
    '426995002': 'Junctional rhythm',
    '10370003':  'Pacing rhythm',
    '164912004': 'U wave abnormal',
    '17366009':  'Atrial arrhythmia NOS',
    '67198005':  'Sick sinus syndrome',
    '426177001': 'Sinus bradycardia',
    '427084000': 'Sinus tachycardia',
    '164865005': 'Myocardial infarction NOS',
    '57054005':  'Acute MI',
    '164917005': 'Prolonged QRS',
    '11157007':  'Ventricular bigeminy',
}


# ── PTB-XL analysis ───────────────────────────────────────────────────────────

def _ptbxl_counts_for_rows(rows: pd.DataFrame, threshold: float) -> tuple:
    """Return (n, class_counts, others_codes, multilabel, unmapped_codes) for a set of rows."""
    class_counts   = Counter()
    others_codes   = Counter()
    multilabel     = Counter()
    unmapped_codes = Counter()
    n = len(rows)
    for _, row in rows.iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        present = {c for c, lh in codes.items() if lh >= threshold}
        record_classes = set()
        has_others = False
        for code in present:
            cls = PTBXL_SCP_TO_CLASS.get(code)
            if cls:
                record_classes.add(cls)
            else:
                has_others = True
                others_codes[code] += 1
        if has_others:
            record_classes.add('OTHERS')
        for cls in record_classes:
            class_counts[cls] += 1
        multilabel[len(record_classes)] += 1
    return n, class_counts, others_codes, multilabel, unmapped_codes


def analyze_ptbxl(threshold: float = 100.0) -> dict:
    """Analyze PTB-XL with Train/Val/Test split breakdown.

    threshold=100.0 matches the training scripts (_scp_to_multilabel uses lh >= 100.0).
    Train = folds 1-8 | Val = fold 9 | Test = fold 10
    """
    ptbxl_dir = ROOT / "data/raw/ptbxl"
    db = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")

    train_rows = db[db["strat_fold"].isin(range(1, 9))]
    val_rows   = db[db["strat_fold"] == 9]
    test_rows  = db[db["strat_fold"] == 10]

    splits = {}
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        n, cc, oc, ml, uc = _ptbxl_counts_for_rows(rows, threshold)
        splits[split_name] = _build_result(n, cc, oc, ml, uc)

    # Print a compact Train / Val / Test table per class
    print(f"\n  PTB-XL class distribution (threshold={threshold}) — Train/Val/Test")
    print(f"  {'Class':<8} {'Train':>8} {'Val':>6} {'Test':>6}   "
          f"{'Train%':>7} {'Val%':>6} {'Test%':>6}")
    print(f"  {'-'*56}")
    for cls in CLASS_ORDER:
        tr = splits["train"]["mapped_classes"][cls]["n_records"]
        va = splits["val"]["mapped_classes"][cls]["n_records"]
        te = splits["test"]["mapped_classes"][cls]["n_records"]
        tr_pct = splits["train"]["mapped_classes"][cls]["pct"]
        va_pct = splits["val"]["mapped_classes"][cls]["pct"]
        te_pct = splits["test"]["mapped_classes"][cls]["pct"]
        print(f"  {cls:<8} {tr:>8,} {va:>6,} {te:>6,}   "
              f"{tr_pct:>6.1f}% {va_pct:>5.1f}% {te_pct:>5.1f}%")
    print(f"  {'TOTAL':<8} {splits['train']['n_records']:>8,} "
          f"{splits['val']['n_records']:>6,} {splits['test']['n_records']:>6,}")

    # Also return fold-10-only result for backward compat with plots/JSON
    n10, cc10, oc10, ml10, uc10 = _ptbxl_counts_for_rows(test_rows, threshold)
    result = _build_result(n10, cc10, oc10, ml10, uc10)
    result["splits"] = splits
    result["threshold_used"] = threshold
    return result


# ── Shared SNOMED .hea parser ─────────────────────────────────────────────────

def _analyze_hea_dir(root_dir: Path) -> dict:
    """
    Scan all .hea files under root_dir (any depth), parse #Dx: SNOMED codes,
    and return a result dict via _build_result().
    Records whose codes are all unmapped are counted as OTHERS (catch-all).
    """
    class_counts   = Counter()
    others_codes   = Counter()
    multilabel     = Counter()
    unmapped_codes = Counter()
    n = 0

    for hea in sorted(root_dir.rglob("*.hea")):
        text = hea.read_text(errors='ignore')
        dx_codes = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('#') and 'Dx' in line:
                codes_str = line.split(':', 1)[1].strip()
                dx_codes = [c.strip() for c in codes_str.split(',') if c.strip()]
                break
        if not dx_codes:
            continue
        n += 1
        record_classes = set()
        record_unmapped = []
        for code in dx_codes:
            cls = SNOMED_TO_CLASS.get(code)
            if cls:
                record_classes.add(cls)
                if cls == 'OTHERS':
                    others_codes[SNOMED_NAMES.get(code, code)] += 1
            else:
                record_unmapped.append(code)
                unmapped_codes[code] += 1

        # Catch-all: if nothing mapped, treat the whole record as OTHERS
        if not record_classes and record_unmapped:
            record_classes.add('OTHERS')
            for code in record_unmapped:
                others_codes[f"[unmapped] {code}"] += 1

        for cls in record_classes:
            class_counts[cls] += 1
        multilabel[len(record_classes)] += 1

    return _build_result(n, class_counts, others_codes, multilabel, unmapped_codes)


# ── CPSC analysis ─────────────────────────────────────────────────────────────

def analyze_cpsc() -> dict:
    cpsc_dir = ROOT / "data/raw/cpsc2018"
    if not cpsc_dir.exists():
        return None
    return _analyze_hea_dir(cpsc_dir)


# ── Georgia analysis ───────────────────────────────────────────────────────────

def analyze_georgia() -> dict:
    georgia_dir = ROOT / "data/raw/georgia"
    if not georgia_dir.exists():
        return None
    return _analyze_hea_dir(georgia_dir)


# ── Shared builder ────────────────────────────────────────────────────────────

def _build_result(n, class_counts, others_codes, multilabel, unmapped_codes) -> dict:
    mapped = {
        cls: {
            "n_records": class_counts.get(cls, 0),
            "pct":       round(class_counts.get(cls, 0) / n * 100, 1) if n else 0,
        }
        for cls in CLASS_ORDER
    }

    others_breakdown = {
        code: {"n_records": cnt, "pct": round(cnt / n * 100, 1)}
        for code, cnt in sorted(others_codes.items(), key=lambda x: -x[1])
    }

    return {
        "n_records":        n,
        "mapped_classes":   mapped,
        "others_breakdown": others_breakdown,
        "unmapped_codes":   dict(unmapped_codes.most_common(20)),
        "multilabel_dist":  {str(k): v for k, v in sorted(multilabel.items())},
        "note": (
            "n_records per class counts records with that label=1. "
            "A record can have multiple classes (multilabel)."
        ),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_ptbxl_splits(splits: dict, out_dir: Path) -> None:
    """Dedicated bar chart for PTB-XL Train / Val / Test per class."""
    x      = np.arange(len(CLASS_ORDER))
    w      = 0.25
    colors = ['#4C72B0', '#DD8452', '#55A868']
    split_labels = ['Train (folds 1-8)', 'Val (fold 9)', 'Test (fold 10)']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_idx, (ylabel, key) in enumerate([("Records with label = 1", "n_records"),
                                             ("% of split records",     "pct")]):
        ax = axes[ax_idx]
        for i, (sname, color, slabel) in enumerate(
                zip(['train', 'val', 'test'], colors, split_labels)):
            vals = [splits[sname]["mapped_classes"][c][key] for c in CLASS_ORDER]
            offset = (i - 1) * w
            bars = ax.bar(x + offset, vals, w, label=slabel, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.3 if key == "pct" else 2),
                            str(v) if key == "n_records" else f"{v:.1f}%",
                            ha='center', va='bottom', fontsize=6.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_ORDER)
        ax.set_ylabel(ylabel)
        suffix = "absolute counts" if key == "n_records" else "% of split"
        ax.set_title(f"PTB-XL Class Distribution — Train/Val/Test ({suffix})")
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend()

    fig.tight_layout()
    p = out_dir / "ptbxl_split_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved -> {p}")


def plot_distribution(results: dict, out_dir: Path) -> None:
    datasets = {k: v for k, v in results["datasets"].items() if v}
    if not datasets:
        return

    # ── Figure 1: class counts per dataset ──
    n_ds  = len(datasets)
    x     = np.arange(len(CLASS_ORDER))
    w     = 0.8 / n_ds
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#9467BD']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (ds_name, data) in enumerate(datasets.items()):
        counts = [data["mapped_classes"][c]["n_records"] for c in CLASS_ORDER]
        offset = (i - (n_ds - 1) / 2) * w
        bars = ax.bar(x + offset, counts, w,
                      label=ds_name, color=colors[i % len(colors)], alpha=0.85)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(cnt), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("Records with label = 1")
    ax.set_title("Class Label Distribution (records per class, multilabel)")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()

    # Highlight OTHERS bar
    ax.axvspan(len(CLASS_ORDER) - 1 - 0.5, len(CLASS_ORDER) - 0.5,
               color='gold', alpha=0.12, zorder=0)
    ax.text(len(CLASS_ORDER) - 1, ax.get_ylim()[1] * 0.97,
            'OTHERS\n(catch-all)', ha='center', va='top', fontsize=8,
            color='goldenrod')

    fig.tight_layout()
    p = out_dir / "class_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved -> {p}")

    # ── Figure 2: OTHERS breakdown ──
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, (ds_name, data) in zip(axes, datasets.items()):
        breakdown = data["others_breakdown"]
        if not breakdown:
            ax.set_visible(False)
            continue
        labels = list(breakdown.keys())
        counts = [breakdown[l]["n_records"] for l in labels]
        # keep top 15
        if len(labels) > 15:
            labels, counts = labels[:15], counts[:15]
        y = np.arange(len(labels))
        ax.barh(y, counts, color='#C44E52', alpha=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Records")
        ax.set_title(f"OTHERS breakdown — {ds_name}")
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    p = out_dir / "others_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved -> {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Analyzing PTB-XL (Train folds 1-8 / Val fold 9 / Test fold 10) ...")
    ptbxl = analyze_ptbxl()
    # Full Train/Val/Test table printed inside analyze_ptbxl()

    print("Analyzing CPSC 2018 ...")
    cpsc = analyze_cpsc()
    if cpsc:
        print(f"  {cpsc['n_records']} records, "
              f"AFLT={cpsc['mapped_classes']['AFLT']['n_records']}, "
              f"OTHERS={cpsc['mapped_classes']['OTHERS']['n_records']} "
              f"({cpsc['mapped_classes']['OTHERS']['pct']}%)")
    else:
        print("  not found — skipping")

    print("Analyzing Georgia (PhysioNet Challenge 2020) ...")
    georgia = analyze_georgia()
    if georgia:
        print(f"  {georgia['n_records']} records, "
              f"AFLT={georgia['mapped_classes']['AFLT']['n_records']}, "
              f"OTHERS={georgia['mapped_classes']['OTHERS']['n_records']} "
              f"({georgia['mapped_classes']['OTHERS']['pct']}%)")
        if georgia['unmapped_codes']:
            top = list(georgia['unmapped_codes'].items())[:5]
            print(f"  top unmapped codes: {top}")
    else:
        print("  not found — skipping")

    splits = ptbxl["splits"]

    results = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mapping": {
            "named_classes": {
                "NORM":  ["NORM (PTB-XL)", "426783006 (SNOMED)"],
                "AFIB":  ["AFIB (PTB-XL)", "164889003 (SNOMED)"],
                "AFLT":  ["AFLT (PTB-XL)", "164890007 (SNOMED)"],
                "1dAVb": ["1AVB (PTB-XL)", "270492004 (SNOMED)"],
                "RBBB":  ["CRBBB/IRBBB (PTB-XL)", "59118001/713427006/713426002 (SNOMED)"],
                "LBBB":  ["CLBBB/ILBBB (PTB-XL)", "164909002/251146004 (SNOMED)"],
            },
            "OTHERS_rule": (
                "PTB-XL: any SCP code not in the 8 named-class codes. "
                "CPSC/Georgia: any mapped OTHERS SNOMED code, plus any record "
                "whose codes are entirely unmapped (catch-all)."
            ),
        },
        # ptbxl_splits: Train/Val/Test breakdown (for data analysis)
        "ptbxl_splits": {
            "PTB-XL train": splits["train"],
            "PTB-XL val":   splits["val"],
            "PTB-XL test":  splits["test"],
        },
        # datasets: all 5 shown together in class_distribution.png
        "datasets": {
            "PTB-XL train": splits["train"],
            "PTB-XL val":   splits["val"],
            "PTB-XL test":  splits["test"],
            "CPSC2018":     cpsc,
            "Georgia":      georgia,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  JSON saved -> {OUT_JSON}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ptbxl_splits(splits, PLOT_DIR)        # ptbxl_split_distribution.png
    plot_distribution(results, PLOT_DIR)        # class_distribution.png (test+CPSC+Georgia)


if __name__ == "__main__":
    main()
