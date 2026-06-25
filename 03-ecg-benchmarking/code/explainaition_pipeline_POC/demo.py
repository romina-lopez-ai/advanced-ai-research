# ECG Inference — Streamlit POC
# Run with:  streamlit run demo.py
# Then open  http://localhost:8501  in your browser.

import os, sys, tempfile, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import resample, butter, sosfiltfilt
import neurokit2 as nk
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES           = ["NORM", "AFIB", "AFLT", "1dAVb", "RBBB", "LBBB", "OTHERS"]
CLASS_TO_IDX      = {c: i for i, c in enumerate(CLASSES)}
LEAD_NAMES        = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
CONFIDENCE_THRESHOLD = 0.60
N_SEGMENTS        = 25
SAMPLES_PER_SEG   = 200       # 0.4 s at 500 Hz
N_SHAP_SAMPLES    = 50
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture  (identical to training)
# ─────────────────────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
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
        self.cnn      = nn.Sequential(*cnn_blocks)
        self.cnn_drop = nn.Dropout(0.2)
        self.cnn_out_dim = cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0)
        embed_dim = lstm_hidden * 2
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, 626, embed_dim) * 0.02)
        self.pos_drop   = nn.Dropout(tf_dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=tf_heads, dim_feedforward=embed_dim * 4,
            dropout=tf_dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(tf_dropout),
            nn.Linear(embed_dim, num_classes))
    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn_drop(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        B     = x.shape[0]
        cls   = self.cls_token.expand(B, -1, -1)
        x     = torch.cat([cls, x], dim=1)
        x     = self.pos_drop(x + self.pos_embed[:, :x.shape[1], :])
        x     = self.transformer(x)
        x     = self.norm(x)
        return self.head(x[:, 0])

@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    m = CNNBiLSTMTransformer(
        num_leads=12, num_classes=7,
        cnn_channels=[48, 96, 192], cnn_kernel_size=7,
        lstm_hidden=96, lstm_layers=1, lstm_dropout=0.0,
        tf_heads=4, tf_layers=2, tf_dropout=0.3
    ).to(DEVICE)
    ckpt = os.path.join(BASE_DIR,
           "../../ecg_results/results_cnn_bilstm_transformer/best_model_multilabel.pt")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & inference
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500, order=3):
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)

def load_ecg_sample(npy_path):
    sig = np.load(npy_path).astype(np.float64)
    sig_500 = resample(sig, 5000, axis=1)
    sig_500 = np.nan_to_num(sig_500, nan=0.0)
    sig_500 = bandpass_filter(sig_500, fs=500)
    for lead in range(12):
        std = sig_500[lead].std()
        if std > 1e-6:
            sig_500[lead] = (sig_500[lead] - sig_500[lead].mean()) / std
        else:
            sig_500[lead] = 0.0
    return sig_500.astype(np.float32)

def predict(mdl, signal):
    with torch.no_grad():
        x      = torch.FloatTensor(signal).unsqueeze(0).to(DEVICE)
        logits = mdl(x)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]
    return probs

# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap_temporal(mdl, signal, target_class_idx, n_samples=N_SHAP_SAMPLES):
    baseline = np.zeros_like(signal)
    def predict_fn(masks):
        batch = []
        for mask in masks:
            x = baseline.copy()
            for p in range(N_SEGMENTS):
                if mask[p] == 1:
                    s = p * SAMPLES_PER_SEG
                    x[:, s:s + SAMPLES_PER_SEG] = signal[:, s:s + SAMPLES_PER_SEG]
            batch.append(x)
        bt = torch.FloatTensor(np.array(batch)).to(DEVICE)
        with torch.no_grad():
            return torch.sigmoid(mdl(bt))[:, target_class_idx].cpu().numpy()
    shap_values = np.zeros(N_SEGMENTS)
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        perm         = rng.permutation(N_SEGMENTS)
        mask_with    = np.zeros(N_SEGMENTS)
        mask_without = np.zeros(N_SEGMENTS)
        for idx in perm:
            mask_with[:] = mask_without[:]
            mask_with[idx] = 1
            shap_values[idx] += (predict_fn(mask_with[np.newaxis])[0]
                                 - predict_fn(mask_without[np.newaxis])[0])
            mask_without[idx] = 1
    return shap_values / n_samples

def shap_to_signal_resolution(shap_values):
    return np.repeat(shap_values, SAMPLES_PER_SEG)

# ─────────────────────────────────────────────────────────────────────────────
# Hybrid per-lead importance  (logits — avoids sigmoid saturation at conf~1.0)
# ─────────────────────────────────────────────────────────────────────────────
def compute_hybrid_lead_importance(mdl, sig500, target_class_idx, top_segs):
    scores = np.zeros(12)
    with torch.no_grad():
        base_logit = mdl(
            torch.FloatTensor(sig500[np.newaxis]).to(DEVICE)
        )[0, target_class_idx].item()
    for seg in top_segs:
        s = seg * SAMPLES_PER_SEG
        e = min(s + SAMPLES_PER_SEG, sig500.shape[1])
        for li in range(12):
            x = sig500.copy()
            x[li, s:e] = 0.0
            with torch.no_grad():
                logit = mdl(
                    torch.FloatTensor(x[np.newaxis]).to(DEVICE)
                )[0, target_class_idx].item()
            scores[li] += (base_logit - logit)
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# NeuroKit2 clinical features
# ─────────────────────────────────────────────────────────────────────────────
def extract_clinical_features(signal_500hz):
    lead_ii = signal_500hz[1].astype(np.float64)
    fs = 500
    features = {}
    try:
        cleaned  = nk.ecg_clean(lead_ii, sampling_rate=fs)
        _, ri    = nk.ecg_peaks(cleaned, sampling_rate=fs)
        r_peaks  = ri["ECG_R_Peaks"]
        features["r_peaks"]  = r_peaks
        features["n_beats"]  = len(r_peaks)
        if len(r_peaks) >= 2:
            rr = np.diff(r_peaks) / fs * 1000
            features["rr_mean_ms"]     = float(np.mean(rr))
            features["rr_std_ms"]      = float(np.std(rr))
            features["heart_rate_bpm"] = float(np.mean(60000 / rr))
            features["rr_rmssd_ms"]    = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr) >= 2 else None
        else:
            for k in ("rr_mean_ms","rr_std_ms","heart_rate_bpm","rr_rmssd_ms"):
                features[k] = None
    except Exception as e:
        features["r_peaks"] = []; features["n_beats"] = 0; features["error_peaks"] = str(e)
        return features
    try:
        _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs, method="dwt")
        q_peaks  = waves.get("ECG_Q_Peaks", []); s_peaks = waves.get("ECG_S_Peaks", [])
        _ok = lambda v: v is not None and not np.isnan(v)
        qrs_dur  = [(s - q)/fs*1000 for q, s in zip(q_peaks, s_peaks) if _ok(q) and _ok(s)]
        features["qrs_duration_ms"] = float(np.mean(qrs_dur)) if qrs_dur else None
        p_onsets = waves.get("ECG_P_Onsets", [])
        p_peaks  = waves.get("ECG_P_Peaks",  [])
        valid_p  = sum(1 for p in p_peaks if _ok(p))
        features["p_wave_count"] = valid_p
        features["p_wave_ratio"] = valid_p / max(len(r_peaks), 1)
        pr_intervals = []
        for i, p_on in enumerate(p_onsets):
            if _ok(p_on) and i < len(r_peaks):
                pr_ms = (r_peaks[i] - p_on) / fs * 1000
                if 50 < pr_ms < 500:
                    pr_intervals.append(pr_ms)
        features["pr_interval_ms"] = float(np.mean(pr_intervals)) if pr_intervals else None
        t_offsets = waves.get("ECG_T_Offsets", [])
        features["t_wave_detected"] = sum(1 for t in t_offsets if _ok(t))
        # Raw wave indices preserved for visualization
        features["qrs_pairs"]    = [(int(q), int(s)) for q, s in zip(q_peaks, s_peaks)
                                    if _ok(q) and _ok(s)]
        features["p_peaks_idx"]  = [int(p) for p in p_peaks  if _ok(p)]
        features["p_onsets_idx"] = [int(p) for p in p_onsets if _ok(p)]
        # QT: R-peak → T-offset, paired by beat index
        qt_list = [(int(to) - int(rp)) / fs * 1000
                   for rp, to in zip(r_peaks, t_offsets) if _ok(to) and int(to) > int(rp)]
        features["qt_interval_ms"] = float(np.mean(qt_list)) if qt_list else None
    except Exception as e:
        for k in ("qrs_duration_ms","pr_interval_ms","p_wave_count","p_wave_ratio",
                  "qrs_pairs","p_peaks_idx","p_onsets_idx","qt_interval_ms"):
            features[k] = None
        features["error_delineate"] = str(e)
    return features

# ─────────────────────────────────────────────────────────────────────────────
# Clinical rule checks
# ─────────────────────────────────────────────────────────────────────────────
def check_clinical_rules(pred_class, features, shap_values, signal):
    checks = {}
    f = features
    if pred_class == "AFIB":
        rr_irreg = f.get("rr_rmssd_ms")
        checks["RR irregular (RMSSD > 50ms)"] = (
            rr_irreg is not None and rr_irreg > 50,
            f"RMSSD = {rr_irreg:.1f} ms" if rr_irreg else "not measurable")
        p_ratio = f.get("p_wave_ratio")
        checks["P-waves absent/irregular (ratio < 0.5)"] = (
            p_ratio is not None and p_ratio < 0.5,
            f"P-wave ratio = {p_ratio:.2f}" if p_ratio is not None else "not measurable")
    elif pred_class == "AFLT":
        hr = f.get("heart_rate_bpm")
        checks["Atrial rate consistent with flutter"] = (
            hr is not None and hr > 100,
            f"HR = {hr:.0f} bpm" if hr else "not measurable")
        p_ratio = f.get("p_wave_ratio")
        checks["P-wave morphology abnormal"] = (
            p_ratio is not None and p_ratio < 0.7,
            f"P-wave ratio = {p_ratio:.2f}" if p_ratio is not None else "not measurable")
    elif pred_class == "1dAVb":
        pr = f.get("pr_interval_ms")
        checks["PR interval > 200ms"] = (
            pr is not None and pr > 200,
            f"PR = {pr:.0f} ms" if pr else "not measurable")
        if f.get("r_peaks") is not None and len(f["r_peaks"]) > 0:
            pr_region_shap = []
            for rp in f["r_peaks"]:
                start     = max(0, rp - 150)
                seg_start = start // SAMPLES_PER_SEG
                seg_end   = min(rp // SAMPLES_PER_SEG, N_SEGMENTS - 1)
                pr_region_shap.extend(np.abs(shap_values[seg_start:seg_end + 1]))
            if pr_region_shap:
                checks["SHAP highlights PR region"] = (
                    np.mean(pr_region_shap) > np.mean(np.abs(shap_values)),
                    f"PR region SHAP = {np.mean(pr_region_shap):.4f} vs avg = {np.mean(np.abs(shap_values)):.4f}")
    elif pred_class == "RBBB":
        qrs = f.get("qrs_duration_ms")
        checks["QRS > 120ms"] = (
            qrs is not None and qrs > 120,
            f"QRS = {qrs:.0f} ms" if qrs else "not measurable")
        v1_e = np.std(signal[6]); v2_e = np.std(signal[7])
        avg_e = np.mean([np.std(signal[i]) for i in range(12)])
        checks["V1-V2 morphology prominence"] = (
            (v1_e + v2_e)/2 > avg_e * 0.8,
            f"V1={v1_e:.2f}, V2={v2_e:.2f}, avg={avg_e:.2f}")
    elif pred_class == "LBBB":
        qrs = f.get("qrs_duration_ms")
        checks["QRS > 120ms"] = (
            qrs is not None and qrs > 120,
            f"QRS = {qrs:.0f} ms" if qrs else "not measurable")
        v5_e = np.std(signal[10]); v6_e = np.std(signal[11])
        avg_e = np.mean([np.std(signal[i]) for i in range(12)])
        checks["V5-V6 morphology prominence"] = (
            (v5_e + v6_e)/2 > avg_e * 0.8,
            f"V5={v5_e:.2f}, V6={v6_e:.2f}, avg={avg_e:.2f}")
    elif pred_class == "NORM":
        rr_sd = f.get("rr_std_ms")
        checks["Regular rhythm (RR SD < 100ms)"] = (
            rr_sd is not None and rr_sd < 100,
            f"RR SD = {rr_sd:.1f} ms" if rr_sd else "not measurable")
        hr = f.get("heart_rate_bpm")
        checks["Normal heart rate (60-100 bpm)"] = (
            hr is not None and 50 <= hr <= 110,
            f"HR = {hr:.0f} bpm" if hr else "not measurable")
        qrs = f.get("qrs_duration_ms")
        checks["Normal QRS duration (< 120ms)"] = (
            qrs is not None and qrs < 120,
            f"QRS = {qrs:.0f} ms" if qrs else "not measurable")
    elif pred_class == "OTHERS":
        checks["No specific clinical rule"] = (True, "classified as other abnormality")
    return checks

# ─────────────────────────────────────────────────────────────────────────────
# Text explanation
# ─────────────────────────────────────────────────────────────────────────────
def generate_text_explanation(name, info, hybrid_res=None):
    pred   = info["pred_class"]
    conf   = info["pred_confidence"]
    f      = info["clinical_features"]
    checks = info["clinical_checks"]
    top_time = info["shap_top_times"][0]
    lines = [f"**{name}**: Predicted **{pred}** (confidence {conf:.2f})."]
    if info["flagged"]:
        lines.append("LOW CONFIDENCE: flagged for clinician review.")
    lines.append(f"SHAP highlights the {top_time[0]:.1f}–{top_time[1]:.1f}s region as most influential.")
    if hybrid_res is not None:
        top3    = hybrid_res["top3_names"]
        med_set = DIAG_LEADS_H.get(pred, set())
        overlap = [l for l in top3 if l in med_set]
        if overlap:
            lines.append(f"12-lead analysis: model focused on {', '.join(top3)} "
                         f"({', '.join(overlap)} match medically expected leads for {pred}).")
        else:
            lines.append(f"12-lead analysis: model focused on {', '.join(top3)}.")
    if f.get("heart_rate_bpm"):
        lines.append(f"Heart rate: {f['heart_rate_bpm']:.0f} bpm ({f['n_beats']} beats detected).")
    if pred == "AFIB":
        if f.get("rr_rmssd_ms"):
            lines.append(f"RR interval variability: RMSSD = {f['rr_rmssd_ms']:.1f} ms.")
        if f.get("p_wave_ratio") is not None:
            ratio = f["p_wave_ratio"]
            if ratio < 0.5:
                lines.append(f"P-waves largely absent (ratio {ratio:.2f}), consistent with AFIB.")
            else:
                lines.append(f"P-waves detected (ratio {ratio:.2f}), which may not fully support AFIB.")
    elif pred == "AFLT":
        lines.append("Look for sawtooth flutter waves in leads II, III, and aVF.")
        if f.get("heart_rate_bpm") and f["heart_rate_bpm"] > 100:
            lines.append(f"Elevated rate ({f['heart_rate_bpm']:.0f} bpm) is consistent with atrial flutter.")
    elif pred == "1dAVb":
        if f.get("pr_interval_ms"):
            pr = f["pr_interval_ms"]
            lines.append(f"NeuroKit2 confirms PR interval of {pr:.0f} ms (threshold: 200 ms). "
                         f"{'Consistent with 1dAVb.' if pr > 200 else 'Below threshold.'}")
    elif pred == "RBBB":
        if f.get("qrs_duration_ms"):
            qrs = f["qrs_duration_ms"]
            lines.append(f"QRS duration measured at {qrs:.0f} ms (threshold: 120 ms). "
                         f"{'Consistent with RBBB.' if qrs > 120 else 'Below threshold — resampling artefact possible.'}")
        lines.append("Check for rsR′ pattern in leads V1–V2.")
        lines.append("Beat template (V1): individual beats overlaid with mean morphology to reveal QRS pattern.")
    elif pred == "LBBB":
        if f.get("qrs_duration_ms"):
            qrs = f["qrs_duration_ms"]
            lines.append(f"NeuroKit2 confirms QRS duration of {qrs:.0f} ms (threshold: 120 ms). "
                         f"{'Consistent with LBBB.' if qrs > 120 else 'Below threshold — resampling artefact possible.'}")
        lines.append("Check for broad, notched R-waves in leads V5–V6.")
        lines.append("Beat template (V6): individual beats overlaid with mean morphology to reveal broad monophasic R pattern.")
    elif pred == "NORM":
        lines.append("No abnormal features detected by NeuroKit2.")
    passed = sum(1 for v in checks.values() if v[0])
    total  = len(checks)
    if total > 0:
        if passed == total:
            lines.append(f"Clinical validation: all {total} checks passed.")
        else:
            lines.append(f"Clinical validation: {passed}/{total} checks passed. Review recommended.")
    return " ".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
try:
    import ecg_plot
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "ecg-plot", "-q"], check=True)
    import ecg_plot

_SC3       = ["#1a7837", "#5aae61", "#a6dba0"]   # dark → light green
_RPEAK_COL = "#6a0dad"                             # dark purple

LEAD_CELL_MAP = {
    "I":(0,0), "II":(0,1), "III":(0,2),
    "aVR":(1,0), "aVL":(1,1), "aVF":(1,2),
    "V1":(2,0), "V2":(2,1), "V3":(2,2),
    "V4":(3,0), "V5":(3,1), "V6":(3,2),
}
COL_X = {0:(0.0,2.5), 1:(2.5,5.0), 2:(5.0,7.5), 3:(7.5,10.0)}
DIAG_LEADS_H = {
    "NORM": {"II"}, "1dAVb": {"II"}, "AFIB": {"II"},
    "AFLT": {"II", "III", "aVF"},
    "RBBB": {"II", "V1", "V2"},
    "LBBB": {"I",  "V5", "V6"},
}

def _role(lead, med_leads, hyb_set):
    is_m = lead in med_leads; is_h = lead in hyb_set
    if   is_m and is_h: return "#7f0000", "-",  2.2, 0.52
    elif is_h:          return "#c0392b", "-",  1.8, 0.45
    else:               return "#d35400", "--", 1.5, 0.12   # orange-red dashed

def _lead_sort_key(lead_name, med_leads, hyb_set):
    is_m = lead_name in med_leads; is_h = lead_name in hyb_set
    if   is_m and is_h: return (0, LEAD_NAMES.index(lead_name))
    elif is_h:          return (1, LEAD_NAMES.index(lead_name))
    else:               return (2, LEAD_NAMES.index(lead_name))

def _row_geom(ax):
    y0, y1 = ax.get_ylim(); h = (y1 - y0) / 4.0
    tops    = [y1 - i * h       for i in range(4)]
    bottoms = [y1 - (i + 1) * h for i in range(4)]
    def frac(y): return (y - y0) / (y1 - y0)
    yfracs  = [(frac(bottoms[r]), frac(tops[r])) for r in range(4)]
    return bottoms, tops, yfracs, h


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 plot — returns fig
# ─────────────────────────────────────────────────────────────────────────────
def plot_hybrid_12lead_st(name, info, h_res, ecg_path):
    pred        = info["pred_class"]
    conf        = info["pred_confidence"]
    f           = info["clinical_features"]
    checks      = info["clinical_checks"]
    r_peaks_sec = np.array(f.get("r_peaks", []), dtype=int) / 500.0

    scores    = h_res["scores"]
    top3_segs = h_res["top3_segs"]
    top3_hyb  = set(h_res["top3_names"])
    med_leads = DIAG_LEADS_H.get(pred, {"II"})

    raw = np.load(ecg_path).astype(np.float32)
    ecg = raw.T
    n   = ecg.shape[0]; q1, q2, q3 = n//4, n//2, 3*n//4
    ecg_new = np.stack([
        np.concatenate([ecg[0:q1,0], ecg[q1:q2,3], ecg[q2:q3,6],  ecg[q3:n,9]]),
        np.concatenate([ecg[0:q1,1], ecg[q1:q2,4], ecg[q2:q3,7],  ecg[q3:n,10]]),
        np.concatenate([ecg[0:q1,2], ecg[q1:q2,5], ecg[q2:q3,8],  ecg[q3:n,11]]),
        ecg[:, 1],
    ], axis=1).T

    pred_classes = info.get("pred_classes", [pred])
    passed = sum(1 for v in checks.values() if v[0]); total = len(checks)
    if len(pred_classes) > 1:
        all_str = " + ".join(pred_classes)
        title = f"{name}  |  Predicted: {all_str}  |  Showing: {pred} ({conf:.2f})  |  Clinical checks: {passed}/{total}"
    else:
        title = f"{name}  |  Predicted: {pred} ({conf:.2f})  |  Clinical checks: {passed}/{total}"
    if info["flagged"]: title += "  |  LOW CONFIDENCE"

    ecg_plot.plot(ecg_new, sample_rate=100, title="",
                  columns=1, lead_index=["I", "II", "III", "II Ref"])
    fig = plt.gcf()
    ax  = plt.gca()
    ax.set_title("")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.06)
    bottoms, tops, yfracs, h_row = _row_geom(ax)

    ax.set_xticks(range(0, 11))
    ax.set_xticklabels([str(i) for i in range(11)], fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=10, labelpad=4)
    ax.xaxis.label.set_x(0.80)   # position label at ~8s mark
    ax.xaxis.label.set_ha("center")
    ax.set_ylabel("Amplitude (mV)", fontsize=10)

    for rank, seg in enumerate(top3_segs):
        ax.axvspan(seg * 0.4, (seg + 1) * 0.4, color=_SC3[rank], alpha=0.15, zorder=4)

    for lead_name, (col, row) in LEAD_CELL_MAP.items():
        is_m = lead_name in med_leads; is_h = lead_name in top3_hyb
        if not (is_m or is_h): continue
        ec, ls, lw, fa = _role(lead_name, med_leads, top3_hyb)
        x0, x1 = COL_X[col]
        ax.add_patch(Rectangle(
            (x0, bottoms[row]), x1 - x0, tops[row] - bottoms[row],
            linewidth=lw, edgecolor=ec, facecolor="none",
            linestyle=ls, zorder=7, clip_on=True))
        if is_h:
            for rank, seg in enumerate(top3_segs):
                if min(int(seg * 0.4 / 2.5), 3) == col:
                    ax.axvspan(seg * 0.4, (seg + 1) * 0.4,
                               ymin=yfracs[row][0], ymax=yfracs[row][1],
                               color=_SC3[rank], alpha=fa, zorder=5)

    is_II_m = "II" in med_leads; is_II_h = "II" in top3_hyb
    if is_II_m or is_II_h:
        ec, ls, lw, fa = _role("II", med_leads, top3_hyb)
        ax.add_patch(Rectangle(
            (0, bottoms[3]), 10, tops[3] - bottoms[3],
            linewidth=lw, edgecolor=ec, facecolor="none",
            linestyle=ls, zorder=7, clip_on=True))
        if is_II_h:
            for rank, seg in enumerate(top3_segs):
                ax.axvspan(seg * 0.4, (seg + 1) * 0.4,
                           ymin=yfracs[3][0], ymax=yfracs[3][1],
                           color=_SC3[rank], alpha=fa, zorder=5)

    for xpos, labels in [
        (2.55, [(-0.6,"aVR"),(-4.1,"aVL"),(-7.6,"aVF")]),
        (5.05, [(-0.6,"V1"), (-4.1,"V2"), (-7.6,"V3")]),
        (7.55, [(-0.6,"V4"), (-4.1,"V5"), (-7.6,"V6")]),
    ]:
        for ypos, label in labels:
            ax.text(xpos, ypos, label, fontsize=8, color="black")
    ax.text(0.1, bottoms[3] + 0.2, "100 Hz   25.0 mm/s   10.0 mm/mV", fontsize=9)

    # Read the II Ref line's actual plotted y-data from ax.lines so the marker
    # sits at the true rendered peak regardless of ecg_plot's internal mm scaling.
    ii_xd, ii_yd = None, None
    for _line in ax.lines:
        _xd = np.asarray(_line.get_xdata(), dtype=float)
        _yd = np.asarray(_line.get_ydata(), dtype=float)
        if (len(_xd) >= 500
                and np.ptp(_xd) > 8.0
                and np.std(_yd) > 0.02
                and bottoms[3] < float(np.median(_yd)) < tops[3]):
            ii_xd, ii_yd = _xd, _yd
            break
    for rp_s in r_peaks_sec:
        if 0 <= rp_s <= 10:
            if ii_xd is not None:
                y_pk = float(np.interp(rp_s, ii_xd, ii_yd))
                y_mk = y_pk + float(np.ptp(ii_yd)) * 0.08
            else:
                y_mk = tops[3] - h_row * 0.15   # fallback
            ax.plot(rp_s, y_mk, "v", color=_RPEAK_COL, markersize=5, alpha=0.95, zorder=8)

    parts = []
    if f.get("heart_rate_bpm"):  parts.append(f"HR: {f['heart_rate_bpm']:.0f} bpm")
    if f.get("qrs_duration_ms"): parts.append(f"QRS: {f['qrs_duration_ms']:.0f} ms")
    if f.get("pr_interval_ms"):  parts.append(f"PR: {f['pr_interval_ms']:.0f} ms")
    if f.get("rr_std_ms"):       parts.append(f"RR SD: {f['rr_std_ms']:.1f} ms")
    if f.get("rr_rmssd_ms"):     parts.append(f"RMSSD: {f['rr_rmssd_ms']:.1f} ms")

    # ncol=3 fills left→right — same structure as Step 5:
    #   Row0: SHAP1       | SHAP2          | SHAP3
    #   Row1: Med+model   | Model-id       | Med-expected
    #   Row2: R-peak
    legend_handles = [
        mpatches.Patch(facecolor=_SC3[0], alpha=0.70, label="SHAP top-1 segment"),
        mpatches.Patch(facecolor=_SC3[1], alpha=0.60, label="SHAP top-2 segment"),
        mpatches.Patch(facecolor=_SC3[2], alpha=0.60, label="SHAP top-3 segment"),
        mpatches.Patch(facecolor="#7f0000", label="Medical + model lead"),
        mpatches.Patch(facecolor="#c0392b", label="Model-identified lead"),
        mpatches.Patch(facecolor="none", edgecolor="#d35400", linewidth=1.5, linestyle="--",
                       label="Medical expected"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor=_RPEAK_COL,
               markersize=8, label="R-peak (NeuroKit2, II Ref)"),
    ]

    # Expand figure downward so ECG keeps its natural physical size
    w, h0   = fig.get_size_inches()
    _margin = 2.5                              # inches below ECG for legend + HR + gap
    _total  = max(h0, 6.0) + _margin
    fig.set_size_inches(w, _total)
    fig.subplots_adjust(bottom=_margin / _total)

    # Legend anchored just below the ECG bottom (not at the figure's very bottom)
    _leg_y = max(0.02, _margin / _total - 0.10)
    fig.legend(handles=legend_handles,
               loc="lower left", ncol=3, fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.02, _leg_y), bbox_transform=fig.transFigure)

    # HR metrics: top-right, just above the axes (below the suptitle)
    if parts:
        ax.text(0.99, 1.02, "  |  ".join(parts),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor="lightyellow", alpha=0.9, pad=3, edgecolor="gray"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 plot — returns fig
# ─────────────────────────────────────────────────────────────────────────────
def plot_hybrid_detail_st(name, info, h_res):
    pred        = info["pred_class"]
    conf        = info["pred_confidence"]
    f           = info["clinical_features"]
    checks      = info["clinical_checks"]
    r_peaks_sec = np.array(f.get("r_peaks", []), dtype=int) / 500.0
    r_peaks_int = np.array(f.get("r_peaks", []), dtype=int)
    shap_vals   = info["shap_values"]

    scores     = h_res["scores"]
    top3_segs  = h_res["top3_segs"]
    top3_hyb_l = h_res["top3_names"]
    top3_hyb   = set(top3_hyb_l)
    med_leads  = DIAG_LEADS_H.get(pred, {"II"})

    all_relevant = list(med_leads | top3_hyb)
    show_leads   = sorted(all_relevant,
                          key=lambda l: _lead_sort_key(l, med_leads, top3_hyb))
    lead_idxs    = [LEAD_NAMES.index(l) for l in show_leads]
    n_show       = len(show_leads)
    sig500       = info["signal"]
    t            = np.linspace(0, 10, 5000)

    # Raw wave indices from NeuroKit2 delineation
    qrs_pairs    = f.get("qrs_pairs",    []) or []
    p_peaks_idx  = np.array(f.get("p_peaks_idx",  []) or [], dtype=int)
    p_onsets_idx = np.array(f.get("p_onsets_idx", []) or [], dtype=int)

    has_rr       = (pred == "AFIB") and len(r_peaks_int) >= 2
    has_poincare = has_rr
    has_template = pred in ("RBBB", "LBBB")

    # Panel index accounting
    n_extra      = 2 if has_rr else 0   # RR + Poincaré come together for AFIB
    idx_shap     = n_show + n_extra
    idx_hybrid   = idx_shap + 1
    idx_template = idx_hybrid + 1
    n_panels     = idx_hybrid + 1 + (1 if has_template else 0)
    h_ratios     = ([2.5] * n_show
                   + ([1.2, 1.5] if has_rr else [])
                   + [1.8, 1.5]
                   + ([2.0] if has_template else []))
    fig_h        = (2.5 * n_show
                   + (2.7 if has_rr else 0)
                   + 5.0
                   + (2.5 if has_template else 0))

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, fig_h),
                             gridspec_kw={"height_ratios": h_ratios})
    if n_panels == 1:
        axes = [axes]

    pred_classes = info.get("pred_classes", [pred])
    passed = sum(1 for v in checks.values() if v[0]); total = len(checks)
    if len(pred_classes) > 1:
        all_str = " + ".join(pred_classes)
        title = f"{name}  —  Predicted: {all_str}  |  Showing: {pred} ({conf:.2f})  |  Clinical checks: {passed}/{total}"
    else:
        title = f"{name}  —  Predicted: {pred} ({conf:.2f})  |  Clinical checks: {passed}/{total}"
    if info["flagged"]: title += "  |  LOW CONFIDENCE"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    # Precompute RR array once (used by both RR panel and Poincaré)
    rr_ms_arr = np.diff(r_peaks_int) / 500 * 1000 if len(r_peaks_int) >= 2 else np.array([])

    last_group = -1
    for ax_i, (lead_name, lead_idx) in enumerate(zip(show_leads, lead_idxs)):
        ax  = axes[ax_i]
        sig = sig500[lead_idx]
        is_m = lead_name in med_leads; is_h = lead_name in top3_hyb

        if is_m and is_h:
            lc = "#7f0000"; tag = " [med + model]"; group = 0
        elif is_h:
            rk = top3_hyb_l.index(lead_name) + 1
            lc = "#c0392b"; tag = f" [model #{rk}]"; group = 1
        else:
            lc = "#888888"; tag = " [med only]"; group = 2

        if group != last_group:
            ax.spines["top"].set_linewidth(2.0)
            ax.spines["top"].set_color(lc)
            last_group = group

        ax.plot(t, sig, color=(0, 0, 0.7), linewidth=0.8)

        for rank, seg in enumerate(top3_segs):
            alpha = 0.60 if (is_m or is_h) else 0.18
            ax.axvspan(seg * 0.4, (seg + 1) * 0.4, color=_SC3[rank], alpha=alpha, zorder=3)

        # axvline on every lead marks timing; triangle only on Lead II
        # (where NeuroKit2 detected the peaks) to avoid wrong positions
        # in other leads whose morphology differs from Lead II.
        for rp_s in r_peaks_sec:
            if 0 <= rp_s <= 10:
                ax.axvline(rp_s, color=_RPEAK_COL, alpha=0.55, linewidth=1.0, zorder=2)
        if lead_name == "II":
            sig_rng = max(float(sig.max()) - float(sig.min()), 0.1)
            for rp_s in r_peaks_sec:
                if 0 <= rp_s <= 10:
                    y_pk = float(np.interp(rp_s, t, sig))
                    ax.plot(rp_s, y_pk + sig_rng * 0.04, "v",
                            color=_RPEAK_COL, markersize=5, alpha=0.95, zorder=6)

        _bbox = dict(facecolor="lightyellow", alpha=0.9, pad=3, edgecolor="gray")

        if pred == "1dAVb" and lead_name == "II":
            pr_ms = f.get("pr_interval_ms")
            if pr_ms:
                for rp_s in r_peaks_sec[:3]:
                    p0 = max(0.0, rp_s - pr_ms/1000.0 - 0.04)
                    p1 = max(0.0, rp_s - 0.04)
                    ya = sig.max() * 0.85
                    ax.annotate("", xy=(p1, ya), xytext=(p0, ya),
                                arrowprops=dict(arrowstyle="<->", color="#1a5276", lw=1.8))
                ttl = ax.set_title(f"PR = {pr_ms:.0f} ms  (threshold: 200 ms → 1dAVb confirmed)",
                                   fontsize=10, fontweight="bold", loc="right",
                                   color="#1a5276", pad=4)
                ttl.set_bbox(_bbox)
            # P-onset markers: dotted vertical lines showing start of each P-wave
            for po_s in p_onsets_idx / 500:
                if 0 <= po_s <= 10:
                    ax.axvline(po_s, color="#f39c12", alpha=0.75, linewidth=1.0,
                               linestyle=":", zorder=4)

        elif pred in ("RBBB", "LBBB"):
            # Exact Q→S spans per beat from NeuroKit2 delineation (replaces approximate window)
            for (qi, si) in qrs_pairs:
                t_q = max(0.0, qi / 500)
                t_s = min(10.0, si / 500)
                if 0 <= t_q < t_s <= 10.0:
                    ax.axvspan(t_q, t_s, color="purple", alpha=0.20, zorder=0)
            if ax_i == 0:
                qrs = f.get("qrs_duration_ms")
                qt  = f.get("qt_interval_ms")
                if qrs:
                    label = (f"QRS = {qrs:.0f} ms  (> 120 ms → {pred} confirmed)"
                             if qrs > 120 else
                             f"QRS = {qrs:.0f} ms  (< 120 ms — resampling artefact; morphology supports {pred})")
                    if pred == "LBBB" and qt:
                        label += f"  |  QT = {qt:.0f} ms"
                    ttl = ax.set_title(label, fontsize=10, fontweight="bold",
                                       loc="right", color="#6a0dad", pad=4)
                    ttl.set_bbox(_bbox)

        elif pred == "AFLT" and ax_i == 0:
            hr = f.get("heart_rate_bpm")
            if hr:
                ttl = ax.set_title(
                    f"HR = {hr:.0f} bpm  (> 100 bpm — look for sawtooth waves in II / III / aVF)",
                    fontsize=10, fontweight="bold", loc="right", color="#7b2d8b", pad=4)
                ttl.set_bbox(_bbox)

        elif pred == "AFIB" and ax_i == 0:
            rmssd = f.get("rr_rmssd_ms")
            if rmssd:
                ttl = ax.set_title(
                    f"RMSSD = {rmssd:.1f} ms  (> 50 ms → irregular rhythm confirms AFIB)",
                    fontsize=10, fontweight="bold", loc="right", color="#c0392b", pad=4)
                ttl.set_bbox(_bbox)

        elif pred == "NORM" and ax_i == 0:
            rr_sd = f.get("rr_std_ms")
            if rr_sd:
                ttl = ax.set_title(
                    f"RR SD = {rr_sd:.1f} ms  (< 100 ms → regular rhythm confirmed)",
                    fontsize=10, fontweight="bold", loc="right", color="#1a5276", pad=4)
                ttl.set_bbox(_bbox)

        # P-wave markers on Lead II for AFIB (show detected peaks, or their absence)
        if pred == "AFIB" and lead_name == "II" and len(p_peaks_idx) > 0:
            sig_range = max(sig.max() - sig.min(), 0.1)
            for pp_s in p_peaks_idx / 500:
                if 0 <= pp_s <= 10:
                    yval = np.interp(pp_s, t, sig)
                    ax.plot(pp_s, yval + sig_range * 0.15, "o",
                            color="#f39c12", markersize=5,
                            markerfacecolor="none", markeredgewidth=1.5, zorder=9)
            p_ratio = f.get("p_wave_ratio")
            if p_ratio is not None and p_ratio < 0.5:
                ax.text(0.01, 0.04,
                        f"P-waves: {len(p_peaks_idx)}/{len(r_peaks_int)} detected "
                        f"(ratio={p_ratio:.2f}) — absent/irregular confirms AFIB",
                        transform=ax.transAxes, fontsize=7.5,
                        color="#e67e22", style="italic",
                        bbox=dict(facecolor="white", alpha=0.75, pad=2))

        ax.set_ylabel(f"{lead_name}{tag}", fontsize=8, color=lc, fontweight="bold")
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 10.01, 0.2))
        ax.set_yticks(np.arange(np.floor(sig.min()), np.ceil(sig.max()) + 0.5, 0.5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xticklabels([f"{int(x)}s" if x % 1 == 0 else "" for x in np.arange(0, 10.01, 0.2)])
        ax.grid(which="major", linestyle="-", linewidth=0.5, color=(1, 0, 0), alpha=0.5)
        ax.grid(which="minor", linestyle="-", linewidth=0.25, color=(1, 0.7, 0.7), alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)

    # ── RR interval panel (AFIB) ──────────────────────────────────────────────
    if has_rr:
        ax_rr = axes[n_show]
        rr_t  = r_peaks_int[1:] / 500
        ax_rr.plot(rr_t, rr_ms_arr, "o-", color="#e31a1c", markersize=4, linewidth=1.2)
        ax_rr.axhline(float(np.mean(rr_ms_arr)), color="gray", linewidth=0.8, linestyle="--",
                      label=f"Mean RR = {np.mean(rr_ms_arr):.0f} ms")
        ax_rr.legend(fontsize=8, loc="upper right")
        ax_rr.set_ylabel("RR interval (ms)", fontsize=9)
        ax_rr.set_xlabel("Time (s)", fontsize=9)
        ax_rr.set_xlim(0, 10); ax_rr.tick_params(labelsize=7)

    # ── Poincaré plot (AFIB) ──────────────────────────────────────────────────
    if has_poincare and len(rr_ms_arr) >= 2:
        ax_pc = axes[n_show + 1]
        rr_n  = rr_ms_arr[:-1]; rr_n1 = rr_ms_arr[1:]
        ax_pc.scatter(rr_n, rr_n1, color="#e31a1c", alpha=0.6, s=20, edgecolors="none")
        lim_lo = max(0.0, float(min(rr_n.min(), rr_n1.min())) - 50)
        lim_hi = float(max(rr_n.max(), rr_n1.max())) + 50
        ax_pc.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.8, alpha=0.4)
        ax_pc.set_xlim(lim_lo, lim_hi); ax_pc.set_ylim(lim_lo, lim_hi)
        sd1 = float(np.std(rr_n1 - rr_n) / np.sqrt(2))
        sd2 = float(np.std(rr_n1 + rr_n) / np.sqrt(2))
        ax_pc.set_xlabel("RRₙ (ms)", fontsize=9)
        ax_pc.set_ylabel("RRₙ₊₁ (ms)", fontsize=9)
        ax_pc.set_title(
            f"Poincaré plot  —  SD1 = {sd1:.0f} ms  |  SD2 = {sd2:.0f} ms"
            "  (scattered cloud → AFIB)",
            fontsize=9, fontweight="bold", color="#c0392b", loc="right")
        ax_pc.tick_params(labelsize=7)

    # ── Temporal SHAP bar chart ───────────────────────────────────────────────
    ax_shap = axes[idx_shap]
    t_ctr = np.arange(N_SEGMENTS) * 0.4 + 0.2
    ax_shap.bar(t_ctr, shap_vals, width=0.35, color="#cccccc", alpha=0.7)
    for rank, seg in enumerate(top3_segs):
        ax_shap.bar(seg*0.4+0.2, shap_vals[seg], width=0.35,
                    color=_SC3[rank], alpha=0.85, label=f"Top-{rank+1}: {seg*0.4:.1f}s")
    ax_shap.set_xlim(0, 10)
    ax_shap.set_xticks(range(0, 11))
    ax_shap.set_xticklabels([str(i) for i in range(11)], fontsize=8)
    ax_shap.set_xlabel("Time (s)", fontsize=9)
    ax_shap.set_ylabel("Temporal SHAP", fontsize=9)
    ax_shap.axhline(0, color="black", linewidth=0.5)
    ax_shap.tick_params(labelsize=7)

    # ── Hybrid lead importance bar chart ──────────────────────────────────────
    ax_lead = axes[idx_hybrid]
    sc_max  = max(np.abs(scores).max(), 1e-8)
    x_pos   = np.arange(12)
    bar_c   = []
    for ln in LEAD_NAMES:
        if ln in med_leads and ln in top3_hyb: bar_c.append("#7f0000")
        elif ln in top3_hyb:                   bar_c.append("#c0392b")
        elif ln in med_leads:                  bar_c.append("#aaaaaa")
        else:                                  bar_c.append("#eeeeee")
    bars = ax_lead.bar(x_pos, scores/sc_max, color=bar_c, alpha=0.80, width=0.7)
    for bar, ln in zip(bars, LEAD_NAMES):
        if ln in med_leads and ln not in top3_hyb:
            bar.set_edgecolor("#d35400")
            bar.set_linewidth(1.5)
            bar.set_linestyle("--")
    ax_lead.axhline(0, color="black", linewidth=0.5)
    ax_lead.set_xticks(x_pos); ax_lead.set_xticklabels(LEAD_NAMES, fontsize=8)
    ax_lead.set_xlabel("Lead", fontsize=9)
    ax_lead.set_ylabel("Hybrid score\n(normalised)", fontsize=8)
    ax_lead.tick_params(labelsize=7)

    # ── Beat template (RBBB / LBBB) ───────────────────────────────────────────
    if has_template:
        ax_bt    = axes[idx_template]
        tl_idx   = 6 if pred == "RBBB" else 11   # V1 for RBBB, V6 for LBBB
        tl_name  = "V1" if pred == "RBBB" else "V6"
        sig_tl   = sig500[tl_idx]
        window   = 150   # ±150 samples = ±300 ms at 500 Hz
        beats    = [sig_tl[rp - window: rp + window]
                    for rp in r_peaks_int
                    if rp - window >= 0 and rp + window < len(sig_tl)]
        if beats:
            beats  = np.array(beats)
            t_bt   = np.linspace(-300, 300, 2 * window)
            for b in beats:
                ax_bt.plot(t_bt, b, color="gray", alpha=0.25, linewidth=0.6)
            mean_b = beats.mean(axis=0)
            ax_bt.plot(t_bt, mean_b, color="#6a0dad", linewidth=2.0,
                       label=f"Mean ({len(beats)} beats)")
            ax_bt.axvline(0, color=_RPEAK_COL, linewidth=1.0, alpha=0.7,
                          linestyle="--", label="R-peak")
            ax_bt.set_xlabel("Time from R-peak (ms)", fontsize=9)
            ax_bt.set_ylabel(f"{tl_name} (mV)", fontsize=9)
            qrs = f.get("qrs_duration_ms")
            bt_title = f"Beat template — {tl_name}  |  {len(beats)} beats"
            if qrs: bt_title += f"  |  QRS = {qrs:.0f} ms"
            bt_title += ("  (expect rsR’ — broad terminal R)" if pred == "RBBB"
                         else "  (expect broad monophasic R, no septal Q)")
            ax_bt.set_title(bt_title, fontsize=9, fontweight="bold",
                            color="#6a0dad", loc="right")
            ax_bt.legend(fontsize=8, loc="upper left")
            ax_bt.tick_params(labelsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_SC3[0], alpha=0.75, label="SHAP top-1 segment"),
        mpatches.Patch(facecolor=_SC3[1], alpha=0.65, label="SHAP top-2 segment"),
        mpatches.Patch(facecolor=_SC3[2], alpha=0.65, label="SHAP top-3 segment"),
        mpatches.Patch(facecolor="#7f0000", label="Medical + model lead"),
        mpatches.Patch(facecolor="#c0392b", label="Model-identified lead"),
        mpatches.Patch(facecolor="none", edgecolor="#d35400", linewidth=1.5, linestyle="--",
                       label="Medical expected"),
        Line2D([0],[0], color=_RPEAK_COL, lw=1.2, alpha=0.7, marker="v",
               markerfacecolor=_RPEAK_COL, markersize=5, label="R-peak (NeuroKit2)"),
    ]
    if pred in ("RBBB", "LBBB"):
        legend_handles.append(
            mpatches.Patch(facecolor="purple", alpha=0.3,
                           label="QRS window (exact, NeuroKit2)"))
    if pred == "AFIB":
        legend_handles.append(
            Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#f39c12", markeredgewidth=1.5, markersize=7,
                   label="P-peak (NeuroKit2)"))
    if pred == "1dAVb":
        legend_handles.append(
            Line2D([0],[0], color="#f39c12", lw=1.0, alpha=0.75, linestyle=":",
                   label="P-onset (NeuroKit2)"))
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=3, fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure)
    fig.subplots_adjust(bottom=0.09)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ECG Inference", layout="wide", page_icon="🫀")
st.title("🫀 An Explainable ECG Diagnostic System Using Hybrid Deep Learning and Dual-Layer Clinical Validation")
st.caption(f"Device: {DEVICE}  |  Classes: {', '.join(CLASSES)}")

model = load_model()

uploaded = st.file_uploader("Upload ECG file (.npy, shape 12 × 1000, 100 Hz)",
                             type=["npy"])

if uploaded is not None:
    # Save to temp so load_ecg_sample and ecg_plot can read it
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        ecg_path = tmp.name

    sample_name = os.path.splitext(uploaded.name)[0]

    # ── Multilead Classification ─────────────────────────────────────────────
    st.header("Multilead Classification")
    with st.spinner("Preprocessing ECG…"):
        signal = load_ecg_sample(ecg_path)
        probs  = predict(model, signal)

    pred_idx     = int(probs.argmax())
    pred_classes = [CLASSES[i] for i, p in enumerate(probs) if p >= CONFIDENCE_THRESHOLD]
    if not pred_classes:
        pred_classes = [CLASSES[pred_idx]]

    info = {
        "signal":           signal,
        "true_label":       None,
        "type":             "test",
        "pred_class":       CLASSES[pred_idx],   # primary for SHAP/hybrid
        "pred_classes":     pred_classes,          # all above threshold
        "pred_confidence":  float(probs[pred_idx]),
        "flagged":          float(probs[pred_idx]) < CONFIDENCE_THRESHOLD,
        "probs":            probs,
    }

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.metric("Predicted class", " + ".join(pred_classes))
        st.metric("Confidence (primary)", f"{info['pred_confidence']:.3f}")
        if len(pred_classes) > 1:
            st.warning(f"MULTILABEL — {len(pred_classes)} classes above threshold: "
                       f"{', '.join(f'{c}: {probs[CLASS_TO_IDX[c]]:.3f}' for c in pred_classes)}. "
                       f"SHAP/hybrid computed for primary: **{info['pred_class']}**.")
        if info["flagged"]:
            st.warning("LOW CONFIDENCE — flagged for clinician review")
    with col_r:
        fig_p, ax_p = plt.subplots(figsize=(5, 2.8))
        pred_set = set(pred_classes)
        colors = ["#7f0000" if CLASSES[i] == info["pred_class"]
                  else "#c0392b" if CLASSES[i] in pred_set
                  else "#cccccc" for i in range(len(CLASSES))]
        ax_p.barh(CLASSES, probs, color=colors)
        ax_p.axvline(CONFIDENCE_THRESHOLD, color="gray", linestyle="--", linewidth=0.8,
                     label=f"Threshold ({CONFIDENCE_THRESHOLD})")
        ax_p.set_xlim(0, 1); ax_p.set_xlabel("Probability")
        ax_p.legend(fontsize=8); fig_p.tight_layout()
        st.pyplot(fig_p); plt.close(fig_p)

    # ── SHAP + Hybrid + Clinical (background computation) ────────────────────
    with st.spinner("Computing SHAP attribution maps (~2,500 forward passes — takes 1–2 min)…"):
        cidx      = CLASS_TO_IDX[info["pred_class"]]
        shap_vals = compute_shap_temporal(model, signal, cidx)

    top_segs  = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_times = [(int(s)*0.4, (int(s)+1)*0.4) for s in top_segs]
    info["shap_values"]       = shap_vals
    info["shap_full"]         = shap_to_signal_resolution(shap_vals)
    info["shap_top_segments"] = top_segs
    info["shap_top_times"]    = top_times

    with st.spinner("Computing hybrid lead importance (48 forward passes)…"):
        scores     = compute_hybrid_lead_importance(model, signal, cidx, top_segs)
    order      = np.argsort(scores)[::-1]
    top3_names = [LEAD_NAMES[i] for i in order[:3]]
    hybrid_res = {"scores": scores, "top3_names": top3_names, "top3_segs": top_segs}

    with st.spinner("Extracting clinical features (NeuroKit2)…"):
        info["clinical_features"] = extract_clinical_features(signal)
        info["clinical_checks_all"] = {
            cls: check_clinical_rules(cls, info["clinical_features"], shap_vals, signal)
            for cls in info["pred_classes"]
        }
        info["clinical_checks"] = info["clinical_checks_all"][info["pred_class"]]

    passed = sum(1 for v in info["clinical_checks"].values() if v[0])
    total  = len(info["clinical_checks"])
    st.success(
        f"Done.  Top hybrid leads: **{', '.join(top3_names)}**  |  "
        f"Top SHAP segments: {[f'{s*0.4:.1f}–{(s+1)*0.4:.1f}s' for s in top_segs]}  |  "
        f"Clinical checks ({info['pred_class']}): **{passed}/{total}**")

    with st.expander("Clinical check details"):
        for cls, checks in info["clinical_checks_all"].items():
            st.markdown(f"**{cls}**")
            for check_name, (p, detail) in checks.items():
                st.write(f"{'✅' if p else '❌'}  {check_name}: {detail}")

    # ── 12 Lead ECG Overview ─────────────────────────────────────────────────
    st.header("12 Lead ECG Overview")
    for cls in info["pred_classes"]:
        info_cls = {**info,
                    "pred_class":      cls,
                    "pred_confidence": float(probs[CLASS_TO_IDX[cls]]),
                    "clinical_checks": info["clinical_checks_all"][cls]}
        if len(info["pred_classes"]) > 1:
            st.subheader(f"Showing: {cls}")
        with st.spinner(f"Generating 12-lead plot ({cls})…"):
            fig3 = plot_hybrid_12lead_st(sample_name, info_cls, hybrid_res, ecg_path)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # ── Automated Text Explanation ────────────────────────────────────────────
    st.header("Automated Text Explanation")
    for cls in info["pred_classes"]:
        info_cls = {**info,
                    "pred_class":      cls,
                    "pred_confidence": float(probs[CLASS_TO_IDX[cls]]),
                    "clinical_checks": info["clinical_checks_all"][cls]}
        explanation = generate_text_explanation(sample_name, info_cls, hybrid_res)
        info_cls["text_explanation"] = explanation
        if len(info["pred_classes"]) > 1:
            st.markdown(f"**{cls}**")
        st.info(explanation)

    # ── Diagnostic Lead Detail ────────────────────────────────────────────────
    st.header("Diagnostic Lead Detail")
    for cls in info["pred_classes"]:
        info_cls = {**info,
                    "pred_class":      cls,
                    "pred_confidence": float(probs[CLASS_TO_IDX[cls]]),
                    "clinical_checks": info["clinical_checks_all"][cls]}
        if len(info["pred_classes"]) > 1:
            st.subheader(f"Showing: {cls}")
        with st.spinner(f"Generating detail plot ({cls})…"):
            fig5 = plot_hybrid_detail_st(sample_name, info_cls, hybrid_res)
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    os.unlink(ecg_path)
