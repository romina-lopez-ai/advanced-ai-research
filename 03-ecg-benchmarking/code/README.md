# ECG Classification Benchmark

Training and evaluation of 7 deep learning models for 7-class multilabel ECG classification
across PTB-XL, CPSC 2018, and Georgia datasets.

---

## Models

| Model | Script | Params | Input |
|-------|--------|--------|-------|
| BiLSTM1D | `experiments/bilstm1d_experiment.py` | ~0.7M | 100 Hz (1000 samples) |
| XResNet1d50 | `experiments/xresnet1d.py` | ~18.5M | 500 Hz (5000 samples) |
| InceptionTime | `experiments/inception1d.py` | ~0.5M | 500 Hz (5000 samples) |
| ResNet18-2D | `experiments/resnet18_2d_experiment.py` | ~11.2M | ECG spectrogram (ImageNet pretrained) |
| CNN-BiLSTM-Transformer | `experiments/cnn_bilstm_transformer_experiment.py` | ~1.4M | 500 Hz (5000 samples) |
| PatchECG (1D ViT) | `experiments/vit_ecg_experiment.py` | ~1.2M | 500 Hz (5000 samples) |
| HuBERT-ECG-S | `experiments/hubert_ecg_finetune_experiment.py` | ~21M | 100 Hz (1000 samples, fine-tune) |

All models trained from scratch on PTB-XL (folds 1–8), validated on fold 9, tested on fold 10.
HuBERT-ECG-S fine-tunes the pretrained `Edoardo-BS/hubert-ecg-small` backbone.

---

## Task Definition

- **Classes (7):** `NORM | AFIB | AFLT | 1dAVb | RBBB | LBBB | OTHERS`
- **Formulation:** multilabel — a single record can carry multiple labels simultaneously
- **PTB-XL label threshold:** 100 (maximum clinical certainty only)
- **Loss:** BCEWithLogitsLoss with per-class `pos_weight` to handle label imbalance
- **Classification threshold:** 0.5 (binary predictions); does not affect AUROC
- **Confidence flagging:** records with no class exceeding 0.60 probability are flagged for manual review

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib \
            wfdb tqdm transformers accelerate
```

Python 3.10+ recommended. GPU strongly advised for XResNet1d50 and HuBERT-ECG-S.

### 2. Download datasets

```bash
python scripts/download_ptb-xl.py    # PTB-XL 500 Hz  (~2 GB)
python scripts/download_cpsc.py      # CPSC 2018       (~1.5 GB)
python scripts/download_georgia.py   # Georgia 2020    (~5 GB)
```

Data lands in `data/raw/{ptbxl,cpsc2018,georgia}/`.

### 3. Train a model

```bash
# From the repo root — runs full train + evaluation pipeline
python code/experiments/xresnet1d.py
python code/experiments/inception1d.py
python code/experiments/bilstm1d_experiment.py
python code/experiments/cnn_bilstm_transformer_experiment.py
python code/experiments/vit_ecg_experiment.py
python code/experiments/resnet18_2d_experiment.py
python code/experiments/hubert_ecg_finetune_experiment.py
```

Each script:
1. Loads and caches datasets (PTB-XL + cross-corpus)
2. Trains with early stopping (patience varies per model)
3. Runs per-class threshold optimisation on the validation fold
4. Evaluates on PTB-XL fold 10, CPSC 2018, and Georgia
5. Saves results JSON and best checkpoint to `code/experiments/results_<model>/`

Set `"skip_training": True` in a script's CONFIG dict to load a saved checkpoint and only run evaluation.

---

## Methodology Notes

- **PTB-XL split:** folds 1–8 train, fold 9 validation, fold 10 test (standard benchmark split)
- **Cross-corpus evaluation:** CPSC 2018 and Georgia are held-out at inference — no fine-tuning on these datasets
- **Signal preprocessing:** bandpass filter (0.5–50 Hz), per-lead z-score normalisation
- **CPSC / Georgia resampling:** signals resampled to match each model's target sampling rate
- **Per-class thresholds:** optimised on PTB-XL fold 9 by maximising per-class F1; applied at test time
- **Metrics:** Macro F1, Macro AUROC, Macro Sensitivity, Macro Specificity, Flag Rate

---

## Report Figures

Figures are generated from `code/report_plots/` using the aggregated results in `results_summary_all.json`:

```bash
python code/report_plots/benchmark_plot.py   # scatter: AUROC vs F1 across datasets
python code/report_plots/benchmark_bar.py    # grouped bar: F1 + AUROC per model
```

Output saved to `code/report_plots/benchmark_all.png` and `benchmark_bar.png` (500 dpi).

---

## Results

Full per-class results and per-dataset summaries are in `code/experiments/results_<model>/`.
Aggregated benchmark numbers used in the report are in `code/report_plots/results_summary_all.json`.
