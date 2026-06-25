# Advanced AI Research

Three research studies completed during my **Master of Computing (Artificial Intelligence major)**.
Each one pairs a focused literature review with hands-on benchmarking of state-of-the-art
models, and selects the best architecture per task — evaluating not only accuracy, but also
**robustness, computational sustainability, and AI ethics**.

| # | Study | Domain | Key result |
|---|-------|--------|-----------|
| 01 | [Semantic Segmentation](01-semantic-segmentation/) | Computer Vision | Fine-tuning **SegFormer-B0** recovers the Cityscapes→SydneyScapes domain gap to **57.3% mIoU** (vs. 48.98% in the literature), staying lightweight |
| 02 | [LLM Suicide-Risk Detection](02-llm-suicide-risk-detection/) | NLP / LLMs | Benchmarked **LLMs/SLMs** for clinical risk (C-SSRS); IBM Granite 3.2-8B strongest, proposing a **privacy-first hybrid** with a 100% high-risk-recall mandate |
| 03 | [ECG Classification Benchmark](03-ecg-benchmarking/) | Time-Series Deep Learning | **7 models** for 7-class ECG diagnosis; best (CNN-BiLSTM-Transformer) reached **Macro F1 0.748 / AUROC 0.940** with SHAP explainability *(team project)* |

Each folder contains the **code** and the **final research report (PDF)**.

## A note on data and reproducibility

Raw datasets are **intentionally not included** — some are licensed or clinically sensitive
(e.g. the suicide-risk corpora). For the same reason, notebooks are provided with **cell
outputs cleared**. Each study's README documents its data sources so the work can be
reproduced from the original datasets.

---

*Studies completed for COMP6011 — Advanced AI Research Topics, Curtin University (2026).*
