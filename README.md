# Transformer NLP Assignment — BERT + SHAP + LIME + Attention Analysis

## Overview

Fine-tunes **BERT (bert-base-uncased)** on the [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) dataset for binary sentiment classification, then analyzes the model using:

- **Attention visualization** (layers 1, 6, 12 × multiple heads)
- **SHAP** — PartitionExplainer with text masker (25 samples)
- **LIME** — LimeTextExplainer (25 samples)
- **Comparative analysis** — Faithfulness, Stability, Runtime
- **Error analysis** — Top confident misclassifications with LIME explanations

---

## Project Structure

```
transformer_nlp_project/
├── notebooks/
│   ├── 01_train_evaluate.py       ← Dataset prep, fine-tuning, metrics
│   ├── 02_attention_analysis.py   ← Attention heatmaps (all layer/head combos)
│   ├── 03_shap_explainability.py  ← SHAP explanations (25 samples)
│   ├── 04_lime_explainability.py  ← LIME explanations (25 samples)
│   ├── 05_comparative_analysis.py ← SHAP vs LIME comparison
│   └── 06_error_analysis.py       ← Misclassification inspection
├── results/
│   ├── bert_amazon_polarity/      ← Saved model & tokenizer (created after training)
│   ├── attention_maps/            ← PNG heatmaps per layer/head
│   ├── shap_outputs/              ← SHAP plots + CSV
│   ├── lime_outputs/              ← LIME plots + CSV
│   ├── comparison/                ← SHAP vs LIME charts + CSV
│   └── error_analysis/            ← Error plots + CSV
├── requirements.txt
└── README.md
```

---

## Hardware & Software Environment

| Item | Value |
|------|-------|
| **OS** | Ubuntu 22.04 / Windows 10+ / macOS 13+ |
| **Python** | 3.9–3.11 |
| **GPU** | NVIDIA GPU with CUDA 11.8+ (optional, falls back to CPU) |
| **VRAM** | 8 GB+ recommended for batch_size=16 |
| **RAM** | 16 GB+ recommended |
| **Training time** | ~30 min (GPU) / ~3 h (CPU) for 20,000 samples |

---

## Setup Instructions

### Step 1 — Clone / create the project folder

```bash
git clone https://github.com/YOUR_USERNAME/transformer-nlp-assignment.git
cd transformer-nlp-assignment
```

### Step 2 — Create a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU support (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Dataset preparation

The dataset is automatically downloaded from HuggingFace Hub on first run. No manual download needed.

```bash
# Optional: pre-download and cache
python -c "from datasets import load_dataset; load_dataset('fancyzhx/amazon_polarity')"
```

---

## Running the Experiments

> Run scripts **in order** — each script depends on the previous one's output.

### 1. Train and Evaluate

```bash
python notebooks/01_train_evaluate.py
```

**Outputs:**
- `results/bert_amazon_polarity/` — saved model & tokenizer
- `results/classification_report.txt` — Accuracy, Precision, Recall, F1
- `results/confusion_matrix.png`

### 2. Attention Analysis

```bash
python notebooks/02_attention_analysis.py
```

**Outputs:**
- `results/attention_maps/` — PNG heatmaps (layer × head × sample)
- `results/attention_maps/entropy_by_layer.png`

### 3. SHAP Explanations

```bash
python notebooks/03_shap_explainability.py
```

**Outputs:**
- `results/shap_outputs/shap_sample_XX.png` (25 plots)
- `results/shap_outputs/shap_summary.csv`

### 4. LIME Explanations

```bash
python notebooks/04_lime_explainability.py
```

**Outputs:**
- `results/lime_outputs/lime_sample_XX.png` (25 plots)
- `results/lime_outputs/lime_summary.csv`

### 5. SHAP vs LIME Comparison

```bash
python notebooks/05_comparative_analysis.py
```

**Outputs:**
- `results/comparison/shap_lime_comparison.png`
- `results/comparison/shap_lime_comparison.csv`

### 6. Error Analysis

```bash
python notebooks/06_error_analysis.py
```

**Outputs:**
- `results/error_analysis/error_XX.png` (top 10 confident errors)
- `results/error_analysis/error_overview.png`
- `results/error_analysis/error_records.csv`

---

## Reproducing Results

To fully reproduce all results from scratch:

```bash
# One-shot (sequential)
python notebooks/01_train_evaluate.py && \
python notebooks/02_attention_analysis.py && \
python notebooks/03_shap_explainability.py && \
python notebooks/04_lime_explainability.py && \
python notebooks/05_comparative_analysis.py && \
python notebooks/06_error_analysis.py
```

All random seeds are fixed at `SEED = 42` throughout every script.

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Accuracy | ~94% (on 2,000 test samples) |
| Precision | ~94% |
| Recall | ~94% |
| F1-Score | ~94% |
| Training epochs | 3 |
| Subset size | 20,000 train / 2,000 val / 2,000 test |

---

## References

1. Dataset: https://huggingface.co/datasets/fancyzhx/amazon_polarity
2. SHAP: https://shap.readthedocs.io/en/latest/text_examples.html
3. LIME: https://lime-ml.readthedocs.io/en/latest/
4. BERT: Devlin et al. (2019) — https://arxiv.org/abs/1810.04805
5. HuggingFace Transformers: https://huggingface.co/docs/transformers
