# IPA-SEC — IoT Protocol Attack Security Classifier

> A modular, multi-output machine learning and deep learning pipeline for classifying IoT network traffic by **device type** and **attack category** simultaneously, using the NIMLAB IoT Dataset 2025.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Jupyter Notebook](#jupyter-notebook)
- [Output Modes](#output-modes)
- [Configuration](#configuration)
- [Output Artefacts](#output-artefacts)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)

---

## Overview

IPA-SEC is designed to solve a **dual classification problem** on IoT network traffic:

| Task | Target Column | Description |
|---|---|---|
| Device Identification | `Label` | Which IoT device generated the traffic? |
| Attack Detection | `Traffic Type` | What type of network attack (or benign traffic) is present? |

The pipeline supports three output modes — predict both targets simultaneously (**multi-output**), or either target individually — across 11 deep learning architectures and 6 classical machine learning algorithms. All models are evaluated with a comprehensive set of metrics and produce confusion matrix PDFs and prediction CSVs automatically.

---

## Project Structure

```
ipasec/
├── config.py        # Central configuration: paths, hyperparameters, model toggles
├── data_utils.py    # Data loading, sampling, preprocessing, train/test split
├── models_dl.py     # All deep learning model builders + ESN custom layer
├── models_ml.py     # All ML estimator definitions and registry
├── trainer.py       # Unified train–evaluate–save loop (DL and ML)
├── evaluator.py     # Metrics, classification report, confusion matrix, CSV export
└── main.py          # Orchestration entry point with CLI
```

Each module has a single responsibility. To add a new model, you only touch `models_dl.py` or `models_ml.py` and add one entry to `config.MODELS_ENABLED` — no other file changes needed.

---

## Supported Models

### Deep Learning (11 architectures)

All DL builders share an identical signature:
```python
build_<name>(input_shape, num_classes_device, num_classes_attack=None)
```
Setting `num_classes_attack=None` produces a **single-output** model; passing an integer produces a **multi-output** model with two softmax heads (`device`, `attack`). A shared `_compile_and_return()` helper handles compilation for both cases.

| Model | Key Architecture |
|---|---|
| **1D-CNN** | Conv(64) → Pool → Conv(128) → Pool → Flatten → Dense(128) → Dropout |
| **Autoencoder** | Encoder (128→64→32 latent) + Decoder reconstruction branch + classification head. Reconstruction is an auxiliary loss (weight 0.1) to regularise the latent space. |
| **LSTM** | Stacked 2-layer LSTM (64 units each) with Dropout |
| **BiLSTM** | Stacked 2-layer Bidirectional LSTM (64 units each) with Dropout |
| **GRU** | Stacked 2-layer GRU (64 units each) with Dropout |
| **CNN-GRU** | Two Conv1D blocks followed by a GRU layer |
| **CNN-LSTM** | Two Conv1D blocks followed by an LSTM layer |
| **MLP** | Three Dense layers (256→128→64) with BatchNorm + Dropout |
| **ResNet1D** | Residual blocks with skip connections + GlobalAveragePooling |
| **RNN** | Stacked 2-layer SimpleRNN (64 units each) with Dropout |
| **ESN** | Echo State Network — frozen random reservoir (128 units, spectral radius ≈ 0.9) + trainable read-out layer. `@tf.function` + `tf.while_loop` for graph-mode compatibility. |

### Machine Learning (6 algorithms)

In **multi-output** mode each ML model is automatically wrapped in both a `MultiOutputClassifier` and a `ClassifierChain`, giving two result rows per algorithm. In single-output mode the base estimator is used directly.

| Model | Notes |
|---|---|
| **XGBoost** | Gradient boosted trees (`mlogloss` eval metric) |
| **Random Forest** | Balanced class weights, feature importances reported |
| **Decision Tree** | Balanced class weights, feature importances reported |
| **AdaBoost** | 100 boosted depth-3 decision trees |
| **Extra Trees** | Extremely randomised trees, balanced class weights, feature importances reported |
| **LightGBM** | Histogram-based gradient boosting (skipped gracefully if not installed) |

---

## Dataset

**NIMLAB IoT Dataset 2025** — a network traffic capture dataset covering a range of consumer IoT devices under both normal and attack conditions.

- 80+ raw network-level features (DNS, TCP, UDP, HTTP, TLS, ICMP, packet statistics, …)
- `Label` column — IoT device identity (e.g. *Amazon Alexa Echo Dot*, *LG Smart TV*, *Arlo Q Camera*)
- `Traffic Type` column — traffic category (e.g. *Benign*, *Slowloris*, …)

The pipeline performs **stratified proportional sampling** before training (40% per class for single-output, 70% for multi-output) to keep run times manageable while preserving class balance.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ipasec.git
cd ipasec
```

### 2. Create and activate a virtual environment

```bash
# conda (recommended)
conda create -n ipasec python=3.9
conda activate ipasec

# or venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow scikit-learn xgboost lightgbm \
            pandas numpy psutil tabulate \
            matplotlib seaborn scienceplots
```

> **Python version:** 3.9 or later required. The `X | Y` union type syntax used internally is replaced with `Optional[X]` from `typing` for full 3.9 compatibility.

---

## Usage

### Command Line

```bash
# Multi-output: predict both device type and attack category
python main.py --mode multi

# Single-output: device type only
python main.py --mode device

# Single-output: attack category only
python main.py --mode traffic
```

The default mode is set by `OUTPUT_MODE` in `config.py`.

### Jupyter Notebook

The argparse CLI is bypassed — call `main()` directly:

```python
import sys
sys.path.insert(0, r"path/to/ipasec")   # only if notebook is in a different folder

# Optional: override config before importing
import config
from pathlib import Path
config.CSV_PATH    = Path(r"path/to/NIMLABIoT_processed.csv")
config.OUTPUT_DIR  = Path(r"path/to/output/folder")
config.DL_EPOCHS   = 5                        # faster iteration
config.MODELS_ENABLED["ESN"] = False          # skip slow models while testing

# Run the pipeline
%matplotlib inline
from main import main
main('multi')   # or 'device' / 'traffic'
```

---

## Output Modes

| Mode | Targets | ML wrappers used |
|---|---|---|
| `multi` | Device Type **+** Attack Category | `MultiOutputClassifier` and `ClassifierChain` for each ML model |
| `device` | Device Type only | Base estimator directly |
| `traffic` | Attack Category only | Base estimator directly |

---

## Configuration

All settings live in `config.py`. The most commonly changed values:

```python
# ── Dataset ────────────────────────────────────────────────────────────────
CSV_PATH    = Path(r"path/to/NIMLABIoT_processed.csv")
OUTPUT_DIR  = None          # None → write results next to the CSV

# Optional class filters (None = include everything)
TARGET_LABELS        = None  # e.g. ['Amazon Alexa Echo Spot', 'LG Smart TV']
TARGET_TRAFFIC_TYPES = None  # e.g. ['Benign', 'Slowloris']

# ── Pipeline ────────────────────────────────────────────────────────────────
OUTPUT_MODE  = 'multi'      # 'multi' | 'device' | 'traffic'
TEST_SIZE    = 0.3
RANDOM_SEED  = 42

# ── Deep Learning ───────────────────────────────────────────────────────────
DL_EPOCHS     = 10
DL_BATCH_SIZE = 32
DL_VAL_SPLIT  = 0.2

# ── Enable / disable individual models ─────────────────────────────────────
MODELS_ENABLED = {
    "1D-CNN": True,  "Autoencoder": True, "LSTM": True,
    "BiLSTM": True,  "GRU": True,         "CNN-GRU": True,
    "CNN-LSTM": True,"MLP": True,         "ResNet1D": True,
    "RNN": True,     "ESN": True,
    "XGBoost": True, "Random Forest": True, "Decision Tree": True,
    "AdaBoost": True,"Extra Trees": True,   "LightGBM": True,
}
```

---

## Output Artefacts

All artefacts are written to `OUTPUT_DIR` (defaults to the same directory as the CSV):

| File | Description |
|---|---|
| `metrics_<dataset>_<mode>.csv` | Aggregate results table — one row per model per target |
| `predictions_<dataset>_<model>.csv` | True labels, predicted labels, and confidence scores |
| `classification_report_<target>.csv` | Per-class precision, recall, F1, support, accuracy |
| `confusion_matrix_<target>.csv` | Confusion matrix in tabular form |
| `confusion_matrix_<target>.pdf` | Confusion matrix heatmap (IEEE-style, 300 dpi) |

---

## Evaluation Metrics

Each model-target pair is evaluated with the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | Macro-averaged precision across classes |
| Recall (TPR) | Macro-averaged recall across classes |
| F1 Score | Macro-averaged harmonic mean of precision and recall |
| Cohen's Kappa | Agreement corrected for chance |
| MCC | Matthews Correlation Coefficient — robust to class imbalance |
| TNR | Macro-averaged true negative rate (specificity) |
| FPR | Macro-averaged false positive rate |
| FNR | Macro-averaged false negative rate |
| AUC | Macro-averaged one-vs-rest ROC AUC (where probabilities available) |
| Hamming Loss | Joint multi-label prediction error (multi-output mode only) |
| Train / Test Time | Wall-clock seconds total and per sample |
| CPU / Memory Usage | Mean process CPU % and RAM % during train and inference |

---

## Requirements

```
python        >= 3.9
tensorflow    >= 2.10
scikit-learn  >= 1.2
xgboost       >= 1.7
lightgbm      >= 3.3
pandas        >= 1.5
numpy         >= 1.23
psutil        >= 5.9
tabulate      >= 0.9
matplotlib    >= 3.6
seaborn       >= 0.12
scienceplots  >= 2.0
```

---

## License

This project is released for academic and research use. Please cite the NIMLAB IoT Dataset 2025 if you use this pipeline in published work.
