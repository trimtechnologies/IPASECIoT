"""
config.py
=========
Central configuration for the IPA-SEC pipeline.
Edit this file to change dataset paths, hyperparameters, and which models to run.
"""

from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
CSV_PATH = Path(
    r"F:\Datasets_Project\NIM LAB IoT Dataset 2025\slowloriss\small\processed\NIMLABIoT_processed.csv"
    #r"D:\HybridDualDistillation\NIM LAB IoT2025\federated_kd_updated\data\NIMSLABIoT_multi.csv"
)

# Optional label / traffic-type filters (set to None to include everything)
TARGET_LABELS = None          # e.g. ['Amazon Alexa Echo Spot', 'LG Smart TV']
TARGET_TRAFFIC_TYPES = None   # e.g. ['Benign', 'Slowloris']

# Fraction to sample per class (multi-output uses 0.7 internally)
SAMPLE_FRAC_SINGLE = 0.4
SAMPLE_FRAC_MULTI  = 0.8

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# 'multi'   → predict both Device Type (Label) and Attack Category (Traffic Type)
# 'device'  → predict Device Type only
# 'traffic' → predict Attack Category only
OUTPUT_MODE = 'multi'

TEST_SIZE   = 0.3
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Deep Learning shared hyper-parameters
# ---------------------------------------------------------------------------
DL_EPOCHS     = 20
DL_BATCH_SIZE = 32
DL_VAL_SPLIT  = 0.2

# ---------------------------------------------------------------------------
# Model selection — set False to skip a model entirely
# ---------------------------------------------------------------------------
MODELS_ENABLED = {
    # ── Deep Learning ──────────────────────────────────────────────────────
    "1D-CNN":        True,
    "Autoencoder":   True,   # classification head on latent space
    "LSTM":          True,
    "BiLSTM":        True,
    "GRU":           True,
    "CNN-GRU":       True,
    "CNN-LSTM":      True,
    "MLP":           True,
    "ResNet1D":      True,
    "RNN":           True,
    "ESN":           True,   # Echo State Network (reservoir computing)
    # ── Machine Learning ───────────────────────────────────────────────────
    "XGBoost":       True,
    "Random Forest": True,
    "Decision Tree": True,
    "AdaBoost":      True,
    "Extra Trees":   True,
    "LightGBM":      True,
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# All artefacts (CSVs, PDFs, …) are written next to the CSV by default.
# Override by setting OUTPUT_DIR to an explicit Path.
OUTPUT_DIR: Optional[Path] = r"D:\HybridDualDistillation\iPASECIoT\AllFeatures"   # None → use CSV_PATH.parent
