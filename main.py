"""
main.py
=======
Entry-point for the IPA-SEC multi-output / single-output classification pipeline.

Usage
-----
    python main.py              # uses OUTPUT_MODE from config.py
    python main.py --mode multi
    python main.py --mode device
    python main.py --mode traffic

Edit config.py to change dataset paths, hyper-parameters, and enabled models.
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

import config
from data_utils  import load_and_preprocess, split_data
from models_dl   import DL_BUILDERS
from models_ml   import ML_MODELS
from trainer     import run_dl_model, run_ml_model


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_output_dir() -> Path:
    if config.OUTPUT_DIR is not None:
        d = Path(config.OUTPUT_DIR)
    else:
        d = config.CSV_PATH.parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _print_split_summary(splits, output_mode):
    print("\n── Train / Test split summary ──────────────────────────────")
    print(f"  X_train : {splits['X_train'].shape}")
    print(f"  X_test  : {splits['X_test'].shape}")
    if output_mode == 'multi':
        import numpy as np
        print(f"  y_device_train: {splits['y_device_train'].shape}  "
              f"classes={len(set(splits['y_device_train']))}")
        print(f"  y_attack_train: {splits['y_attack_train'].shape}  "
              f"classes={len(set(splits['y_attack_train']))}")
    else:
        print(f"  y_train : {splits['y_train'].shape}  "
              f"classes={len(set(splits['y_train']))}")


def _save_results(results, output_dir, dataset_name, output_mode):
    if not results:
        print("\n[WARN] No results to save.")
        return
    df   = pd.DataFrame(results)
    path = output_dir / f"metrics_{dataset_name}_{output_mode}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved metrics → {path}")
    print("\n── Results summary ─────────────────────────────────────────")
    print(df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main(output_mode: str):
    print("=" * 70)
    print(f"  IPA-SEC Pipeline   |   output_mode={output_mode}")
    print("=" * 70)

    csv_path     = config.CSV_PATH
    output_dir   = _resolve_output_dir()
    dataset_name = csv_path.stem

    # ── 1. Load & preprocess ─────────────────────────────────────────────
    try:
        X, y_encoded, le, feature_names = load_and_preprocess(
            csv_path,
            target_labels       = config.TARGET_LABELS,
            target_traffic_types= config.TARGET_TRAFFIC_TYPES,
            output_mode         = output_mode,
        )
    except Exception as exc:
        print(f"[FATAL] Data loading failed: {exc}")
        return

    # ── 2. Train / test split ────────────────────────────────────────────
    try:
        splits = split_data(X, y_encoded, output_mode)
        _print_split_summary(splits, output_mode)
    except Exception as exc:
        print(f"[FATAL] Train/test split failed: {exc}")
        return

    results = []

    # ── 3. Deep Learning models ──────────────────────────────────────────
    for name, builder in DL_BUILDERS.items():
        if not config.MODELS_ENABLED.get(name, True):
            print(f"\n[SKIP] {name} (disabled in config)")
            continue
        rows = run_dl_model(
            name, builder, splits, le,
            output_dir, output_mode, dataset_name,
        )
        results.extend(rows)

    # ── 4. Machine Learning models ───────────────────────────────────────
    for name, estimator in ML_MODELS.items():
        if not config.MODELS_ENABLED.get(name, True):
            print(f"\n[SKIP] {name} (disabled in config)")
            continue
        rows = run_ml_model(
            name, estimator, splits, le,
            output_dir, output_mode, dataset_name, feature_names,
        )
        results.extend(rows)

    # ── 5. Persist aggregate results ─────────────────────────────────────
    _save_results(results, output_dir, dataset_name, output_mode)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IPA-SEC — IoT Protocol Attack Security Classifier"
    )
    parser.add_argument(
        '--mode',
        choices=['multi', 'device', 'traffic'],
        default=config.OUTPUT_MODE,
        help="Output mode: 'multi' (default), 'device', or 'traffic'.",
    )
    args = parser.parse_args()
    main(args.mode)
