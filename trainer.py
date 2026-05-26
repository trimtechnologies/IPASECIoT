"""
trainer.py
==========
Unified training and evaluation runner.

Provides two public entry points:
    run_dl_model(name, builder, splits, le, output_dir, output_mode, dataset_name)
    run_ml_model(name, base_estimator, splits, le, output_dir, output_mode,
                 dataset_name, feature_names)

Both return a list of result-dicts that main.py appends to the global
results table.
"""

import time
import copy
import numpy as np
import psutil
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.preprocessing import LabelBinarizer

import config
from evaluator import evaluate_model, save_predictions
from models_ml import IMPORTANCE_MODELS


# ═══════════════════════════════════════════════════════════════════════════
# Resource measurement helpers
# ═══════════════════════════════════════════════════════════════════════════

class _ResourceTimer:
    """Context manager that measures wall time, mean CPU %, and mean RAM %."""

    def __init__(self):
        self._proc = psutil.Process()
        self.elapsed = 0.0
        self.cpu     = 0.0
        self.mem     = 0.0

    def __enter__(self):
        self._cpu0  = psutil.cpu_percent(interval=None)
        self._mem0  = self._proc.memory_percent()
        self._t0    = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self._t0
        cpu1         = psutil.cpu_percent(interval=None)
        mem1         = self._proc.memory_percent()
        self.cpu     = (self._cpu0 + cpu1) / 2
        self.mem     = (self._mem0 + mem1) / 2


# ═══════════════════════════════════════════════════════════════════════════
# Shared result-row builder
# ═══════════════════════════════════════════════════════════════════════════

def _result_row(dataset_name, model_label, metrics,
                train_timer, test_timer, n_train, n_test):
    return {
        'Dataset'                    : dataset_name,
        'Model'                      : model_label,
        'Accuracy'                   : metrics['Accuracy'],
        'Recall'                     : metrics['Recall'],
        'Precision'                  : metrics['Precision'],
        'F1 Score'                   : metrics['F1 Score'],
        'Kappa'                      : metrics['Kappa'],
        'MCC'                        : metrics['MCC'],
        'TNR'                        : metrics['TNR'],
        'FPR'                        : metrics['FPR'],
        'FNR'                        : metrics['FNR'],
        'AUC'                        : metrics['AUC'],
        'Hamming Loss'               : metrics['Hamming Loss'],
        'Train Time (s)'             : train_timer.elapsed,
        'Test Time (s)'              : test_timer.elapsed,
        'Train Time per Sample (s)'  : train_timer.elapsed / max(n_train, 1),
        'Test Time per Sample (s)'   : test_timer.elapsed  / max(n_test,  1),
        'Train Memory Usage (%)'     : train_timer.mem,
        'Test Memory Usage (%)'      : test_timer.mem,
        'Train CPU Usage (%)'        : train_timer.cpu,
        'Test CPU Usage (%)'         : test_timer.cpu,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DL runner
# ═══════════════════════════════════════════════════════════════════════════

def run_dl_model(name, builder, splits, le, output_dir, output_mode, dataset_name):
    """
    Train and evaluate one deep-learning model.

    The Autoencoder is special: it has an extra 'reconstruction' output that
    we must feed during training and strip during inference.

    Parameters
    ----------
    name         : str  — display name (e.g. '1D-CNN')
    builder      : callable — from models_dl.DL_BUILDERS
    splits       : dict returned by data_utils.split_data
    le           : LabelEncoder or (le_device, le_attack)
    output_dir   : Path
    output_mode  : 'multi' | 'device' | 'traffic'
    dataset_name : str

    Returns
    -------
    list of result-row dicts (1 row per output target)
    """
    print(f"\n{'─'*60}")
    print(f"  Training DL model: {name}  [{output_mode}]")
    print(f"{'─'*60}")

    rows      = []
    is_ae     = (name == "Autoencoder")

    X_train   = splits['X_train']
    X_test    = splits['X_test']

    # 3-D input required by all sequential / convolutional DL models
    Xtr_3d    = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    Xte_3d    = X_test.reshape (X_test.shape[0],  X_test.shape[1],  1)

    try:
        if output_mode == 'multi':
            le_device, le_attack = le
            yd_train = splits['y_device_train']
            yd_test  = splits['y_device_test']
            ya_train = splits['y_attack_train']
            ya_test  = splits['y_attack_test']
            y_test   = splits['y_test']

            n_dev = len(np.unique(yd_train))
            n_atk = len(np.unique(ya_train))

            lb_attack = LabelBinarizer().fit(ya_train)
            lb_device = LabelBinarizer().fit(yd_train)

            model = builder((X_train.shape[1], 1), n_dev, n_atk)

            # ── Train ─────────────────────────────────────────────────────
            with _ResourceTimer() as train_t:
                fit_y = {'device': yd_train, 'attack': ya_train}
                if is_ae:
                    flat = Xtr_3d.reshape(len(Xtr_3d), -1)
                    fit_y['reconstruction'] = flat
                model.fit(
                    Xtr_3d, fit_y,
                    epochs=config.DL_EPOCHS,
                    batch_size=config.DL_BATCH_SIZE,
                    validation_split=config.DL_VAL_SPLIT,
                    verbose=1,
                )

            # ── Predict ───────────────────────────────────────────────────
            with _ResourceTimer() as test_t:
                preds = model.predict(Xte_3d)

            # Autoencoder returns [device, attack, reconstruction]
            if is_ae:
                yp_dev_proba, yp_atk_proba = preds[0], preds[1]
            else:
                yp_dev_proba, yp_atk_proba = preds[0], preds[1]

            yp_dev = np.argmax(yp_dev_proba, axis=1)
            yp_atk = np.argmax(yp_atk_proba, axis=1)
            yp_stk = np.column_stack((yp_atk, yp_dev))

            for target, y_true, y_pred, y_proba, enc, label in [
                ('Device Type',    yd_test, yp_dev, yp_dev_proba, le_device, f'{name} (Device)'),
                ('Attack Category',ya_test, yp_atk, yp_atk_proba, le_attack, f'{name} (Attack)'),
            ]:
                met = evaluate_model(
                    y_true, y_pred, enc, f"{target} ({name})",
                    output_dir, y_proba,
                    y_test, yp_stk, lb_attack, lb_device,
                )
                rows.append(_result_row(
                    dataset_name, label, met,
                    train_t, test_t,
                    len(X_train), len(X_test),
                ))

            save_predictions(
                yd_test, yp_dev, yp_dev_proba,
                output_dir, dataset_name, name, le_device,
                output_mode='multi',
                y_true_attack=ya_test, y_pred_attack=yp_atk,
                y_proba_attack=yp_atk_proba, le_attack=le_attack,
            )

        else:  # single output
            target_name = 'Device Type' if output_mode == 'device' else 'Attack Category'
            y_train     = splits['y_train']
            y_test      = splits['y_test']
            n_cls       = len(np.unique(y_train))

            model = builder((X_train.shape[1], 1), n_cls)

            # ── Train ─────────────────────────────────────────────────────
            with _ResourceTimer() as train_t:
                if is_ae:
                    flat  = Xtr_3d.reshape(len(Xtr_3d), -1)
                    fit_y = {'output': y_train, 'reconstruction': flat}
                else:
                    fit_y = y_train
                model.fit(
                    Xtr_3d, fit_y,
                    epochs=config.DL_EPOCHS,
                    batch_size=config.DL_BATCH_SIZE,
                    validation_split=config.DL_VAL_SPLIT,
                    verbose=1,
                )

            # ── Predict ───────────────────────────────────────────────────
            with _ResourceTimer() as test_t:
                preds = model.predict(Xte_3d)

            yp_proba = preds[0] if is_ae else preds
            yp       = np.argmax(yp_proba, axis=1)

            met = evaluate_model(
                y_test, yp, le, f"{target_name} ({name})",
                output_dir, yp_proba,
            )
            rows.append(_result_row(
                dataset_name, f'{name} ({target_name})', met,
                train_t, test_t,
                len(X_train), len(X_test),
            ))

            save_predictions(
                y_test, yp, yp_proba,
                output_dir, dataset_name, name, le,
                output_mode=output_mode,
            )

    except Exception as exc:
        print(f"[ERROR] DL model '{name}': {exc}")
        import traceback; traceback.print_exc()

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ML runner (MultiOutputClassifier + ClassifierChain variants)
# ═══════════════════════════════════════════════════════════════════════════

def run_ml_model(name, base_estimator, splits,
                 le, output_dir, output_mode, dataset_name, feature_names):
    """
    Train and evaluate one ML model.

    In multi-output mode, trains both MultiOutputClassifier and
    ClassifierChain wrappers.  In single-output mode, trains the base
    estimator directly.

    Returns
    -------
    list of result-row dicts
    """
    rows = []

    if output_mode == 'multi':
        rows += _run_ml_multi(
            name, base_estimator, splits, le,
            output_dir, dataset_name, feature_names,
        )
    else:
        rows += _run_ml_single(
            name, base_estimator, splits, le,
            output_dir, output_mode, dataset_name, feature_names,
        )

    return rows


# ── Multi-output ──────────────────────────────────────────────────────────

def _run_ml_multi(name, base_estimator, splits, le,
                  output_dir, dataset_name, feature_names):
    rows     = []
    le_dev, le_atk = le
    X_train  = splits['X_train']
    X_test   = splits['X_test']
    y_train  = splits['y_train']  # stacked (attack, device)
    y_test   = splits['y_test']
    yd_test  = splits['y_device_test']
    ya_test  = splits['y_attack_test']
    yd_train = splits['y_device_train']
    ya_train = splits['y_attack_train']

    lb_attack = LabelBinarizer().fit(ya_train)
    lb_device = LabelBinarizer().fit(yd_train)

    # ──────────────────────────────────────────────────────────────────────
    # 1) MultiOutputClassifier
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ML model: {name} — MultiOutputClassifier  [multi]")
    print(f"{'─'*60}")
    try:
        mo_model = MultiOutputClassifier(copy.deepcopy(base_estimator))

        with _ResourceTimer() as train_t:
            mo_model.fit(X_train, y_train)

        with _ResourceTimer() as test_t:
            yp      = mo_model.predict(X_test)
            yp_prob = mo_model.predict_proba(X_test)

        yp_atk, yp_dev         = yp[:, 0], yp[:, 1]
        yp_atk_proba, yp_dev_proba = yp_prob[0], yp_prob[1]

        for target, y_true, y_pred, y_proba, enc, label in [
            ('Device Type',     yd_test, yp_dev, yp_dev_proba, le_dev, f'{name} MultiOutput (Device)'),
            ('Attack Category', ya_test, yp_atk, yp_atk_proba, le_atk, f'{name} MultiOutput (Attack)'),
        ]:
            met = evaluate_model(
                y_true, y_pred, enc,
                f"{target} ({name} MultiOutput)",
                output_dir, y_proba,
                y_test, yp, lb_attack, lb_device,
            )
            rows.append(_result_row(
                dataset_name, label, met,
                train_t, test_t,
                len(X_train), len(X_test),
            ))

        save_predictions(
            yd_test, yp_dev, yp_dev_proba,
            output_dir, dataset_name, f"{name}_MultiOutput", le_dev,
            output_mode='multi',
            y_true_attack=ya_test, y_pred_attack=yp_atk,
            y_proba_attack=yp_atk_proba, le_attack=le_atk,
        )

        if name in IMPORTANCE_MODELS:
            for i, est in enumerate(mo_model.estimators_):
                tgt = 'Attack' if i == 0 else 'Device'
                _print_feature_importance(est, feature_names, f"{name} MultiOutput ({tgt})")

    except Exception as exc:
        print(f"[ERROR] {name} MultiOutputClassifier: {exc}")
        import traceback; traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────
    # 2) ClassifierChain
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ML model: {name} — ClassifierChain  [multi]")
    print(f"{'─'*60}")
    try:
        chain_model = ClassifierChain(copy.deepcopy(base_estimator), order=[0, 1])

        with _ResourceTimer() as train_t:
            chain_model.fit(X_train, y_train)

        with _ResourceTimer() as test_t:
            yp = chain_model.predict(X_test)

        yp_atk, yp_dev = yp[:, 0], yp[:, 1]

        for target, y_true, y_pred, enc, label in [
            ('Device Type',     yd_test, yp_dev, le_dev, f'{name} Chain (Device)'),
            ('Attack Category', ya_test, yp_atk, le_atk, f'{name} Chain (Attack)'),
        ]:
            met = evaluate_model(
                y_true, y_pred, enc,
                f"{target} ({name} Chain)",
                output_dir, None,
                y_test, yp, lb_attack, lb_device,
            )
            rows.append(_result_row(
                dataset_name, label, met,
                train_t, test_t,
                len(X_train), len(X_test),
            ))

        save_predictions(
            yd_test, yp_dev, None,
            output_dir, dataset_name, f"{name}_Chain", le_dev,
            output_mode='multi',
            y_true_attack=ya_test, y_pred_attack=yp_atk,
            y_proba_attack=None, le_attack=le_atk,
        )

        if name in IMPORTANCE_MODELS:
            for i, est in enumerate(chain_model.estimators_):
                tgt  = 'Attack' if i == 0 else 'Device'
                fnames = (list(feature_names) + ['Attack_Pred']
                          if i == 1 else feature_names)
                _print_feature_importance(est, fnames, f"{name} Chain ({tgt})")

    except Exception as exc:
        print(f"[ERROR] {name} ClassifierChain: {exc}")
        import traceback; traceback.print_exc()

    return rows


# ── Single output ─────────────────────────────────────────────────────────

def _run_ml_single(name, base_estimator, splits, le,
                   output_dir, output_mode, dataset_name, feature_names):
    rows        = []
    target_name = 'Device Type' if output_mode == 'device' else 'Attack Category'
    X_train     = splits['X_train']
    X_test      = splits['X_test']
    y_train     = splits['y_train']
    y_test      = splits['y_test']

    print(f"\n{'─'*60}")
    print(f"  ML model: {name}  [{output_mode}]")
    print(f"{'─'*60}")

    try:
        model = copy.deepcopy(base_estimator)

        with _ResourceTimer() as train_t:
            model.fit(X_train, y_train)

        with _ResourceTimer() as test_t:
            yp       = model.predict(X_test)
            yp_proba = (model.predict_proba(X_test)
                        if hasattr(model, 'predict_proba') else None)

        met = evaluate_model(
            y_test, yp, le, f"{target_name} ({name})",
            output_dir, yp_proba,
        )
        rows.append(_result_row(
            dataset_name, f'{name} ({target_name})', met,
            train_t, test_t,
            len(X_train), len(X_test),
        ))

        save_predictions(
            y_test, yp, yp_proba,
            output_dir, dataset_name, name, le,
            output_mode=output_mode,
        )

        if name in IMPORTANCE_MODELS and hasattr(model, 'feature_importances_'):
            _print_feature_importance(model, feature_names, f"{name} ({target_name})")

    except Exception as exc:
        print(f"[ERROR] {name}: {exc}")
        import traceback; traceback.print_exc()

    return rows


# ── Feature importance ────────────────────────────────────────────────────

def _print_feature_importance(estimator, feature_names, label):
    if not hasattr(estimator, 'feature_importances_'):
        return
    print(f"\nFeature Importances — {label}:")
    pairs = sorted(
        zip(feature_names, estimator.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    for feat, imp in pairs[:20]:      # top-20 to keep output manageable
        print(f"  {feat:<35s}: {imp:.4f}")
