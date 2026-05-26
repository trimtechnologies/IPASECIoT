"""
evaluator.py
============
Model evaluation: metrics, classification report, confusion matrix,
confidence-score logging, and prediction CSV export.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    hamming_loss, roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])

# ── Label helpers ──────────────────────────────────────────────────────────

def shorten_label(label: str, max_length: int = 10) -> str:
    return str(label)[:max_length]


# ══════════════════════════════════════════════════════════════════════════
# Main evaluation function
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    y_true,
    y_pred,
    label_encoder,
    target_name: str,
    output_dir: Path,
    y_pred_proba=None,
    # multi-label Hamming support
    y_true_multi=None,
    y_pred_multi=None,
    lb_attack=None,
    lb_device=None,
) -> dict:
    """
    Compute and log full evaluation metrics for one prediction target.

    Parameters
    ----------
    y_true        : 1-D int array of true encoded labels
    y_pred        : 1-D int array of predicted encoded labels
    label_encoder : fitted LabelEncoder for this target
    target_name   : human-readable name (used in prints & file names)
    output_dir    : directory where CSVs / PDFs are written
    y_pred_proba  : (n_samples, n_classes) probability matrix, or None
    y_true_multi / y_pred_multi / lb_attack / lb_device :
                    optional artefacts for joint Hamming-loss computation

    Returns
    -------
    dict with keys: Accuracy, Recall, Precision, F1 Score, Kappa, MCC,
                    TNR, FPR, FNR, AUC, Hamming Loss, Per-Class Accuracy,
                    Confidence Scores
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {target_name}")
    print(f"{'='*60}")

    if y_true.ndim > 1:
        raise ValueError(f"y_true must be 1-D; got shape {y_true.shape}")

    # ── Valid label subset ────────────────────────────────────────────────
    unique_cls    = np.unique(np.concatenate([y_true, y_pred]))
    valid_labels  = [i for i, _ in enumerate(label_encoder.classes_)
                     if i in unique_cls]
    valid_names   = [label_encoder.classes_[i] for i in valid_labels]
    short_names   = [shorten_label(n) for n in valid_names]

    mask              = np.isin(y_true, valid_labels) & np.isin(y_pred, valid_labels)
    y_true_f          = y_true[mask]
    y_pred_f          = y_pred[mask]
    y_proba_f         = y_pred_proba[mask] if y_pred_proba is not None else None

    # ── Hamming Loss (multi-label) ────────────────────────────────────────
    hamming = 0.0
    if (y_true_multi is not None and y_pred_multi is not None
            and lb_attack is not None and lb_device is not None):
        try:
            yt_bin = np.hstack((
                lb_attack.transform(y_true_multi[:, 0]),
                lb_device.transform(y_true_multi[:, 1]),
            ))
            yp_bin = np.hstack((
                lb_attack.transform(y_pred_multi[:, 0]),
                lb_device.transform(y_pred_multi[:, 1]),
            ))
            hamming = hamming_loss(yt_bin[mask], yp_bin[mask])
            print(f"Hamming Loss : {hamming:.4f}")
        except Exception as exc:
            print(f"[WARN] Hamming loss skipped: {exc}")

    if len(y_true_f) == 0:
        print(f"[WARN] No valid samples for {target_name}. Returning zeros.")
        return _zero_metrics(hamming)

    # ── Scalar metrics ────────────────────────────────────────────────────
    acc   = accuracy_score(y_true_f, y_pred_f)
    rec   = recall_score   (y_true_f, y_pred_f, average='macro', zero_division=0)
    prec  = precision_score(y_true_f, y_pred_f, average='macro', zero_division=0)
    f1    = f1_score       (y_true_f, y_pred_f, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true_f, y_pred_f)
    mcc   = matthews_corrcoef(y_true_f, y_pred_f)

    # ── AUC ──────────────────────────────────────────────────────────────
    auc = 0.0
    if y_proba_f is not None:
        try:
            auc = roc_auc_score(
                y_true_f, y_proba_f,
                multi_class='ovr', average='macro',
            )
            print(f"AUC          : {auc:.4f}")
        except Exception as exc:
            print(f"[WARN] AUC skipped: {exc}")

    # ── Confusion matrix + per-class stats ───────────────────────────────
    cm = confusion_matrix(y_true_f, y_pred_f, labels=valid_labels)

    per_class_acc   = {}
    tnr_per_class   = {}
    fpr_per_class   = {}
    fnr_per_class   = {}

    for i, lbl in enumerate(valid_labels):
        name = valid_names[i]
        tp   = cm[i, i]
        row  = cm[i].sum()
        per_class_acc[name] = tp / row if row > 0 else 0.0

        tn  = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp  = cm[:, i].sum() - cm[i, i]
        fn  = cm[i].sum()    - cm[i, i]

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tnr_per_class[name] = tnr
        fpr_per_class[name] = 1 - tnr
        fnr_per_class[name] = fnr

    tnr = float(np.mean(list(tnr_per_class.values())))
    fpr = float(np.mean(list(fpr_per_class.values())))
    fnr = float(np.mean(list(fnr_per_class.values())))

    # ── Classification report (table) ────────────────────────────────────
    _print_classification_report(
        y_true_f, y_pred_f,
        valid_labels, valid_names,
        per_class_acc, prec, rec, f1, acc,
        target_name, output_dir,
    )

    # ── Confusion matrix (table + heatmap) ───────────────────────────────
    _print_confusion_matrix(cm, short_names, target_name, output_dir)

    return {
        'Accuracy'        : acc,
        'Recall'          : rec,
        'Precision'       : prec,
        'F1 Score'        : f1,
        'Kappa'           : kappa,
        'MCC'             : mcc,
        'TNR'             : tnr,
        'FPR'             : fpr,
        'FNR'             : fnr,
        'AUC'             : auc,
        'Hamming Loss'    : hamming,
        'Per-Class Accuracy': per_class_acc,
        'Confidence Scores' : y_proba_f,
    }


# ══════════════════════════════════════════════════════════════════════════
# Prediction export
# ══════════════════════════════════════════════════════════════════════════

def save_predictions(
    y_true_device, y_pred_device, y_proba_device,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    le_device,
    output_mode: str = 'multi',
    y_true_attack=None, y_pred_attack=None, y_proba_attack=None,
    le_attack=None,
):
    """Save true vs. predicted labels (and confidence scores) to CSV."""
    try:
        if output_mode == 'multi':
            true_dev  = le_device.inverse_transform(y_true_device.astype(int))
            pred_dev  = le_device.inverse_transform(y_pred_device.astype(int))
            true_atk  = le_attack.inverse_transform(y_true_attack.astype(int))
            pred_atk  = le_attack.inverse_transform(y_pred_attack.astype(int))
            dev_conf  = (np.max(y_proba_device, axis=1)
                         if y_proba_device is not None
                         else np.ones(len(y_pred_device)))
            atk_conf  = (np.max(y_proba_attack, axis=1)
                         if y_proba_attack is not None
                         else np.ones(len(y_pred_attack)))
            df = pd.DataFrame({
                'True Device'           : true_dev,
                'Predicted Device'      : pred_dev,
                'Device Confidence'     : dev_conf,
                'True Traffic'          : true_atk,
                'Predicted Traffic'     : pred_atk,
                'Traffic Confidence'    : atk_conf,
            })
        else:
            true_lbl  = le_device.inverse_transform(y_true_device.astype(int))
            pred_lbl  = le_device.inverse_transform(y_pred_device.astype(int))
            conf      = (np.max(y_proba_device, axis=1)
                         if y_proba_device is not None
                         else np.ones(len(y_pred_device)))
            tname = 'Device' if output_mode == 'device' else 'Traffic'
            df = pd.DataFrame({
                f'True {tname}'         : true_lbl,
                f'Predicted {tname}'    : pred_lbl,
                f'{tname} Confidence'   : conf,
            })

        fname = output_dir / f"predictions_{dataset_name}_{model_name.replace(' ', '_')}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved predictions → {fname}")
    except Exception as exc:
        print(f"[ERROR] Saving predictions for {model_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════

def _zero_metrics(hamming: float = 0.0) -> dict:
    return {
        'Accuracy': 0, 'Recall': 0, 'Precision': 0, 'F1 Score': 0,
        'Kappa': 0, 'MCC': 0, 'TNR': 0, 'FPR': 0, 'FNR': 0,
        'AUC': 0, 'Hamming Loss': hamming,
        'Per-Class Accuracy': {}, 'Confidence Scores': None,
    }


def _print_classification_report(
    y_true_f, y_pred_f,
    valid_labels, valid_names,
    per_class_acc, prec, rec, f1, acc,
    target_name, output_dir,
):
    try:
        report = classification_report(
            y_true_f, y_pred_f,
            labels=valid_labels, target_names=valid_names,
            zero_division=0, output_dict=True,
        )
        rows = []
        for name in valid_names:
            m = report.get(name, {})
            rows.append([
                name,
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall',    0):.4f}",
                f"{m.get('f1-score',  0):.4f}",
                int(m.get('support',  0)),
                f"{per_class_acc.get(name, 0):.4f}",
            ])
        rows.append(['macro avg',
                     f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                     len(y_true_f), f"{acc:.4f}"])
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy']
        print(f"\nClassification Report — {target_name}")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        pd.DataFrame(rows, columns=headers).to_csv(
            output_dir / f"classification_report_{target_name.replace(' ', '_')}.csv",
            index=False,
        )
    except Exception as exc:
        print(f"[WARN] Classification report error: {exc}")


def _print_confusion_matrix(cm, short_names, target_name, output_dir):
    try:
        rows    = [[sn] + cm[i].tolist() for i, sn in enumerate(short_names)]
        headers = ['True\\Pred'] + short_names
        print(f"\nConfusion Matrix — {target_name}")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        pd.DataFrame(rows, columns=headers).to_csv(
            output_dir / f"confusion_matrix_{target_name.replace(' ', '_')}.csv",
            index=False,
        )

        plt.figure(figsize=(max(6, len(short_names)), max(5, len(short_names) - 1)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='tab20b',
                    xticklabels=short_names, yticklabels=short_names)
        plt.title(f"Confusion Matrix — {target_name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_matrix_{target_name.replace(' ', '_')}.pdf",
            format='pdf', dpi=300,
        )
        plt.close()
    except Exception as exc:
        print(f"[WARN] Confusion matrix error: {exc}")
