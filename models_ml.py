"""
models_ml.py
============
Machine-learning model registry.

Each entry in ML_MODELS is a (model_name, base_estimator) pair.
The runner in trainer.py wraps them in MultiOutputClassifier / ClassifierChain
automatically when output_mode == 'multi'.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier as SklearnMLP
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _lgbm_available = True
except ImportError:
    _lgbm_available = False
    print("[WARN] LightGBM not installed — 'LightGBM' model will be skipped.")

import config

# ---------------------------------------------------------------------------
# Base estimators
# ---------------------------------------------------------------------------

def _make_ml_models() -> dict:
    """Return {name: base_estimator} for all enabled ML models."""
    models = {}

    if config.MODELS_ENABLED.get("XGBoost", True):
        models["XGBoost"] = XGBClassifier(
            eval_metric='mlogloss',
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )

    if config.MODELS_ENABLED.get("Random Forest", True):
        models["Random Forest"] = RandomForestClassifier(
            random_state=config.RANDOM_SEED,
            class_weight='balanced',
            n_jobs=-1,
        )

    if config.MODELS_ENABLED.get("Decision Tree", True):
        models["Decision Tree"] = DecisionTreeClassifier(
            random_state=config.RANDOM_SEED,
            class_weight='balanced',
        )

    if config.MODELS_ENABLED.get("AdaBoost", True):
        models["AdaBoost"] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            random_state=config.RANDOM_SEED,
        )

    if config.MODELS_ENABLED.get("Extra Trees", True):
        models["Extra Trees"] = ExtraTreesClassifier(
            random_state=config.RANDOM_SEED,
            class_weight='balanced',
            n_jobs=-1,
        )

    if config.MODELS_ENABLED.get("LightGBM", True) and _lgbm_available:
        models["LightGBM"] = LGBMClassifier(
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )

    return models


# Public constant — imported by trainer.py
ML_MODELS: dict = _make_ml_models()

# Models that expose feature_importances_ (used for importance reporting)
IMPORTANCE_MODELS = {"Random Forest", "Decision Tree", "Extra Trees", "LightGBM"}
