"""
models/catboost_model.py

Optuna-tuned CatBoost for multi-class severity prediction.
Labels are 1-indexed (1-4). CatBoost expects 0-indexed — shift handled here.

CatBoost key advantage: ordered boosting reduces overfitting on repeated
majority-class patterns; handles imbalance via explicit class weights.
"""

import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, OPTUNA_N_TRIALS, CATBOOST_PATH


def _get_class_weights(y_train: np.ndarray) -> list:
    """
    Returns balanced class weights as a list [w0, w1, w2, w3] (0-indexed).
    CatBoost accepts class_weights as a list aligned to 0-indexed class order.
    Explicit weights give more control than auto_class_weights="Balanced".
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # classes are 1-indexed (1,2,3,4); weights aligned to that order
    return weights.tolist()


def _minority_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted F1 emphasising minority classes C1 and C4.
    Weights: C1=0.35, C2=0.10, C3=0.20, C4=0.35.
    """
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=[1, 2, 3, 4], zero_division=0)
    return 0.35 * per_class[0] + 0.10 * per_class[1] + 0.20 * per_class[2] + 0.35 * per_class[3]


def train_catboost_optuna(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple:
    """
    Optuna TPE search for CatBoost.

    All features are numerical after our pipeline, so no cat_features needed.
    Uses explicit class_weights (balanced) instead of auto_class_weights for
    finer control. Optuna objective is minority-weighted F1 (same as XGBoost
    and LightGBM) to force search toward C1/C4 improvements.

    Returns (best_model, best_params, best_macro_f1).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    class_weights = _get_class_weights(y_train)

    # CatBoost Pool for efficient repeated evaluation
    train_pool = Pool(X_train, label=y_train - 1)
    val_pool   = Pool(X_val,   label=y_val   - 1)

    def objective(trial):
        # bootstrap_type controls how subsampling works:
        # "Bayesian" uses bagging_temperature (no subsample param)
        # "Bernoulli" uses subsample directly (like other boosters)
        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        )
        params = {
            "iterations":        trial.suggest_int("iterations", 300, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
            "depth":             trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "border_count":      trial.suggest_int("border_count", 32, 255),
            "random_strength":   trial.suggest_float("random_strength", 0.0, 2.0),
            "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 1, 50),
            "bootstrap_type":    bootstrap_type,
            "loss_function":     "MultiClass",
            "eval_metric":       "MultiClass",
            "class_weights":     class_weights,
            "random_seed":       RANDOM_SEED,
            "verbose":           False,
            "early_stopping_rounds": 30,
        }
        # bagging_temperature only applies to Bayesian; subsample only to Bernoulli
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 2.0)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        preds = model.predict(X_val).flatten() + 1
        return _minority_weighted_f1(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nCatBoost best weighted-minority score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({
        "loss_function":         "MultiClass",
        "eval_metric":           "MultiClass",
        "class_weights":         class_weights,
        "random_seed":           RANDOM_SEED,
        "verbose":               False,
        "early_stopping_rounds": 30,
    })

    # Probe to find best iteration via early stopping
    _probe = CatBoostClassifier(**best_params)
    _probe.fit(train_pool, eval_set=val_pool, verbose=False)
    best_iter = (_probe.best_iteration_ + 1) if _probe.best_iteration_ else best_params["iterations"]
    print(f"Best iteration: {best_iter}")

    best_params["iterations"] = best_iter
    best_params.pop("early_stopping_rounds", None)

    best_model = CatBoostClassifier(**best_params)
    best_model.fit(train_pool, verbose=False)

    # Compute true macro F1 for reporting
    preds_val = best_model.predict(X_val).flatten() + 1
    best_macro_f1 = float(f1_score(y_val, preds_val, average="macro", zero_division=0))
    print(f"Final model val macro_F1: {best_macro_f1:.4f}")

    best_model.save_model(CATBOOST_PATH)
    print(f"Model saved to {CATBOOST_PATH}")
    return best_model, best_params, best_macro_f1
