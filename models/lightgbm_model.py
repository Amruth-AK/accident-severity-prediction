"""
models/lightgbm_model.py

Optuna-tuned LightGBM for multi-class severity prediction.
Labels are 1-indexed (1-4). LightGBM expects 0-indexed — shift handled here.
"""

import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, OPTUNA_N_TRIALS, LIGHTGBM_PATH


def _get_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))
    return np.array([cw_dict[label] for label in y_train])


def _minority_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted F1 that penalises minority class (C1, C4) failures more than C2/C3.
    Weights: C1=0.35, C2=0.10, C3=0.20, C4=0.35.
    Forces Optuna to search for params that improve the hardest classes, not
    just the easy majority.
    """
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=[1, 2, 3, 4], zero_division=0)
    return 0.35 * per_class[0] + 0.10 * per_class[1] + 0.20 * per_class[2] + 0.35 * per_class[3]


def train_lightgbm_optuna(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple:
    """
    Optuna TPE search for LightGBM.

    LightGBM key advantages over XGBoost:
    - Leaf-wise tree growth (num_leaves controls complexity, not max_depth)
    - Faster on large data via histogram binning
    - DART boosting mode: drops trees randomly during training to reduce
      overfitting on majority class patterns

    Optuna objective: minority-weighted F1 (C1+C4 each weighted 0.35) so
    the search doesn't converge to C2-dominated solutions.

    Returns (best_model, best_params, best_macro_f1).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sample_weights = _get_class_weights(y_train)

    def objective(trial):
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
        params = {
            "objective":        "multiclass",
            "num_class":        4,
            "metric":           "multi_logloss",
            "verbosity":        -1,
            "random_state":     RANDOM_SEED,
            "boosting_type":    boosting_type,
            "n_estimators":     trial.suggest_int("n_estimators", 300, 2000),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 15),
            "learning_rate":    trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 200),
            "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq":   1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain":   trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":      trial.suggest_categorical("extra_trees", [True, False]),
        }
        # DART-specific: drop rate controls regularisation strength
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

        model = lgb.LGBMClassifier(**params)
        # DART does not support early stopping — use fixed n_estimators
        if boosting_type == "dart":
            model.fit(
                X_train, y_train - 1,
                sample_weight=sample_weights,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
        else:
            model.fit(
                X_train, y_train - 1,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val - 1)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
        preds = model.predict(X_val) + 1
        return _minority_weighted_f1(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nLightGBM best weighted-minority score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({
        "objective":    "multiclass",
        "num_class":    4,
        "metric":       "multi_logloss",
        "verbosity":    -1,
        "random_state": RANDOM_SEED,
    })

    boosting_type = best_params.get("boosting_type", "gbdt")

    if boosting_type == "dart":
        # DART: no early stopping — use n_estimators from best trial directly
        best_iter = best_params["n_estimators"]
        print(f"Best iteration (DART, fixed): {best_iter}")
    else:
        # GBDT: probe with early stopping to find true best iteration
        _probe = lgb.LGBMClassifier(**best_params)
        _probe.fit(
            X_train, y_train - 1,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val - 1)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        best_iter = _probe.best_iteration_ or best_params["n_estimators"]
        print(f"Best iteration (early stopping probe): {best_iter}")

    best_params["n_estimators"] = best_iter

    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train - 1, sample_weight=sample_weights,
                   callbacks=[lgb.log_evaluation(period=-1)])

    # Compute true macro F1 for reporting
    preds_val = best_model.predict(X_val) + 1
    best_macro_f1 = float(f1_score(y_val, preds_val, average="macro", zero_division=0))
    print(f"Final model val macro_F1: {best_macro_f1:.4f}")

    joblib.dump(best_model, LIGHTGBM_PATH)
    print(f"Model saved to {LIGHTGBM_PATH}")
    return best_model, best_params, best_macro_f1
