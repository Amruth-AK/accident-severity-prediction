"""
models/xgboost_model.py

Phase 2: Class imbalance strategies + Phase 3: Optuna-tuned XGBoost.

Imbalance strategies implemented:
  1. Class weights (sample_weight in fit)
  2. SMOTE-NC (oversample minority classes on mixed feature types)
  3. Focal loss (custom XGBoost objective)
  4. Per-class threshold tuning (post-hoc, zero retraining cost)

All functions return (model, macro_f1) or (model, macro_f1, extra).
Labels are 1-indexed (1-4). XGBoost internally uses 0-indexed — the
shift is handled inside each function with a clear comment.
"""

import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from itertools import product as iter_product

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, OPTUNA_N_TRIALS, XGBOOST_PATH


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Returns balanced class weights {class_label: weight}.
    XGBoost multi-class requires per-sample weights, not class_weight param.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def weights_to_sample_weights(y: np.ndarray, class_weight_dict: dict) -> np.ndarray:
    """Maps per-class weight dict to a per-sample weight array."""
    return np.array([class_weight_dict[label] for label in y])


def _base_params(tree_method: str) -> dict:
    """Default XGBoost params shared across all strategies."""
    return {
        "objective":        "multi:softprob",
        "num_class":        4,
        "tree_method":      tree_method,
        "random_state":     RANDOM_SEED,
        "eval_metric":      "mlogloss",
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
    }


def _get_tree_method() -> str:
    try:
        import torch
        return "gpu_hist" if torch.cuda.is_available() else "hist"
    except ImportError:
        return "hist"


def evaluate(model, X_val: np.ndarray, y_val: np.ndarray,
             thresholds: np.ndarray = None) -> dict:
    """
    Returns accuracy, macro_f1, weighted_f1, and per-class F1.
    y_val is 1-indexed. If thresholds provided, uses apply_thresholds.
    """
    from sklearn.metrics import accuracy_score, classification_report
    proba = model.predict_proba(X_val)
    if thresholds is not None:
        preds = apply_thresholds(proba, thresholds)
    else:
        preds = proba.argmax(axis=1) + 1  # 0-indexed → 1-indexed

    per_class = f1_score(y_val, preds, average=None, labels=[1, 2, 3, 4],
                         zero_division=0)
    return {
        "accuracy":   round(float(np.mean(preds == y_val)), 4),
        "macro_f1":   round(float(f1_score(y_val, preds, average="macro",    zero_division=0)), 4),
        "weighted_f1":round(float(f1_score(y_val, preds, average="weighted", zero_division=0)), 4),
        "f1_class_1": round(float(per_class[0]), 4),
        "f1_class_2": round(float(per_class[1]), 4),
        "f1_class_3": round(float(per_class[2]), 4),
        "f1_class_4": round(float(per_class[3]), 4),
    }


# ---------------------------------------------------------------------------
# Strategy 1: Class weights
# ---------------------------------------------------------------------------

def train_xgboost_weighted(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple:
    """
    XGBoost with balanced class weights via sample_weight.
    Returns (model, metrics_dict).
    """
    tree_method = _get_tree_method()
    cw = get_class_weights(y_train)
    sw = weights_to_sample_weights(y_train, cw)

    model = xgb.XGBClassifier(**_base_params(tree_method))
    # XGBoost expects 0-indexed labels
    model.fit(
        X_train, y_train - 1,
        sample_weight=sw,
        eval_set=[(X_val, y_val - 1)],
        verbose=False,
    )
    metrics = evaluate(model, X_val, y_val)
    print(f"[Weighted] macro_F1={metrics['macro_f1']:.4f}  "
          f"F1-C1={metrics['f1_class_1']:.4f}  F1-C4={metrics['f1_class_4']:.4f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Strategy 2: SMOTE-NC
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """
    Original Preprocessing.ipynb SMOTE approach (verbatim):
    1. Downsample Class 2 to 500k (meaningful on 7.7M where C2 has ~5.4M training rows)
    2. SMOTE sampling_strategy='auto' — oversamples all minorities to match Class 2

    On the full 7.7M dataset, C1/C3/C4 have tens of thousands of real examples
    to interpolate from, so synthetic quality is high. This approach would not
    be appropriate on a small sample where minorities have only a few thousand rows.

    Returns (X_resampled, y_resampled).
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    actual_counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    print(f"  Class counts before SMOTE: {actual_counts}")

    # Downsample C2 to 500k. Also downsample C3 if it exceeds 500k —
    # otherwise C3 becomes the new majority and SMOTE overshoots the 500k target.
    target_c2 = min(500_000, actual_counts.get(2, 500_000))
    target_c3 = min(500_000, actual_counts.get(3, 500_000))
    under_strategy = {2: target_c2}
    if actual_counts.get(3, 0) > 500_000:
        under_strategy[3] = target_c3
    print(f"  Downsampling to: {under_strategy}...")
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  After undersampling: { {int(c): int(n) for c, n in zip(unique, counts)} }")

    # SMOTE oversamples all minorities to match the majority (500k).
    # With C2 and C3 both at 500k, C1 and C4 get upsampled to 500k each.
    print("  Applying SMOTE (sampling_strategy='auto')...")
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_SEED, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_res, return_counts=True)
    print(f"  Post-SMOTE counts: { {int(c): int(n) for c, n in zip(unique, counts)} }")
    return X_res, y_res


def train_xgboost_smote(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple:
    """
    XGBoost trained on SMOTE-resampled data (all features are numerical).
    Returns (model, metrics_dict).
    """
    tree_method = _get_tree_method()
    X_res, y_res = apply_smote(X_train, y_train)

    model = xgb.XGBClassifier(**_base_params(tree_method))
    model.fit(
        X_res, y_res - 1,
        eval_set=[(X_val, y_val - 1)],
        verbose=False,
    )
    metrics = evaluate(model, X_val, y_val)
    print(f"[SMOTE]    macro_F1={metrics['macro_f1']:.4f}  "
          f"F1-C1={metrics['f1_class_1']:.4f}  F1-C4={metrics['f1_class_4']:.4f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Strategy 3: Focal loss
# ---------------------------------------------------------------------------

def focal_loss_objective(gamma: float = 2.0, n_classes: int = 4):
    """
    Custom XGBoost multi-class objective implementing focal loss.
    Focal weight (1-p_t)^gamma down-weights easy majority-class examples
    and focuses learning on hard minority-class examples.
    gamma=2.0 is the standard starting point.

    XGBoost passes y_pred as a flat (n*k,) array in row-major order.
    n_classes must be passed explicitly — inferring it from y_pred.shape
    is unreliable in early boosting rounds.
    """
    def objective(y_pred, dtrain):
        y_true = dtrain.get_label().astype(int)
        n      = len(y_true)
        # y_pred is always (n * n_classes,) in row-major order
        y_pred = y_pred.reshape(n, n_classes)

        # Softmax
        y_pred = y_pred - y_pred.max(axis=1, keepdims=True)
        y_pred = np.exp(y_pred)
        y_pred /= y_pred.sum(axis=1, keepdims=True)

        p_t          = y_pred[np.arange(n), y_true]
        focal_weight = (1 - p_t) ** gamma

        grad = y_pred.copy()
        grad[np.arange(n), y_true] -= 1
        grad *= focal_weight[:, None]

        hess = 2.0 * y_pred * (1 - y_pred) * focal_weight[:, None]

        return grad.ravel(), hess.ravel()
    return objective


def train_xgboost_focal(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    gamma: float = 2.0,
) -> tuple:
    """
    XGBoost with focal loss custom objective.
    Uses DMatrix API since custom objectives require it.
    Returns (model, metrics_dict).
    """
    tree_method = _get_tree_method()

    dtrain = xgb.DMatrix(X_train, label=y_train - 1)
    dval   = xgb.DMatrix(X_val,   label=y_val   - 1)

    params = {
        "num_class":        4,
        "tree_method":      tree_method,
        "seed":             RANDOM_SEED,
        "eval_metric":      "mlogloss",
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        obj=focal_loss_objective(gamma),
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # Wrap in sklearn-compatible class for evaluate()
    class FocalModel:
        def __init__(self, booster):
            self.booster = booster
        def predict_proba(self, X):
            dm  = xgb.DMatrix(X)
            # output_margin=True returns raw leaf scores (n*k,) before softmax
            raw = self.booster.predict(dm, output_margin=True)
            n   = X.shape[0]
            raw = raw.reshape(n, 4)
            # apply softmax manually
            raw = raw - raw.max(axis=1, keepdims=True)
            raw = np.exp(raw)
            raw /= raw.sum(axis=1, keepdims=True)
            return raw

    model   = FocalModel(booster)
    metrics = evaluate(model, X_val, y_val)
    print(f"[Focal g={gamma}] macro_F1={metrics['macro_f1']:.4f}  "
          f"F1-C1={metrics['f1_class_1']:.4f}  F1-C4={metrics['f1_class_4']:.4f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Strategy 4: Per-class threshold tuning
# ---------------------------------------------------------------------------

def tune_thresholds(
    model,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """
    Searches per-class probability thresholds that maximise macro F1 on val set.

    How threshold scaling works: pred = argmax(proba / threshold).
    - Lowering a threshold boosts that class's scaled probability → model predicts
      it more often. Use for minority classes (C1, C4).
    - Raising a threshold suppresses that class → use for the dominant majority
      class (C2 at ~80%) to push borderline samples toward minority classes.

    Search ranges:
      C1, C4 (minorities): 0.05 – 0.50  (lower = predict more aggressively)
      C2     (majority):   0.50 – 0.95  (higher = predict less aggressively)
      C3     (moderate):   0.30 – 0.80  (both directions possible)

    Returns thresholds array of shape (n_classes,).
    """
    proba = model.predict_proba(X_val)
    best_thresholds = np.array([0.5, 0.5, 0.5, 0.5])
    best_macro_f1   = 0.0

    # Pass 1 — coarse grid (~500 combinations)
    minority_grid = np.arange(0.10, 0.55, 0.10)   # C1, C4: 0.10 0.20 0.30 0.40 0.50
    majority_grid = np.arange(0.50, 1.00, 0.10)   # C2:     0.50 0.60 0.70 0.80 0.90
    mid_grid      = np.arange(0.30, 0.85, 0.10)   # C3:     0.30 0.40 0.50 0.60 0.70 0.80

    for t1, t2, t3, t4 in iter_product(minority_grid, majority_grid,
                                        mid_grid, minority_grid):
        thresholds = np.array([t1, t2, t3, t4])
        preds      = apply_thresholds(proba, thresholds)
        score      = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_macro_f1:
            best_macro_f1   = score
            best_thresholds = thresholds.copy()

    # Pass 2 — fine grid around coarse best (±0.10, step 0.02, ~2500 combinations)
    def fine(center, lo, hi):
        return np.clip(np.arange(center - 0.10, center + 0.12, 0.02), lo, hi)

    for t1, t2, t3, t4 in iter_product(
        fine(best_thresholds[0], 0.02, 0.50),
        fine(best_thresholds[1], 0.50, 0.98),
        fine(best_thresholds[2], 0.20, 0.90),
        fine(best_thresholds[3], 0.02, 0.50),
    ):
        thresholds = np.array([t1, t2, t3, t4])
        preds      = apply_thresholds(proba, thresholds)
        score      = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_macro_f1:
            best_macro_f1   = score
            best_thresholds = thresholds.copy()

    print(f"  Best thresholds: C1={best_thresholds[0]:.2f}, C2={best_thresholds[1]:.2f}, "
          f"C3={best_thresholds[2]:.2f}, C4={best_thresholds[3]:.2f}  "
          f"macro_F1={best_macro_f1:.4f}")
    return best_thresholds


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Applies per-class thresholds by scaling probabilities.
    Assigning sample to class with highest scaled probability.
    Returns 1-indexed predictions.
    """
    scaled = proba / (thresholds + 1e-9)
    return scaled.argmax(axis=1) + 1  # 0-indexed → 1-indexed


# ---------------------------------------------------------------------------
# Phase 3: Optuna-tuned XGBoost
# ---------------------------------------------------------------------------

def _minority_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted combination that penalises C1/C4 failures more than C2/C3.
    Weights: C1=0.35, C2=0.10, C3=0.20, C4=0.35 (must sum to 1.0).
    Using macro F1 alone lets C2 dominate; this objective explicitly
    forces the model to compete on minority classes.
    """
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=[1, 2, 3, 4], zero_division=0)
    return 0.35 * per_class[0] + 0.10 * per_class[1] + 0.20 * per_class[2] + 0.35 * per_class[3]


def train_xgboost_optuna(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    class_imbalance_strategy: str = "weights",
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple:
    """
    Optuna hyperparameter search for XGBoost.
    strategy: "weights" | "focal" | "smote"

    Optuna objective: minority-weighted F1 (C1+C4 each weighted 0.35 vs
    C2=0.10, C3=0.20). This forces trials to compete on minority classes
    rather than optimising macro F1 which is dominated by C2+C3.

    Returns (best_model, best_params, best_macro_f1).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tree_method = _get_tree_method()
    cw = get_class_weights(y_train)
    sw = weights_to_sample_weights(y_train, cw)

    # Prepare training data based on strategy
    if class_imbalance_strategy == "smote":
        X_tr, y_tr = apply_smote(X_train, y_train)
        sw_tr = None
    elif class_imbalance_strategy == "weights":
        X_tr, y_tr = X_train, y_train
        sw_tr = sw
    else:
        # "none" — data is already balanced (e.g. pre-SMOTE'd before calling this)
        X_tr, y_tr = X_train, y_train
        sw_tr = None

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step":    trial.suggest_int("max_delta_step", 0, 10),
            "objective":         "multi:softprob",
            "num_class":         4,
            "tree_method":       tree_method,
            "random_state":      RANDOM_SEED,
            "eval_metric":       "mlogloss",
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBClassifier(**params)
        fit_kwargs = dict(eval_set=[(X_val, y_val - 1)], verbose=False)
        if sw_tr is not None:
            fit_kwargs["sample_weight"] = sw_tr
        model.fit(X_tr, y_tr - 1, **fit_kwargs)

        preds = model.predict(X_val) + 1
        # Minority-weighted objective so trials don't converge to C2-dominated solutions
        return _minority_weighted_f1(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Report the actual macro F1 of the best trial (not the weighted objective)
    best_params = study.best_params.copy()
    best_params.update({
        "objective":   "multi:softprob",
        "num_class":   4,
        "tree_method": tree_method,
        "random_state": RANDOM_SEED,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": 30,
    })
    # Probe to find true best iteration via early stopping
    _probe = xgb.XGBClassifier(**best_params)
    probe_kwargs = dict(eval_set=[(X_val, y_val - 1)], verbose=False)
    if sw_tr is not None:
        probe_kwargs["sample_weight"] = sw_tr
    _probe.fit(X_tr, y_tr - 1, **probe_kwargs)
    best_iter = _probe.best_iteration + 1
    print(f"\nBest weighted-minority score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Best iteration: {best_iter}")

    best_params.update({"n_estimators": best_iter, "early_stopping_rounds": None})
    best_model = xgb.XGBClassifier(**best_params)
    fit_final = {}
    if sw_tr is not None:
        fit_final["sample_weight"] = sw_tr
    best_model.fit(X_tr, y_tr - 1, **fit_final)

    # Compute true macro F1 for reporting
    preds_val = best_model.predict(X_val) + 1
    best_macro_f1 = float(f1_score(y_val, preds_val, average="macro", zero_division=0))
    print(f"Final model val macro_F1: {best_macro_f1:.4f}")

    joblib.dump(best_model, XGBOOST_PATH)
    print(f"Model saved to {XGBOOST_PATH}")
    return best_model, best_params, best_macro_f1


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_imbalance_comparison(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    categorical_indices: list,
) -> dict:
    """
    Runs all four strategies and returns a results dict for comparison.
    Threshold tuning is applied on top of the best-scoring strategy.
    """
    import time
    results = {}

    print("\n" + "="*60)
    print("IMBALANCE STRATEGY COMPARISON")
    print("="*60)

    # Baseline — no imbalance handling
    print("\n[Baseline] No imbalance handling...")
    tree_method = _get_tree_method()
    t0 = time.time()
    base = xgb.XGBClassifier(**_base_params(tree_method))
    base.fit(X_train, y_train - 1, eval_set=[(X_val, y_val - 1)], verbose=False)
    m = evaluate(base, X_val, y_val)
    print(f"[Baseline] macro_F1={m['macro_f1']:.4f}  "
          f"F1-C1={m['f1_class_1']:.4f}  F1-C4={m['f1_class_4']:.4f}  "
          f"({time.time()-t0:.0f}s)")
    results["Baseline"] = m

    # Strategy 1: Class weights
    print("\n[Strategy 1] Class weights...")
    t0 = time.time()
    m1_model, m1 = train_xgboost_weighted(X_train, y_train, X_val, y_val)
    print(f"  ({time.time()-t0:.0f}s)")
    results["ClassWeights"] = m1

    # Strategy 2: SMOTE
    print("\n[Strategy 2] SMOTE...")
    t0 = time.time()
    try:
        m2_model, m2 = train_xgboost_smote(X_train, y_train, X_val, y_val)
        print(f"  ({time.time()-t0:.0f}s)")
        results["SMOTE"] = m2
    except Exception as e:
        print(f"  SMOTE failed: {e}")
        results["SMOTE"] = None
        m2_model = None

    # Strategy 3: Focal loss
    print("\n[Strategy 3] Focal loss (g=2.0)...")
    t0 = time.time()
    m3_model, m3 = train_xgboost_focal(X_train, y_train, X_val, y_val, gamma=2.0)
    print(f"  ({time.time()-t0:.0f}s)")
    results["FocalLoss"] = m3

    # Strategy 4: Threshold tuning on focal loss (best calibrated model)
    print("\n[Strategy 4] Threshold tuning on FocalLoss model...")
    t0 = time.time()
    thresholds = tune_thresholds(m3_model, X_val, y_val)
    m4 = evaluate(m3_model, X_val, y_val, thresholds=thresholds)
    print(f"[FocalLoss+Thresholds] macro_F1={m4['macro_f1']:.4f}  "
          f"F1-C1={m4['f1_class_1']:.4f}  F1-C4={m4['f1_class_4']:.4f}  "
          f"({time.time()-t0:.0f}s)")
    results["FocalLoss+Thresholds"] = m4

    # Also try threshold tuning on baseline (best default model)
    print("\n[Strategy 4b] Threshold tuning on Baseline model...")
    t0 = time.time()
    thresholds_base = tune_thresholds(base, X_val, y_val)
    m4b = evaluate(base, X_val, y_val, thresholds=thresholds_base)
    print(f"[Baseline+Thresholds]  macro_F1={m4b['macro_f1']:.4f}  "
          f"F1-C1={m4b['f1_class_1']:.4f}  F1-C4={m4b['f1_class_4']:.4f}  "
          f"({time.time()-t0:.0f}s)")
    results["Baseline+Thresholds"] = m4b

    # Return best thresholds (from whichever gave higher macro F1)
    thresholds = thresholds if m4['macro_f1'] >= m4b['macro_f1'] else thresholds_base

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Strategy':<28} {'MacroF1':>8} {'F1-C1':>7} {'F1-C2':>7} {'F1-C3':>7} {'F1-C4':>7}")
    print("-"*60)
    for name, m in results.items():
        if m is None:
            print(f"  {name:<26}  FAILED")
            continue
        print(f"  {name:<26} {m['macro_f1']:>8.4f} {m['f1_class_1']:>7.4f} "
              f"{m['f1_class_2']:>7.4f} {m['f1_class_3']:>7.4f} {m['f1_class_4']:>7.4f}")
    print("="*60)

    return results, thresholds
