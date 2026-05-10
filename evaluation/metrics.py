"""
evaluation/metrics.py

Shared evaluation logic used across all phases.
All models are compared on exactly the same metrics with the same plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR, RESULTS_CSV_PATH


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, thresholds: np.ndarray = None) -> dict:
    """
    Returns accuracy, macro_f1, weighted_f1, per-class F1, and classification report.
    y_test is 1-indexed. Pass thresholds array for threshold-tuned evaluation.
    """
    proba = model.predict_proba(X_test)

    if thresholds is not None:
        scaled = proba / (thresholds + 1e-9)
        preds  = scaled.argmax(axis=1) + 1
    else:
        preds = proba.argmax(axis=1) + 1

    per_class = f1_score(y_test, preds, average=None, labels=[1,2,3,4], zero_division=0)
    report    = classification_report(y_test, preds, zero_division=0)

    results = {
        "accuracy":            round(float(accuracy_score(y_test, preds)), 4),
        "macro_f1":            round(float(f1_score(y_test, preds, average="macro",    zero_division=0)), 4),
        "weighted_f1":         round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "f1_class_1":          round(float(per_class[0]), 4),
        "f1_class_2":          round(float(per_class[1]), 4),
        "f1_class_3":          round(float(per_class[2]), 4),
        "f1_class_4":          round(float(per_class[3]), 4),
        "classification_report": report,
    }

    print(f"\n=== {model_name} ===")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    print(f"  F1 per class: C1={results['f1_class_1']:.4f}  C2={results['f1_class_2']:.4f}  "
          f"C3={results['f1_class_3']:.4f}  C4={results['f1_class_4']:.4f}")
    return results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, save: bool = True):
    """Plots and optionally saves confusion matrix."""
    cm  = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"C{i}" for i in range(1,5)],
                yticklabels=[f"C{i}" for i in range(1,5)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"confusion_{model_name.replace(' ','_')}.png")
        plt.savefig(path, dpi=120)
    plt.show()
    plt.close()


def plot_side_by_side_confusion_matrices(
    y_true: np.ndarray,
    preds1: np.ndarray, name1: str,
    preds2: np.ndarray, name2: str,
):
    """Plots two confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, preds, name in zip(axes, [preds1, preds2], [name1, name2]):
        cm = confusion_matrix(y_true, preds, labels=[1,2,3,4])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[f"C{i}" for i in range(1,5)],
                    yticklabels=[f"C{i}" for i in range(1,5)])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_comparison_{name1}_vs_{name2}.png".replace(" ","_"))
    plt.savefig(path, dpi=120)
    plt.show()
    plt.close()


def plot_shap_summary(model, X_test: np.ndarray, feature_names: list,
                      model_name: str):
    """
    SHAP beeswarm plot for tree models. Subsamples to 5000 rows for speed.
    Uses TreeExplainer for XGBoost/LightGBM/CatBoost.
    """
    import shap
    explainer   = shap.TreeExplainer(model)
    sample      = X_test[:5000]
    shap_values = explainer.shap_values(sample)
    feature_names_arr = np.array(feature_names)  # SHAP requires numpy array for indexing

    # Normalise to list-of-2D format regardless of SHAP version:
    # newer SHAP returns (n_samples, n_features, n_classes); older returns list of (n_samples, n_features)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_list = shap_values  # already a list

    for cls_idx, cls_label in [(0, "Class 1"), (3, "Class 4")]:
        plt.figure()
        shap.summary_plot(
            shap_list[cls_idx],
            sample,
            feature_names=feature_names_arr,
            show=False,
            plot_type="bar",
        )
        plt.title(f"SHAP Feature Importance — {model_name} ({cls_label})")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR,
                            f"shap_{model_name.replace(' ','_')}_{cls_label.replace(' ','')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  SHAP plot saved: {path}")


def print_model_comparison_table(all_results: dict):
    """
    Prints a formatted comparison table and saves to RESULTS_CSV_PATH.
    all_results = {"ModelName": evaluate_model(...) return value, ...}
    """
    rows = []
    print(f"\n{'Model':<32} {'Accuracy':>9} {'MacroF1':>8} {'F1-C1':>7} "
          f"{'F1-C2':>7} {'F1-C3':>7} {'F1-C4':>7}")
    print("-" * 82)
    for name, m in all_results.items():
        if m is None:
            continue
        print(f"  {name:<30} {m['accuracy']:>9.4f} {m['macro_f1']:>8.4f} "
              f"{m['f1_class_1']:>7.4f} {m['f1_class_2']:>7.4f} "
              f"{m['f1_class_3']:>7.4f} {m['f1_class_4']:>7.4f}")
        rows.append({"Model": name, **{k: v for k, v in m.items()
                                       if k != "classification_report"}})
    print("-" * 82)

    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(RESULTS_CSV_PATH)
    print(f"\nResults saved to {RESULTS_CSV_PATH}")
    return df


def save_results_csv(all_results: dict):
    """Saves all_results dict to RESULTS_CSV_PATH."""
    rows = [{"Model": name, **{k: v for k, v in m.items()
                                if k != "classification_report"}}
            for name, m in all_results.items() if m is not None]
    pd.DataFrame(rows).set_index("Model").to_csv(RESULTS_CSV_PATH)
    print(f"Results saved to {RESULTS_CSV_PATH}")
