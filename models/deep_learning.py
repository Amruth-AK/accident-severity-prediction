"""
models/deep_learning.py

Phase 4: Deep learning models for tabular accident severity prediction.

Two models:
  1. MLP  — properly regularised baseline (BatchNorm, Dropout, AdamW, cosine LR)
  2. FT-Transformer — via pytorch-tabular, self-attention over feature tokens

Both receive the same preprocessed numpy arrays produced by
PreprocessingPipeline (already scaled). SMOTE is applied to the MLP
training data to match the winning Phase-3 imbalance strategy.

Label convention: pipeline emits 1-indexed labels (1-4).
PyTorch CrossEntropyLoss expects 0-indexed. Shift handled here.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import RANDOM_SEED, DL_MODEL_PATH, PLOTS_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _class_weights_tensor(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Balanced class weights as a tensor for CrossEntropyLoss (0-indexed)."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # y_train is 1-indexed; weights are ordered by class value → index 0 = class 1
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Mirrors the Phase-3 SMOTE strategy:
      1. Downsample class 2 (and class 3 if > 500k) to 500k.
      2. SMOTE auto — oversample minorities to match the majority.

    Applied only to training data. Never call on val/test.
    Returns (X_resampled, y_resampled).
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    counts = {int(c): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))}
    print(f"  Pre-SMOTE class counts: {counts}")

    under_strategy = {2: min(500_000, counts.get(2, 500_000))}
    if counts.get(3, 0) > 500_000:
        under_strategy[3] = 500_000
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    smote = SMOTE(sampling_strategy="auto", random_state=RANDOM_SEED, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    counts_post = {int(c): int(n) for c, n in zip(*np.unique(y_res, return_counts=True))}
    print(f"  Post-SMOTE class counts: {counts_post}")
    return X_res, y_res


def evaluate_dl(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns the standard metrics dict used across all phases."""
    from sklearn.metrics import accuracy_score
    per_class = f1_score(y_true, y_pred, average=None, labels=[1, 2, 3, 4],
                         zero_division=0)
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_class_1":  round(float(per_class[0]), 4),
        "f1_class_2":  round(float(per_class[1]), 4),
        "f1_class_3":  round(float(per_class[2]), 4),
        "f1_class_4":  round(float(per_class[3]), 4),
    }


# ---------------------------------------------------------------------------
# Model 1: MLP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    use_smote: bool = True,
    batch_size: int = 2048,
    max_epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-3,
) -> tuple:
    """
    Trains a regularised MLP with:
      - BatchNorm + Dropout for regularisation
      - AdamW optimiser with weight_decay=1e-4
      - CosineAnnealingLR scheduler
      - CrossEntropyLoss with balanced class weights
      - Early stopping on val macro F1 (patience=10)
      - SMOTE on training data (optional, default True)

    Returns (model, test_metrics, loss_history, val_f1_history).
    loss_history = {"train_loss": [...], "val_loss": [...]}
    val_f1_history = [macro_f1_per_epoch]
    """
    torch.manual_seed(RANDOM_SEED)
    device = _device()
    print(f"MLP training on {device}")

    if use_smote:
        print("Applying SMOTE to training data...")
        X_tr, y_tr = apply_smote(X_train, y_train)
    else:
        X_tr, y_tr = X_train.copy(), y_train.copy()

    cw = _class_weights_tensor(y_tr, device)

    # Build tensors (labels shifted to 0-indexed)
    def _tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y - 1, dtype=torch.long),
        )

    Xt, yt = _tensors(X_tr, y_tr)
    Xv, yv = _tensors(X_val, y_val)
    train_loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model = _MLP(X_tr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_losses, val_losses, val_f1s = [], [], []
    best_val_f1   = -1.0
    best_state    = None
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(Xt))
        scheduler.step()

        # --- validate in batches to avoid OOM ---
        model.eval()
        val_loader_eval = DataLoader(
            TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False, num_workers=0
        )
        val_logits, val_loss_total = [], 0.0
        with torch.no_grad():
            for xb, yb in val_loader_eval:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss_total += criterion(out, yb).item() * len(xb)
                val_logits.append(out.cpu())
        val_out  = torch.cat(val_logits, dim=0)
        val_loss = val_loss_total / len(Xv)
        preds_val = val_out.argmax(dim=1).numpy() + 1
        val_losses.append(val_loss)
        val_macro = _macro_f1(y_val, preds_val)
        val_f1s.append(val_macro)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_macro_F1={val_macro:.4f}")

        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best val macro F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Test evaluation in batches
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_logits = []
    with torch.no_grad():
        for (xb,) in test_loader:
            test_logits.append(model(xb.to(device)).cpu())
    preds_test = torch.cat(test_logits, dim=0).argmax(dim=1).numpy() + 1

    test_metrics = evaluate_dl(y_test, preds_test)
    print(f"\nMLP test results: {test_metrics}")

    # Save model
    mlp_path = os.path.join(MODELS_DIR, "mlp_best.pt")
    torch.save(model.state_dict(), mlp_path)
    print(f"MLP saved to {mlp_path}")

    loss_history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, test_metrics, loss_history, val_f1s


# ---------------------------------------------------------------------------
# Model 2: FT-Transformer (native PyTorch — no pytorch-tabular dependency)
# ---------------------------------------------------------------------------

class _FeatureTokenizer(nn.Module):
    """Projects each scalar feature into a d_token-dim embedding."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class _FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer (Gorishniy et al., 2021).

    Each input feature is projected to a d_token-dim vector. A learnable
    [CLS] token is prepended. Standard Transformer encoder layers let
    features attend to each other. The CLS output is classified.
    """
    def __init__(
        self,
        n_features: int,
        n_classes:  int = 4,
        d_token:    int = 32,
        n_heads:    int = 8,
        n_blocks:   int = 3,
        attn_drop:  float = 0.1,
        ff_drop:    float = 0.1,
    ):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=ff_drop,
            batch_first=True,
            norm_first=True,         # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        tokens = self.tokenizer(x)                          # (B, F, d_token)
        cls    = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, F+1, d_token)
        out    = self.transformer(tokens)                   # (B, F+1, d_token)
        return self.head(out[:, 0])                         # CLS token → logits


def train_ft_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    feature_names: list,
    use_smote:  bool = True,
    batch_size: int  = 1024,
    max_epochs: int  = 30,
    patience:   int  = 7,
    d_token:    int  = 32,
    n_heads:    int  = 8,
    n_blocks:   int  = 3,
    lr:         float = 1e-4,
) -> tuple:
    """
    Native PyTorch FT-Transformer — no pytorch-tabular dependency.

    Architecture:
      FeatureTokenizer (per-feature linear projection to d_token)
      → prepend CLS token
      → n_blocks x TransformerEncoderLayer (pre-norm, nhead attention)
      → CLS output → LayerNorm → Linear(n_classes)

    Training:
      - AdamW, lr=1e-4, weight_decay=1e-4
      - CosineAnnealingLR
      - CrossEntropyLoss with balanced class weights
      - SMOTE on training data (optional)
      - Early stopping on val macro F1 (patience=7)

    Returns (model, test_metrics, loss_history, val_f1_history).
    """
    torch.manual_seed(RANDOM_SEED)
    device = _device()
    print(f"FT-Transformer training on {device}")

    if use_smote:
        print("Applying SMOTE to training data...")
        X_tr, y_tr = apply_smote(X_train, y_train)
    else:
        X_tr, y_tr = X_train.copy(), y_train.copy()

    cw = _class_weights_tensor(y_tr, device)

    def _tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y - 1, dtype=torch.long),
        )

    Xt, yt = _tensors(X_tr, y_tr)
    Xv, yv = _tensors(X_val, y_val)
    train_loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    n_features = X_tr.shape[1]
    model = _FTTransformer(
        n_features=n_features,
        d_token=d_token,
        n_heads=n_heads,
        n_blocks=n_blocks,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_losses, val_losses, val_f1s = [], [], []
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(Xt))
        scheduler.step()

        model.eval()
        val_loader = DataLoader(
            TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False, num_workers=0
        )
        val_logits, val_loss_total = [], 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss_total += criterion(out, yb).item() * len(xb)
                val_logits.append(out.cpu())
        val_out  = torch.cat(val_logits, dim=0)
        val_loss = val_loss_total / len(Xv)
        preds_val = val_out.argmax(dim=1).numpy() + 1
        val_losses.append(val_loss)
        val_macro = _macro_f1(y_val, preds_val)
        val_f1s.append(val_macro)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_macro_F1={val_macro:.4f}")

        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best val macro F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_logits = []
    with torch.no_grad():
        for (xb,) in test_loader:
            test_logits.append(model(xb.to(device)).cpu())
    preds_test = torch.cat(test_logits, dim=0).argmax(dim=1).numpy() + 1

    test_metrics = evaluate_dl(y_test, preds_test)
    print(f"\nFT-Transformer test results: {test_metrics}")

    ft_path = os.path.join(MODELS_DIR, "ft_transformer_best.pt")
    torch.save(model.state_dict(), ft_path)
    print(f"FT-Transformer saved to {ft_path}")

    loss_history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, test_metrics, loss_history, val_f1s


# ---------------------------------------------------------------------------
# Loss / F1 curve plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(
    loss_history: dict,
    model_name: str,
    val_f1_history: list = None,
    save: bool = True,
) -> None:
    """
    Plots train and validation loss curves, with an optional second axis
    for validation macro F1 (MLP only).

    loss_history = {"train_loss": [...], "val_loss": [...]}
    Saves to PLOTS_DIR/{model_name}_loss_curves.png
    """
    epochs = range(1, len(loss_history["train_loss"]) + 1)

    if val_f1_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))

    ax1.plot(epochs, loss_history["train_loss"], label="Train loss", linewidth=1.8)
    if "val_loss" in loss_history and loss_history["val_loss"]:
        ax1.plot(epochs, loss_history["val_loss"], label="Val loss", linewidth=1.8, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if val_f1_history is not None:
        ax2.plot(range(1, len(val_f1_history) + 1), val_f1_history,
                 color="green", linewidth=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Macro F1")
        ax2.set_title(f"{model_name} — Validation Macro F1")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_loss_curves.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_final_comparison(all_results: dict, save: bool = True) -> None:
    """
    Horizontal bar chart comparing macro F1 across all models (Phases 2-4).
    all_results = {"ModelName": metrics_dict, ...}
    """
    names  = list(all_results.keys())
    macro  = [all_results[n]["macro_f1"] for n in names]

    colors = ["#4C72B0"] * len(names)
    # Highlight deep learning models
    for i, n in enumerate(names):
        if "MLP" in n or "FT-Transformer" in n or "Transformer" in n:
            colors[i] = "#DD8452"

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.55)))
    bars = ax.barh(names, macro, color=colors)
    ax.set_xlabel("Macro F1")
    ax.set_title("All Models — Macro F1 Comparison")
    ax.set_xlim(0, max(macro) * 1.15)
    for bar, val in zip(bars, macro):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.axvline(x=0.55, color="red", linestyle="--", linewidth=1, label="Original baseline (0.55)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "phase4_final_comparison.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()
