"""
data/load_data.py

Loads the full US Accidents dataset (7.7M rows) from HuggingFace.

Usage:
    from data.load_data import load_dataset_sample
    df = load_dataset_sample()   # full 7.7M rows
"""

from datasets import load_dataset as hf_load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, HUGGINGFACE_DATASET


def load_dataset_sample(full: bool = True) -> pd.DataFrame:
    """
    Loads the full US Accidents dataset from HuggingFace (7.7M rows).
    The `full` parameter is kept for backward compatibility but ignored —
    the complete dataset is always returned.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns intact.
    """
    print(f"Loading dataset from HuggingFace: {HUGGINGFACE_DATASET}")
    dataset = hf_load_dataset(HUGGINGFACE_DATASET, split="train")
    df = dataset.to_pandas()
    print(f"Full dataset shape: {df.shape}")

    # Drop rows with missing Severity
    df = df.dropna(subset=["Severity"])
    df["Severity"] = df["Severity"].astype(int)

    print(f"Shape after dropping missing Severity: {df.shape}")
    print(f"Class distribution:\n{df['Severity'].value_counts(normalize=True).round(3)}")
    return df.reset_index(drop=True)


def train_val_test_split(df: pd.DataFrame):
    """
    Performs a stratified 70-15-15 train/val/test split.

    Returns
    -------
    (df_train, df_val, df_test) : tuple of DataFrames
    """
    from config import TRAIN_RATIO, VAL_RATIO, RANDOM_SEED

    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        stratify=df["Severity"],
        random_state=RANDOM_SEED,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        stratify=df_temp["Severity"],
        random_state=RANDOM_SEED,
    )
    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    return df_train, df_val, df_test
