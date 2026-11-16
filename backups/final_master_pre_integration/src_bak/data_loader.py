"""
Data loader module for reading CNN/DailyMail and XSum datasets from local Parquet files.
Handles deterministic sampling, subsetting, and column mapping.
"""

import os
import glob
from typing import Tuple, Optional
import pandas as pd
import pyarrow.parquet as pq


def get_parquet_files(dataset_name: str, split: str, base_dir: str = "datasets") -> list[str]:
    """
    Locates parquet files for a given dataset and split.
    """
    dataset_name = dataset_name.lower()
    split = split.lower()

    if dataset_name in ["cnn_dailymail", "cnn", "dailymail"]:
        search_path = os.path.join(base_dir, "cnn_dailymail", "3.0.0", f"{split}*.parquet")
    elif dataset_name == "xsum":
        search_path = os.path.join(base_dir, "xsum", "data", f"{split}*.parquet")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Must be 'cnn_dailymail' or 'xsum'.")

    files = sorted(glob.glob(search_path))
    if not files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {search_path}")

    return files


def load_dataset_split(
    dataset_name: str,
    split: str,
    base_dir: str = "datasets",
    sample_size: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Loads a specific split of a dataset from local Parquet files, renames columns,
    filters out invalid/empty rows, and applies deterministic sampling.
    """
    files = get_parquet_files(dataset_name, split, base_dir=base_dir)

    # Read parquet files into pandas DataFrame
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Map column names to standard ('source_text', 'target_summary', 'id')
    dataset_name = dataset_name.lower()
    if dataset_name in ["cnn_dailymail", "cnn", "dailymail"]:
        column_mapping = {
            "article": "source_text",
            "highlights": "target_summary",
            "id": "id"
        }
    elif dataset_name == "xsum":
        column_mapping = {
            "document": "source_text",
            "summary": "target_summary",
            "id": "id"
        }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    df = df.rename(columns=column_mapping)

    # Select only relevant columns
    df = df[["source_text", "target_summary", "id"]].copy()

    # Drop null or empty string records
    df = df.dropna(subset=["source_text", "target_summary"])
    df = df[(df["source_text"].str.strip() != "") & (df["target_summary"].str.strip() != "")]

    # Deterministic sampling if sample_size is specified
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def load_summarization_dataset(
    dataset_name: str = "cnn_dailymail",
    train_size: Optional[int] = 5000,
    val_size: Optional[int] = 500,
    test_size: Optional[int] = 500,
    seed: int = 42,
    base_dir: str = "datasets"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test splits for the specified dataset.
    """
    train_df = load_dataset_split(dataset_name, "train", base_dir=base_dir, sample_size=train_size, seed=seed)
    val_df = load_dataset_split(dataset_name, "validation", base_dir=base_dir, sample_size=val_size, seed=seed)
    test_df = load_dataset_split(dataset_name, "test", base_dir=base_dir, sample_size=test_size, seed=seed)

    return train_df, val_df, test_df
