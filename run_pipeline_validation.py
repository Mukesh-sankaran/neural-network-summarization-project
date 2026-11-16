"""
Data Pipeline Validation & Exploratory Data Analysis (EDA) Script.
Performs unit-style validation checks on a small 20-example subset first,
then runs full EDA and vocabulary construction on the 5000/500/500 CNN/DailyMail configuration.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_summarization_dataset, get_parquet_files
from src.preprocess import tokenize, clean_text, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN
from src.vocabulary import Vocabulary
from src.dataset import SummarizationDataset, get_dataloaders


def run_unit_tests_on_small_subset():
    print("=" * 60)
    print("STEP 1: RUNNING DATA PIPELINE VALIDATION TESTS (20 EXAMPLES SUBSET)")
    print("=" * 60)

    # 1. Check dataset files exist
    print("-> Checking dataset file existence...")
    cnn_files = get_parquet_files("cnn_dailymail", "train")
    xsum_files = get_parquet_files("xsum", "train")
    assert len(cnn_files) > 0, "CNN/DailyMail parquet files missing!"
    assert len(xsum_files) > 0, "XSum parquet files missing!"
    print(f"   [PASS] Found {len(cnn_files)} CNN/DailyMail files and {len(xsum_files)} XSum files.")

    # 2. Load small subset (20 examples)
    print("-> Loading small subset (20 train, 10 val, 10 test)...")
    train_df, val_df, test_df = load_summarization_dataset(
        dataset_name="cnn_dailymail", train_size=20, val_size=10, test_size=10, seed=42
    )

    # 3. Check correct columns detected and renamed
    expected_cols = {"source_text", "target_summary", "id"}
    assert expected_cols.issubset(set(train_df.columns)), f"Missing columns in train_df: {train_df.columns}"
    print(f"   [PASS] Correct columns detected: {list(train_df.columns)}")

    # 4. Check non-empty records
    assert (train_df["source_text"].str.strip() != "").all(), "Found empty source_text records!"
    assert (train_df["target_summary"].str.strip() != "").all(), "Found empty target_summary records!"
    print("   [PASS] All records contain non-empty source and target text.")

    # 5. Check tokenization
    sample_text = "On 2026-08-21, Apple Inc. reported a 15.5% profit increase, earning $3.4 billion!"
    tokens = tokenize(sample_text)
    assert isinstance(tokens, list) and len(tokens) > 0, "Tokenization failed!"
    print(f"   [PASS] Tokenization working. Sample token count: {len(tokens)}.")

    # 6. Check Vocabulary construction & special tokens
    print("-> Constructing vocabulary on 20 examples...")
    tokenized_train = [tokenize(text) for text in train_df["source_text"]]
    vocab = Vocabulary(max_size=1000)
    vocab.build_vocab(tokenized_train)

    assert vocab.pad_idx == 0 and vocab.idx2word[0] == PAD_TOKEN, "<PAD> index/token mismatch"
    assert vocab.unk_idx == 1 and vocab.idx2word[1] == UNK_TOKEN, "<UNK> index/token mismatch"
    assert vocab.sos_idx == 2 and vocab.idx2word[2] == SOS_TOKEN, "<SOS> index/token mismatch"
    assert vocab.eos_idx == 3 and vocab.idx2word[3] == EOS_TOKEN, "<EOS> index/token mismatch"
    print("   [PASS] Vocabulary contains all 4 special tokens (<PAD>, <UNK>, <SOS>, <EOS>) at fixed indices 0, 1, 2, 3.")

    # 7. Check PyTorch Dataset tensor dimensions
    max_src_len = 300
    max_tgt_len = 60
    dataset = SummarizationDataset(train_df, vocab, max_source_len=max_src_len, max_target_len=max_tgt_len)
    src_tensor, tgt_tensor = dataset[0]

    assert src_tensor.shape == (max_src_len,), f"Expected source tensor shape ({max_src_len},), got {src_tensor.shape}"
    assert tgt_tensor.shape == (max_tgt_len,), f"Expected target tensor shape ({max_tgt_len},), got {tgt_tensor.shape}"
    assert src_tensor.dtype == torch.long, f"Expected torch.long for source tensor, got {src_tensor.dtype}"
    assert tgt_tensor.dtype == torch.long, f"Expected torch.long for target tensor, got {tgt_tensor.dtype}"
    print(f"   [PASS] Dataset output tensor shapes: src={src_tensor.shape}, tgt={tgt_tensor.shape}, dtype={src_tensor.dtype}.")

    # 8. Check DataLoader batch shapes
    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, vocab, batch_size=batch_size, max_source_len=max_src_len, max_target_len=max_tgt_len
    )
    batch_src, batch_tgt = next(iter(train_loader))
    assert batch_src.shape == (batch_size, max_src_len), f"Expected batch src shape ({batch_size}, {max_src_len}), got {batch_src.shape}"
    assert batch_tgt.shape == (batch_size, max_tgt_len), f"Expected batch tgt shape ({batch_size}, {max_tgt_len}), got {batch_tgt.shape}"
    print(f"   [PASS] DataLoader batch shapes: src={batch_src.shape}, tgt={batch_tgt.shape}.")
    print("\nALL 8 VALIDATION TESTS PASSED SUCCESSFULLY ON SMALL SUBSET!\n")


def run_prototype_eda_and_pipeline(
    dataset_name: str = "cnn_dailymail",
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    vocab_size: int = 20000,
    max_source_len: int = 300,
    max_target_len: int = 60,
    batch_size: int = 16
):
    print("=" * 60)
    print(f"STEP 2: RUNNING PROTOTYPE PIPELINE & EDA FOR {dataset_name.upper()}")
    print(f"Configuration: train={train_size}, val={val_size}, test={test_size}, vocab={vocab_size}")
    print("=" * 60)

    # 1. Load Data Subsets
    train_df, val_df, test_df = load_summarization_dataset(
        dataset_name=dataset_name,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=42
    )

    print(f"Dataset Loaded Successfully:")
    print(f"  - Train examples:      {len(train_df)}")
    print(f"  - Validation examples: {len(val_df)}")
    print(f"  - Test examples:       {len(test_df)}")

    # 2. Tokenize and Compute Length Statistics
    print("\nTokenizing training data to build vocabulary...")
    train_src_tokens = [tokenize(text) for text in train_df["source_text"]]
    train_tgt_tokens = [tokenize(text) for text in train_df["target_summary"]]

    val_src_tokens = [tokenize(text) for text in val_df["source_text"]]
    val_tgt_tokens = [tokenize(text) for text in val_df["target_summary"]]

    test_src_tokens = [tokenize(text) for text in test_df["source_text"]]
    test_tgt_tokens = [tokenize(text) for text in test_df["target_summary"]]

    # Combine for EDA stats across full prototype split
    all_src_lens = [len(t) for t in train_src_tokens + val_src_tokens + test_src_tokens]
    all_tgt_lens = [len(t) for t in train_tgt_tokens + val_tgt_tokens + test_tgt_tokens]

    avg_src_len = np.mean(all_src_lens)
    max_src_len = np.max(all_src_lens)
    avg_tgt_len = np.mean(all_tgt_lens)
    max_tgt_len = np.max(all_tgt_lens)

    truncated_src_pct = (np.array(all_src_lens) > max_source_len).mean() * 100
    # Note: target summary includes <SOS> and <EOS> (+2 tokens)
    truncated_tgt_pct = (np.array(all_tgt_lens) + 2 > max_target_len).mean() * 100

    print("\nSequence Length Analysis:")
    print(f"  - Source Text Token Lengths:  Avg = {avg_src_len:.2f}, Max = {max_src_len}")
    print(f"  - Target Summary Token Lengths: Avg = {avg_tgt_len:.2f}, Max = {max_tgt_len}")
    print(f"  - % Source Examples Truncated (> {max_source_len} tokens): {truncated_src_pct:.2f}%")
    print(f"  - % Target Summaries Truncated (> {max_target_len} tokens including <SOS>/<EOS>): {truncated_tgt_pct:.2f}%")

    # 3. Build & Save Vocabulary ONLY from Training Data
    print("\nBuilding vocabulary ONLY from training set (source + target tokens)...")
    tokenized_train_all = train_src_tokens + train_tgt_tokens
    vocab = Vocabulary(max_size=vocab_size)
    vocab.build_vocab(tokenized_train_all)

    vocab_save_path = "models/vocab.pkl"
    vocab.save(vocab_save_path)
    print(f"Vocabulary constructed and saved to: [vocab.pkl]({vocab_save_path})")
    print(f"  - Final Vocabulary Size: {len(vocab)} unique tokens")

    # 4. Build DataLoaders & Verify Tensors
    print("\nBuilding PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, vocab,
        batch_size=batch_size,
        max_source_len=max_source_len,
        max_target_len=max_target_len
    )

    sample_src_b, sample_tgt_b = next(iter(train_loader))
    print(f"DataLoaders ready:")
    print(f"  - Train Loader Batches: {len(train_loader)} (Batch size = {batch_size})")
    print(f"  - Sample Source Batch Tensor Shape: {sample_src_b.shape}")
    print(f"  - Sample Target Batch Tensor Shape: {sample_tgt_b.shape}")

    # 5. Plot and Save Length Distribution Histograms
    print("\nPlotting sequence length distributions...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(all_src_lens, bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.axvline(max_source_len, color="red", linestyle="--", linewidth=2, label=f"Max Limit ({max_source_len})")
    ax1.set_title("Source Document Token Length Distribution")
    ax1.set_xlabel("Token Count")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.hist(all_tgt_lens, bins=30, color="salmon", edgecolor="black", alpha=0.7)
    ax2.axvline(max_target_len - 2, color="red", linestyle="--", linewidth=2, label=f"Max Content Limit ({max_target_len-2})")
    ax2.set_title("Target Summary Token Length Distribution")
    ax2.set_xlabel("Token Count")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = "results/sequence_lengths.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Histogram plot saved successfully to: [sequence_lengths.png]({plot_path})")

    print("\n" + "=" * 60)
    print("PROTOTYPE PREPROCESSING & EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return {
        "dataset_name": dataset_name,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "vocab_size": len(vocab),
        "avg_src_len": avg_src_len,
        "max_src_len": max_src_len,
        "avg_tgt_len": avg_tgt_len,
        "max_tgt_len": max_tgt_len,
        "truncated_src_pct": truncated_src_pct,
        "truncated_tgt_pct": truncated_tgt_pct,
        "src_tensor_shape": list(sample_src_b.shape),
        "tgt_tensor_shape": list(sample_tgt_b.shape),
    }


if __name__ == "__main__":
    # Step 1: Run unit test validation on 20 examples
    run_unit_tests_on_small_subset()

    # Step 2: Run prototype EDA & dataset pipeline on 5000/500/500 CNN/DailyMail config
    results = run_prototype_eda_and_pipeline(
        dataset_name="cnn_dailymail",
        train_size=5000,
        val_size=500,
        test_size=500,
        vocab_size=20000,
        max_source_len=300,
        max_target_len=60,
        batch_size=16
    )
