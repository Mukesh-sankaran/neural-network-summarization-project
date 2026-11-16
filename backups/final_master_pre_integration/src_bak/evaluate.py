"""
Complete Evaluation Module for Summarization.
Computes ROUGE-1, ROUGE-2, ROUGE-L metrics alongside Entity Preservation and Numeric Consistency.
Generates evaluation report JSON and plots for results presentation.
"""

import os
import sys
import json
from collections import Counter
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_summarization_dataset
from src.preprocess import tokenize, PAD_IDX
from src.vocabulary import Vocabulary
from src.model import Seq2Seq, get_device
from src.entity_evaluation import (
    evaluate_entity_preservation,
    evaluate_numeric_consistency,
    normalize_text
)


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Computes length of Longest Common Subsequence (LCS) between two token lists using dynamic programming.
    """
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_ngram_overlap(cand_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
    """
    Computes ROUGE-N (precision, recall, f1) for a given n-gram size.
    """
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_ngrams = [tuple(cand_tokens[i: i + n]) for i in range(len(cand_tokens) - n + 1)]
    ref_ngrams = [tuple(ref_tokens[i: i + n]) for i in range(len(ref_tokens) - n + 1)]

    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    overlap = 0
    for ngram, count in cand_counts.items():
        overlap += min(count, ref_counts.get(ngram, 0))

    precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_rouge_l(cand_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE-L (LCS-based precision, recall, f1).
    """
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = compute_lcs_length(cand_tokens, ref_tokens)

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def normalize_rouge_tokens(tokens: List[str]) -> List[str]:
    """
    Normalizes a token list for ROUGE evaluation by:
    1. Excluding standalone punctuation tokens (e.g. '.', ',', ''', '"', '-', '!', '?', etc.)
    2. Preserving alphanumeric words, numbers (100, 3.14), dates (2026-08-21), currencies ($100), and percentages (15.5%).
    3. Normalizing case.
    
    This ensures degenerate repetitions of punctuation tokens do not artificially inflate
    ROUGE-1 and ROUGE-L overlap scores against reference punctuation.
    """
    normalized = []
    for tok in tokens:
        if not tok:
            continue
        # Retain token if it contains at least one alphanumeric character
        if any(c.isalnum() for c in tok):
            normalized.append(tok.lower())
    return normalized


def evaluate_summary(
    cand_text: str,
    ref_text: str,
    exclude_punctuation: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L for a single candidate summary vs reference.
    
    Args:
        cand_text: Generated candidate summary text.
        ref_text: Human reference summary text.
        exclude_punctuation: When True (default), filters out standalone punctuation tokens
                             so punctuation does not artificially inflate ROUGE scores.
    """
    cand_tokens = tokenize(cand_text)
    ref_tokens = tokenize(ref_text)

    if exclude_punctuation:
        cand_tokens = normalize_rouge_tokens(cand_tokens)
        ref_tokens = normalize_rouge_tokens(ref_tokens)

    r1 = compute_ngram_overlap(cand_tokens, ref_tokens, n=1)
    r2 = compute_ngram_overlap(cand_tokens, ref_tokens, n=2)
    rl = compute_rouge_l(cand_tokens, ref_tokens)

    return {
        "rouge1": r1,
        "rouge2": r2,
        "rougel": rl
    }


def run_full_evaluation(test_num_examples: int = 20):
    print("=" * 60)
    print(f"RUNNING PROJECT EVALUATION SUITE ({test_num_examples} TEST EXAMPLES)")
    print("=" * 60)

    device = get_device()
    print(f"Device Selected: {device}")

    # 1. Load Data Subsets (CNN/DailyMail)
    print("\nLoading test dataset...")
    _, _, test_df = load_summarization_dataset(
        dataset_name="cnn_dailymail",
        train_size=10,
        val_size=10,
        test_size=test_num_examples,
        seed=42
    )
    print(f"Test Examples Loaded: {len(test_df)}")

    # 2. Load Vocabulary
    vocab_file = "models/vocab.pkl"
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}. Please run training first.")

    vocab = Vocabulary.load(vocab_file)
    print(f"Vocabulary Loaded: {len(vocab)} tokens")

    # 3. Load Trained Model Checkpoint
    checkpoint_file = "models/best_model.pt"
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Model checkpoint file not found at {checkpoint_file}. Please run training first.")

    print(f"Loading trained model checkpoint from {checkpoint_file}...")
    model = Seq2Seq(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Process Each Test Example
    per_example_results = []
    
    # Aggregators for overall averages
    r1_f1_list, r1_p_list, r1_r_list = [], [], []
    r2_f1_list, r2_p_list, r2_r_list = [], [], []
    rl_f1_list, rl_p_list, rl_r_list = [], [], []
    
    ent_prec_list, ent_rec_list, ent_pres_list = [], [], []
    num_cons_list = []

    print("\nGenerating summaries and computing metrics for test examples...")

    max_source_len = 500
    max_target_len = 60

    for idx, row in test_df.iterrows():
        doc_id = row["id"]
        source_text = row["source_text"]
        ref_summary = row["target_summary"]

        # Prepare source tensor
        src_tokens = tokenize(source_text)[:max_source_len]
        src_indices = vocab.encode(src_tokens)
        if len(src_indices) < max_source_len:
            src_indices = src_indices + [PAD_IDX] * (max_source_len - len(src_indices))
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        # Generate summary via trained Seq2Seq model
        gen_summary, _, _ = model.generate_summary(
            src_tensor, vocab, max_target_len=max_target_len, device=device
        )

        # 1. ROUGE Evaluation
        rouge_scores = evaluate_summary(gen_summary, ref_summary)
        
        # 2. Entity Preservation Evaluation
        entity_scores = evaluate_entity_preservation(source_text, gen_summary)
        
        # 3. Numeric / Factual Consistency Evaluation
        numeric_scores = evaluate_numeric_consistency(source_text, gen_summary)

        # Store metric scores for aggregation
        r1_f1_list.append(rouge_scores["rouge1"]["f1"])
        r1_p_list.append(rouge_scores["rouge1"]["precision"])
        r1_r_list.append(rouge_scores["rouge1"]["recall"])

        r2_f1_list.append(rouge_scores["rouge2"]["f1"])
        r2_p_list.append(rouge_scores["rouge2"]["precision"])
        r2_r_list.append(rouge_scores["rouge2"]["recall"])

        rl_f1_list.append(rouge_scores["rougel"]["f1"])
        rl_p_list.append(rouge_scores["rougel"]["precision"])
        rl_r_list.append(rouge_scores["rougel"]["recall"])

        ent_prec_list.append(entity_scores["entity_precision"])
        ent_rec_list.append(entity_scores["entity_recall"])
        ent_pres_list.append(entity_scores["entity_preservation_rate"])

        num_cons_list.append(numeric_scores["numeric_consistency_rate"])

        record = {
            "example_index": idx + 1,
            "id": doc_id,
            "source_snippet": source_text[:180] + "...",
            "reference_summary": ref_summary,
            "generated_summary": gen_summary,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougel": rouge_scores["rougel"],
            "entity_metrics": entity_scores,
            "numeric_metrics": numeric_scores
        }
        per_example_results.append(record)

    # 5. Compute Overall Aggregate Averages (filtering out undefined None metrics)
    valid_ent_prec = [x for x in ent_prec_list if x is not None]
    valid_ent_rec = [x for x in ent_rec_list if x is not None]
    valid_ent_pres = [x for x in ent_pres_list if x is not None]
    valid_num_cons = [x for x in num_cons_list if x is not None]

    overall_averages = {
        "rouge1_f1": float(np.mean(r1_f1_list)) if r1_f1_list else 0.0,
        "rouge1_precision": float(np.mean(r1_p_list)) if r1_p_list else 0.0,
        "rouge1_recall": float(np.mean(r1_r_list)) if r1_r_list else 0.0,
        
        "rouge2_f1": float(np.mean(r2_f1_list)) if r2_f1_list else 0.0,
        "rouge2_precision": float(np.mean(r2_p_list)) if r2_p_list else 0.0,
        "rouge2_recall": float(np.mean(r2_r_list)) if r2_r_list else 0.0,
        
        "rougel_f1": float(np.mean(rl_f1_list)) if rl_f1_list else 0.0,
        "rougel_precision": float(np.mean(rl_p_list)) if rl_p_list else 0.0,
        "rougel_recall": float(np.mean(rl_r_list)) if rl_r_list else 0.0,
        
        "entity_precision": float(np.mean(valid_ent_prec)) if valid_ent_prec else None,
        "entity_recall": float(np.mean(valid_ent_rec)) if valid_ent_rec else None,
        "entity_preservation_rate": float(np.mean(valid_ent_pres)) if valid_ent_pres else None,
        "numeric_consistency_rate": float(np.mean(valid_num_cons)) if valid_num_cons else None
    }

    # Print Summary Table
    print("\n" + "=" * 60)
    print(f"OVERALL EVALUATION RESULTS ACROSS {test_num_examples} TEST EXAMPLES")
    print("=" * 60)
    print(f"  ROUGE-1 F1:                 {overall_averages['rouge1_f1']:.4f} (Prec: {overall_averages['rouge1_precision']:.4f}, Rec: {overall_averages['rouge1_recall']:.4f})")
    print(f"  ROUGE-2 F1:                 {overall_averages['rouge2_f1']:.4f} (Prec: {overall_averages['rouge2_precision']:.4f}, Rec: {overall_averages['rouge2_recall']:.4f})")
    print(f"  ROUGE-L F1:                 {overall_averages['rougel_f1']:.4f} (Prec: {overall_averages['rougel_precision']:.4f}, Rec: {overall_averages['rougel_recall']:.4f})")
    
    ent_p_str = f"{overall_averages['entity_precision']:.4f}" if overall_averages['entity_precision'] is not None else "N/A"
    ent_r_str = f"{overall_averages['entity_recall']:.4f}" if overall_averages['entity_recall'] is not None else "N/A"
    ent_pres_str = f"{overall_averages['entity_preservation_rate']:.4f}" if overall_averages['entity_preservation_rate'] is not None else "N/A"
    num_cons_str = f"{overall_averages['numeric_consistency_rate']:.4f}" if overall_averages['numeric_consistency_rate'] is not None else "N/A"
    
    print(f"  Entity Precision:           {ent_p_str}")
    print(f"  Entity Recall:              {ent_r_str}")
    print(f"  Entity Preservation Rate:   {ent_pres_str}")
    print(f"  Numeric Consistency Rate:   {num_cons_str}")
    print("=" * 60)

    # 6. Save JSON Evaluation Report
    os.makedirs("results", exist_ok=True)
    report_data = {
        "num_test_examples": test_num_examples,
        "overall_averages": overall_averages,
        "per_example_results": per_example_results
    }
    json_path = "results/evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\nSaved evaluation report JSON to [evaluation_report.json]({json_path})")

    # 7. Generate Visualizations
    
    # Plot 1: ROUGE Scores
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    f1_vals = [overall_averages["rouge1_f1"], overall_averages["rouge2_f1"], overall_averages["rougel_f1"]]
    p_vals = [overall_averages["rouge1_precision"], overall_averages["rouge2_precision"], overall_averages["rougel_precision"]]
    r_vals = [overall_averages["rouge1_recall"], overall_averages["rouge2_recall"], overall_averages["rougel_recall"]]

    x = np.arange(len(metrics))
    width = 0.25

    ax.bar(x - width, f1_vals, width, label="F1-Score", color="#2b5c8f")
    ax.bar(x, p_vals, width, label="Precision", color="#4682b4")
    ax.bar(x + width, r_vals, width, label="Recall", color="#87ceeb")

    ax.set_ylabel("Score")
    ax.set_title("ROUGE Evaluation Metrics (20 Test Examples)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_rouge_path = "results/rouge_scores.png"
    plt.savefig(plot_rouge_path, dpi=300)
    plt.close()
    print(f"Saved ROUGE plot to [rouge_scores.png]({plot_rouge_path})")

    # Plot 2: Entity Preservation Metrics
    fig, ax = plt.subplots(figsize=(7, 5))
    ent_labels = ["Entity Precision", "Entity Recall", "Preservation Rate"]
    ent_vals = [overall_averages["entity_precision"], overall_averages["entity_recall"], overall_averages["entity_preservation_rate"]]
    colors = ["#2e8b57", "#3cb371", "#66cdaa"]

    bars = ax.bar(ent_labels, ent_vals, color=colors, width=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Entity Preservation & Precision Metrics")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    plot_ent_path = "results/entity_preservation.png"
    plt.savefig(plot_ent_path, dpi=300)
    plt.close()
    print(f"Saved Entity Preservation plot to [entity_preservation.png]({plot_ent_path})")

    # Plot 3: Factual / Numeric Consistency
    fig, ax = plt.subplots(figsize=(6, 5))
    num_labels = ["Numeric Consistency"]
    num_vals = [overall_averages["numeric_consistency_rate"]]

    bars = ax.bar(num_labels, num_vals, color="#db7093", width=0.4)
    ax.set_ylabel("Consistency Rate")
    ax.set_title("Factual / Numeric Consistency Rate")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    plot_fact_path = "results/factual_consistency.png"
    plt.savefig(plot_fact_path, dpi=300)
    plt.close()
    print(f"Saved Factual Consistency plot to [factual_consistency.png]({plot_fact_path})")

    return report_data


if __name__ == "__main__":
    run_full_evaluation(test_num_examples=20)
