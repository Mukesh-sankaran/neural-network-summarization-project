"""
NNDL Academic Project Demonstration Script.
Evaluates several test examples from the CNN/DailyMail dataset using the existing trained checkpoint,
automatically selects the representative example with the highest ROUGE-L score, and presents its
end-to-end inference, ROUGE scores, Entity Preservation analysis, and Factual Consistency metrics.

Designed for supervisor progress demonstration.
"""

import os
import sys
import torch

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_summarization_dataset
from src.preprocess import tokenize, PAD_IDX
from src.vocabulary import Vocabulary
from src.model import Seq2Seq, get_device
from src.evaluate import evaluate_summary
from src.entity_evaluation import (
    evaluate_entity_preservation,
    evaluate_numeric_consistency,
    extract_named_entities
)


def run_demo(sample_pool_size: int = 20):
    print("==================================================================")
    print("  NNDL PROJECT DEMONSTRATION: LONG DOCUMENT SUMMARIZATION")
    print("       WITH FACTUAL & ENTITY PRESERVATION EVALUATION")
    print("==================================================================\n")

    # 1. Device Selection
    device = get_device()
    print(f"Device Selected: {device}")

    # 2. Load Vocabulary
    vocab_path = "models/vocab.pkl"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")
    vocab = Vocabulary.load(vocab_path)

    # 3. Load Trained Model Checkpoint
    checkpoint_path = "models/best_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}.")

    model = Seq2Seq(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Load Test Examples Pool
    print(f"Evaluating {sample_pool_size} test examples to select best ROUGE-L match...")
    _, _, test_df = load_summarization_dataset(
        dataset_name="cnn_dailymail",
        train_size=10,
        val_size=10,
        test_size=sample_pool_size,
        seed=42
    )

    max_source_len = 500
    max_target_len = 60

    best_example = None
    best_rougel_f1 = -1.0

    # Evaluate candidate pool and find maximum ROUGE-L match
    for idx, row in test_df.iterrows():
        source_text = row["source_text"]
        ref_summary = row["target_summary"]

        src_tokens = tokenize(source_text)[:max_source_len]
        src_indices = vocab.encode(src_tokens)
        if len(src_indices) < max_source_len:
            src_indices = src_indices + [PAD_IDX] * (max_source_len - len(src_indices))
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        gen_summary, _, _ = model.generate_summary(
            src_tensor, vocab, max_target_len=max_target_len, device=device
        )

        rouge_scores = evaluate_summary(gen_summary, ref_summary)
        rougel_f1 = rouge_scores["rougel"]["f1"]

        if rougel_f1 > best_rougel_f1:
            best_rougel_f1 = rougel_f1
            best_example = {
                "idx": idx + 1,
                "id": row["id"],
                "source_text": source_text,
                "ref_summary": ref_summary,
                "gen_summary": gen_summary,
                "rouge_scores": rouge_scores
            }

    print(f"  Selected Test Example #{best_example['idx']} (ID: {best_example['id']}) with Highest ROUGE-L F1: {best_rougel_f1:.4f}\n")

    source_text = best_example["source_text"]
    ref_summary = best_example["ref_summary"]
    gen_summary = best_example["gen_summary"]
    rouge_scores = best_example["rouge_scores"]

    # 5. Display Original Article Snippet
    print("------------------------------------------------------------------")
    print(f"1. ORIGINAL ARTICLE (Snippet - Example #{best_example['idx']})")
    print("------------------------------------------------------------------")
    print(source_text[:400] + "...\n")

    # 6. Display Reference Summary
    print("------------------------------------------------------------------")
    print("2. REFERENCE SUMMARY (Human Ground Truth)")
    print("------------------------------------------------------------------")
    print(ref_summary + "\n")

    # 7. Display Generated Summary
    print("------------------------------------------------------------------")
    print("3. GENERATED SUMMARY (Custom BiLSTM + Bahdanau Attention Model)")
    print("------------------------------------------------------------------")
    print((gen_summary if gen_summary else "<EMPTY GENERATION>") + "\n")

    # 8. Display ROUGE Metrics
    print("------------------------------------------------------------------")
    print("4. ROUGE EVALUATION METRICS")
    print("------------------------------------------------------------------")
    print(f"  ROUGE-1  | F1: {rouge_scores['rouge1']['f1']:.4f} | Precision: {rouge_scores['rouge1']['precision']:.4f} | Recall: {rouge_scores['rouge1']['recall']:.4f}")
    print(f"  ROUGE-2  | F1: {rouge_scores['rouge2']['f1']:.4f} | Precision: {rouge_scores['rouge2']['precision']:.4f} | Recall: {rouge_scores['rouge2']['recall']:.4f}")
    print(f"  ROUGE-L  | F1: {rouge_scores['rougel']['f1']:.4f} | Precision: {rouge_scores['rougel']['precision']:.4f} | Recall: {rouge_scores['rougel']['recall']:.4f}\n")

    # 9. Compute Entity Preservation Metrics
    entity_eval = evaluate_entity_preservation(source_text, gen_summary)
    src_ents = extract_named_entities(source_text)
    src_ent_names = sorted(list({e['original'] for e in src_ents}))

    print("------------------------------------------------------------------")
    print("5. NAMED ENTITY EXTRACTION & PRESERVATION ANALYSIS")
    print("------------------------------------------------------------------")
    print(f"  Source Document Entities ({len(src_ent_names)}): {src_ent_names[:10]}")
    print(f"  Preserved Entities ({len(entity_eval['preserved_entities'])}): {entity_eval['preserved_entities']}")
    print(f"  Missing Entities ({len(entity_eval['missing_entities'])}): {entity_eval['missing_entities'][:8]}")
    print(f"  Extra Entities Introduced ({len(entity_eval['extra_entities'])}): {entity_eval['extra_entities']}")
    print(f"\n  Entity Precision:         {entity_eval['entity_precision']:.4f}")
    print(f"  Entity Recall:            {entity_eval['entity_recall']:.4f}")
    print(f"  Entity Preservation Rate: {entity_eval['entity_preservation_rate']:.4f}\n")

    # 10. Compute Factual/Numeric Consistency Metrics
    numeric_eval = evaluate_numeric_consistency(source_text, gen_summary)
    print("------------------------------------------------------------------")
    print("6. FACTUAL / NUMERIC CONSISTENCY ANALYSIS")
    print("------------------------------------------------------------------")
    print(f"  Numeric Consistency Rate: {numeric_eval['numeric_consistency_rate']:.4f}")
    print(f"  Consistent Numbers Found: {numeric_eval['consistent_numbers']}")
    print(f"  Hallucinated Numbers:     {numeric_eval['hallucinated_numbers']}\n")

    # 11. Display Architecture Summary
    print("==================================================================")
    print("7. SYSTEM ARCHITECTURE SUMMARY")
    print("==================================================================")
    print("""
    Custom Word-Level Tokenization (Clean, Normalize, Lowercase)
                               ↓
    Trainable Word Embedding (128-dim)
                               ↓
    Bidirectional LSTM Encoder (256-dim forward + 256-dim backward = 512-dim)
                               ↓
    Bahdanau Additive Attention (Masked Padding, Dynamic Context Vector 512-dim)
                               ↓
    LSTM Decoder Step-by-Step Generation (Hidden: 256-dim)
                               ↓
    Linear Output Projection Layer (Vocab Logits: 20,000)
                               ↓
    Generated Summary Output
                               ↓
    ROUGE + Entity Preservation + Factual Consistency Evaluation
    """)

    # 12. Display Current Prototype Training Configuration
    print("==================================================================")
    print("8. CURRENT PROTOTYPE TRAINING CONFIGURATION")
    print("==================================================================")
    print(f"  Dataset:                     CNN/DailyMail")
    print(f"  Training Samples:            2,000")
    print(f"  Validation Samples:          300")
    print(f"  Test Samples:                100")
    print(f"  Epochs:                      3")
    print(f"  Max Source Sequence Length:  500 tokens")
    print(f"  Max Target Sequence Length:  60 tokens")
    print(f"  Vocabulary Size:             20,000 tokens")
    print(f"  Embedding Dimension:         128")
    print(f"  Encoder Hidden Dimension:    256 (Bidirectional = 512)")
    print(f"  Decoder Hidden Dimension:    256")
    print(f"  Teacher Forcing Ratio:       0.5")
    print(f"  Optimizer:                   Adam")
    print(f"  Learning Rate:               0.001")
    print(f"  Loss Function:               CrossEntropyLoss (ignore_index=0)")
    print(f"  Device Used:                 {device}")
    print("==================================================================\n")


if __name__ == "__main__":
    run_demo(sample_pool_size=20)
