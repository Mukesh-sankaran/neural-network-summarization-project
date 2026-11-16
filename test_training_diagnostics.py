"""
Diagnostic script to inspect target preprocessing, token frequencies,
<EOS> placement, truncation rates, and decoder input/output alignment.
"""

import sys
import os
from collections import Counter

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import load_summarization_dataset
from src.preprocess import tokenize, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN
from src.vocabulary import Vocabulary
from src.dataset import SummarizationDataset, get_dataloaders


def run_diagnostics():
    print("=" * 60)
    print("DIAGNOSTIC INSPECTION: TARGET DATA & DECODER ALIGNMENT")
    print("=" * 60)

    # 1. Load dataset subset (CNN/DailyMail)
    print("Loading CNN/DailyMail train (500), val (100), test (100)...")
    train_df, val_df, test_df = load_summarization_dataset(
        dataset_name="cnn_dailymail",
        train_size=500,
        val_size=100,
        test_size=100,
        seed=42
    )

    # Tokenize target summaries
    target_token_lists = [tokenize(text) for text in train_df["target_summary"]]
    
    # Target Token Frequencies
    target_counter = Counter()
    for tokens in target_token_lists:
        target_counter.update(tokens)

    print("\n--- 1. TOP 20 TARGET VOCABULARY TOKENS & FREQUENCIES ---")
    for token, freq in target_counter.most_common(20):
        print(f"  '{token}': {freq}")

    # Build Vocabulary
    vocab = Vocabulary(max_size=20000)
    vocab.build_vocab([tokenize(t) for t in train_df["source_text"]] + target_token_lists)

    # Create SummarizationDataset with max_target_len = 60
    dataset = SummarizationDataset(train_df, vocab, max_source_len=500, max_target_len=60)

    # Analyze Dataset Tensors
    total_tokens = 0
    pad_count = 0
    eos_count_in_seqs = 0
    truncated_count = 0

    for i in range(len(dataset)):
        _, tgt_tensor = dataset[i]
        tgt_list = tgt_tensor.tolist()
        
        total_tokens += len(tgt_list)
        pad_count += tgt_list.count(vocab.pad_idx)
        
        if vocab.eos_idx in tgt_list:
            eos_count_in_seqs += 1
            
        # Check if original raw token length + 2 (<SOS> and <EOS>) > 60
        raw_token_len = len(target_token_lists[i])
        if raw_token_len + 2 > 60:
            truncated_count += 1

    pct_pad = (pad_count / total_tokens) * 100
    pct_eos_in_seqs = (eos_count_in_seqs / len(dataset)) * 100
    pct_truncated = (truncated_count / len(dataset)) * 100

    print("\n--- 2. TARGET DATASET STATISTICS ---")
    print(f"  Total target tokens (including <PAD>): {total_tokens}")
    print(f"  Percentage of target tokens that are <PAD>: {pct_pad:.2f}%")
    print(f"  Percentage of target sequences containing <EOS>: {pct_eos_in_seqs:.2f}%")
    print(f"  Percentage of target sequences truncated (>60 tokens): {pct_truncated:.2f}%")

    print("\n--- 3. FIRST 3 ENCODED & DECODED TARGET SEQUENCES ---")
    for i in range(3):
        _, tgt_tensor = dataset[i]
        tgt_list = tgt_tensor.tolist()
        
        print(f"\n[Example {i+1}] Raw Target Summary:")
        print(f"  '{train_df.iloc[i]['target_summary']}'")
        print(f"  Encoded Indices (first 25): {tgt_list[:25]}")
        print(f"  Has <EOS> (id={vocab.eos_idx}): {vocab.eos_idx in tgt_list}")
        print(f"  Decoded text (with special tokens):")
        print(f"  '{vocab.decode(tgt_list, skip_special_tokens=False)}'")

    print("\n--- 4. DECODER INPUT & EXPECTED PREDICTION ALIGNMENT ---")
    _, tgt_tensor = dataset[0]
    tgt_list = tgt_tensor.tolist()

    # In Seq2Seq:
    # Decoder input is tgt[:-1] (from <SOS> up to second to last token)
    # Expected prediction is tgt[1:] (from first content word up to <EOS>/<PAD>)
    dec_input = tgt_list[:-1]
    expected_pred = tgt_list[1:]

    print(f"Full Target Tensor Length: {len(tgt_list)}")
    print(f"Decoder Input Sequence (len={len(dec_input)}):")
    print(f"  Indices: {dec_input[:10]} ... {dec_input[-5:]}")
    print(f"  Decoded: '{vocab.decode(dec_input, skip_special_tokens=False)[:80]}...'")

    print(f"Expected Prediction Sequence (len={len(expected_pred)}):")
    print(f"  Indices: {expected_pred[:10]} ... {expected_pred[-5:]}")
    print(f"  Decoded: '{vocab.decode(expected_pred, skip_special_tokens=False)[:80]}...'")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
