"""
Training script for the Second Real Training Experiment on CNN/DailyMail dataset.
Trains the Seq2Seq model with EOS/Alignment fixes, records loss metrics across 3 epochs,
saves best model checkpoint, plots loss curves, and performs generation diagnostics on 10 test examples.
"""

import os
import sys
import time
import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_summarization_dataset
from src.preprocess import tokenize, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from src.vocabulary import Vocabulary
from src.dataset import SummarizationDataset, get_dataloaders
from src.model import Seq2Seq, get_device


def verify_pre_training_alignment(dataset: SummarizationDataset, vocab: Vocabulary):
    """
    Pre-training verification of target sequence alignment and EOS placement.
    """
    print("--- PRE-TRAINING VERIFICATION ---")

    # 1. Verify 100% of target sequences contain EOS
    eos_found_count = 0
    for i in range(len(dataset)):
        _, tgt = dataset[i]
        if vocab.eos_idx in tgt.tolist():
            eos_found_count += 1

    eos_pct = (eos_found_count / len(dataset)) * 100
    assert eos_pct == 100.0, f"Verification failed! Only {eos_pct:.2f}% of target sequences contain EOS!"
    print(f"  [VERIFIED] 100% of target sequences contain <EOS> ({eos_found_count}/{len(dataset)}).")

    # 2. Verify decoder input vs target slice shapes
    sample_src, sample_tgt = dataset[0]
    sample_tgt = sample_tgt.unsqueeze(0)  # Shape [1, 60]
    dec_in = sample_tgt[:, :-1]
    pred_tgt = sample_tgt[:, 1:]

    assert dec_in.shape == (1, 59), f"Decoder input shape mismatch: expected (1, 59), got {dec_in.shape}"
    assert pred_tgt.shape == (1, 59), f"Prediction target shape mismatch: expected (1, 59), got {pred_tgt.shape}"
    print(f"  [VERIFIED] Decoder input shape: {dec_in.shape} (target[:, :-1]).")
    print(f"  [VERIFIED] Prediction target shape: {pred_tgt.shape} (target[:, 1:]).")
    print("  [VERIFIED] Pre-training alignment checks completed successfully.\n")


def train_experiment():
    print("=" * 60)
    print("STARTING SECOND REAL TRAINING EXPERIMENT (CNN/DAILYMAIL)")
    print("=" * 60)

    # 1. Hyperparameters & Configuration
    dataset_name = "cnn_dailymail"
    train_size = 2000
    val_size = 300
    test_size = 100
    epochs = 3
    batch_size = 8
    max_source_len = 500
    max_target_len = 60
    vocab_size = 20000
    embedding_dim = 128
    encoder_hidden_dim = 256
    decoder_hidden_dim = 256
    teacher_forcing_ratio = 0.5
    learning_rate = 0.001
    max_grad_norm = 1.0

    # 2. Select Device
    device = get_device()
    print(f"Device Selected for Training: {device}")

    # 3. Load Data Subsets (CNN/DailyMail)
    print("\nLoading dataset subsets from local Parquet files...")
    train_df, val_df, test_df = load_summarization_dataset(
        dataset_name=dataset_name,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=42
    )
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Build or Load Vocabulary (from training set only)
    vocab_file = "models/vocab.pkl"
    print("Building vocabulary from training data (source + target tokens)...")
    tokenized_train_src = [tokenize(text) for text in train_df["source_text"]]
    tokenized_train_tgt = [tokenize(text) for text in train_df["target_summary"]]
    vocab = Vocabulary(max_size=vocab_size)
    vocab.build_vocab(tokenized_train_src + tokenized_train_tgt)
    vocab.save(vocab_file)
    print(f"Vocabulary Size: {len(vocab)} (saved to {vocab_file})")

    # 5. Pre-training alignment verification
    train_dataset_obj = SummarizationDataset(train_df, vocab, max_source_len=max_source_len, max_target_len=max_target_len)
    verify_pre_training_alignment(train_dataset_obj, vocab)

    # 6. Build PyTorch DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, vocab,
        batch_size=batch_size,
        max_source_len=max_source_len,
        max_target_len=max_target_len
    )
    print(f"DataLoaders Ready: {len(train_loader)} train batches, {len(val_loader)} val batches.")

    # 7. Initialize Model, Criterion, and Optimizer
    print("\nInitializing Seq2Seq Model...")
    model = Seq2Seq(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history = {"epochs": [], "train_loss": [], "val_loss": []}
    model_checkpoint_saved = False

    start_time = time.time()

    # 8. Training Loop (3 Epochs)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for batch_idx, (src, tgt) in enumerate(train_loader, 1):
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            # Forward pass: logits shape [batch_size, target_len - 1, vocab_size]
            logits = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

            # Compute loss against tgt[:, 1:]
            loss = model.compute_loss(logits, tgt, criterion)

            # Backward pass & gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                logits = model(src, tgt, teacher_forcing_ratio=0.0)
                loss = model.compute_loss(logits, tgt, criterion)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.2f}s) - "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Save Best Model Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            checkpoint_path = "models/best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "vocab_size": len(vocab)
            }, checkpoint_path)
            model_checkpoint_saved = True
            print(f"  --> Saved new best model checkpoint to [best_model.pt]({checkpoint_path}) (Val Loss: {best_val_loss:.4f})")

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete! Total Training Time: {total_training_time:.2f} seconds ({total_training_time / 60:.2f} mins).")

    # 9. Save History & Plot Loss Curves
    os.makedirs("results", exist_ok=True)
    history_path = "results/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to [training_history.json]({history_path})")

    plot_path = "results/training_validation_loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history["epochs"], history["train_loss"], marker="o", label="Training Loss", color="blue")
    plt.plot(history["epochs"], history["val_loss"], marker="s", label="Validation Loss", color="orange")
    plt.title("CNN/DailyMail Seq2Seq Training & Validation Loss (3 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.xticks(history["epochs"])
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to [training_validation_loss.png]({plot_path})")

    # 10. Post-Training Inference & Quantitative Diagnostics on 10 Test Examples
    print("\n" + "=" * 60)
    print("RUNNING POST-TRAINING INFERENCE & DIAGNOSTICS ON 10 TEST EXAMPLES")
    print("=" * 60)

    if os.path.exists("models/best_model.pt"):
        checkpoint = torch.load("models/best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    
    sample_test_df = test_df.head(10).reset_index(drop=True)
    generated_summaries_list = []
    
    all_generated_tokens = []
    eos_generated_count = 0
    summary_lengths = []

    for idx, row in sample_test_df.iterrows():
        source_text = row["source_text"]
        ref_summary = row["target_summary"]

        src_tokens = tokenize(source_text)[:max_source_len]
        src_indices = vocab.encode(src_tokens)
        if len(src_indices) < max_source_len:
            src_indices = src_indices + [PAD_IDX] * (max_source_len - len(src_indices))

        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        # Generate summary
        gen_summary, gen_token_ids, _ = model.generate_summary(
            src_tensor, vocab, max_target_len=max_target_len, device=device
        )

        summary_lengths.append(len(gen_token_ids))
        all_generated_tokens.extend(gen_token_ids)

        # Check if generation stopped before reaching maximum target limit (indicating EOS emission)
        if len(gen_token_ids) < max_target_len - 1:
            eos_generated_count += 1

        generated_summaries_list.append({
            "idx": idx + 1,
            "id": row["id"],
            "source_snippet": source_text[:180] + "...",
            "reference_summary": ref_summary,
            "generated_summary": gen_summary
        })

        print(f"\n--- TEST EXAMPLE #{idx + 1} (ID: {row['id']}) ---")
        print(f"SOURCE (snippet): {source_text[:180]}...")
        print(f"REFERENCE SUMMARY: {ref_summary}")
        print(f"GENERATED SUMMARY: {gen_summary if gen_summary else '<EMPTY>'}")

    # Compute Generation Metrics across the 10 Test Examples
    eos_gen_pct = (eos_generated_count / 10) * 100
    avg_gen_len = float(sum(summary_lengths) / len(summary_lengths)) if summary_lengths else 0.0

    total_gen_tokens_count = len(all_generated_tokens)
    
    # Token frequency calculation
    token_counter = Counter(all_generated_tokens)
    most_frequent_token_id, most_frequent_count = token_counter.most_common(1)[0] if token_counter else (0, 0)
    most_frequent_token_word = vocab.idx2word.get(most_frequent_token_id, "<UNKNOWN>")

    # Period token '.' count & %
    period_idx = vocab.word2idx.get(".", -1)
    period_count = all_generated_tokens.count(period_idx) if period_idx != -1 else 0
    period_pct = (period_count / total_gen_tokens_count * 100) if total_gen_tokens_count > 0 else 0.0

    # UNK token count & %
    unk_count = all_generated_tokens.count(UNK_IDX)
    unk_pct = (unk_count / total_gen_tokens_count * 100) if total_gen_tokens_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("QUANTITATIVE GENERATION DIAGNOSTICS (10 TEST EXAMPLES)")
    print("=" * 60)
    print(f"  1. EOS Generation Percentage:            {eos_gen_pct:.2f}% ({eos_generated_count}/10)")
    print(f"  2. Average Generated Summary Length:     {avg_gen_len:.2f} tokens")
    print(f"  3. Most Frequent Generated Token:        '{most_frequent_token_word}' (Count: {most_frequent_count})")
    print(f"  4. Period Token ('.') Percentage:        {period_pct:.2f}% ({period_count}/{total_gen_tokens_count})")
    print(f"  5. UNK Token ('<UNK>') Percentage:       {unk_pct:.2f}% ({unk_count}/{total_gen_tokens_count})")
    print("=" * 60)

    return {
        "epochs": history["epochs"],
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "total_training_time": total_training_time,
        "device": str(device),
        "eos_gen_pct": eos_gen_pct,
        "avg_gen_len": avg_gen_len,
        "most_frequent_token": most_frequent_token_word,
        "period_pct": period_pct,
        "unk_pct": unk_pct,
        "summaries": generated_summaries_list
    }


if __name__ == "__main__":
    train_experiment()
