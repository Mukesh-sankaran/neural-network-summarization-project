"""
Fine-tuning pipeline for t5-small on local CNN/DailyMail dataset.
Implements:
- Prefix "summarize: "
- max source length: 512, max target length: 128
- fp32 only
- AdamW optimizer (lr: 3e-4)
- Batch size: 2, Gradient accumulation: 8 (effective batch size: 16)
- Gradient clipping: 1.0
- 1 epoch
- Deterministic train/val/test split with seed 42 (2000 train, 200 val, 100 test)
- Progress logging every 20 steps to _reports/finetune_log.txt
- Model and tokenizer saved exclusively to models/t5_small_finetuned/
"""

import os
import sys
import time
import json
import argparse
import pyarrow.parquet as pq
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW

# Set random seed deterministically
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


class T5SummarizationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, prefix: str = "summarize: ", max_source_len: int = 512, max_target_len: int = 128):
        self.sources = [prefix + str(text) for text in df["article"]]
        self.targets = [str(text) for text in df["highlights"]]
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(
            self.sources[idx],
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tgt_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": labels
        }


def log_message(msg: str, log_file: str):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune t5-small on CNN/DailyMail")
    parser.add_argument("--data_file", type=str, default="datasets/cnn_dailymail/3.0.0/train-00000-of-00003.parquet")
    parser.add_argument("--output_dir", type=str, default="models/t5_small_finetuned")
    parser.add_argument("--log_file", type=str, default="_reports/finetune_log.txt")
    parser.add_argument("--test_file", type=str, default="_reports/test_set.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--prefix", type=str, default="summarize: ")
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize log file
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(f"=== T5-SMALL FINE-TUNING EXECUTION LOG ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {vars(args)}\n\n")

    set_seed(args.seed)

    # Select device: Prefer MPS, test allocation
    device = torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            t = torch.zeros(1, device="mps")
            del t
            device = torch.device("mps")
        except Exception:
            device = torch.device("cpu")

    log_message(f"Selected Compute Device: {device}", args.log_file)

    # Load dataset
    log_message(f"Reading dataset from {args.data_file}...", args.log_file)
    table = pq.read_table(args.data_file, columns=["article", "highlights"])
    df = table.to_pandas()
    df = df.dropna(subset=["article", "highlights"])
    df = df[df["article"].str.strip().str.len() > 0]
    df = df[df["highlights"].str.strip().str.len() > 0]

    total_samples = args.num_train + args.num_val + args.num_test
    if len(df) < total_samples:
        log_message(f"Dataset has {len(df)} samples, less than requested {total_samples}.", args.log_file)
        args.num_train = int(len(df) * 0.8)
        args.num_val = int(len(df) * 0.1)
        args.num_test = len(df) - args.num_train - args.num_val
        total_samples = len(df)

    # Deterministic split with no overlap
    log_message(f"Splitting dataset deterministically (seed={args.seed}): {args.num_train} train, {args.num_val} val, {args.num_test} test...", args.log_file)
    sampled_df = df.sample(n=total_samples, random_state=args.seed).reset_index(drop=True)
    train_df = sampled_df.iloc[:args.num_train].reset_index(drop=True)
    val_df = sampled_df.iloc[args.num_train:args.num_train + args.num_val].reset_index(drop=True)
    test_df = sampled_df.iloc[args.num_train + args.num_val:total_samples].reset_index(drop=True)

    # Save test set for reproducible evaluation
    test_records = []
    for idx, row in test_df.iterrows():
        test_records.append({
            "id": f"test_{idx}",
            "article": row["article"],
            "reference": row["highlights"]
        })
    with open(args.test_file, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)
    log_message(f"Saved {len(test_records)} held-out test examples to {args.test_file}", args.log_file)

    # Load tokenizer and model in fp32
    log_message("Loading t5-small tokenizer and model in fp32...", args.log_file)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = model.float()  # Strict fp32
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})", args.log_file)

    train_dataset = T5SummarizationDataset(train_df, tokenizer, prefix=args.prefix, max_source_len=args.max_source_len, max_target_len=args.max_target_len)
    val_dataset = T5SummarizationDataset(val_df, tokenizer, prefix=args.prefix, max_source_len=args.max_source_len, max_target_len=args.max_target_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    total_steps = len(train_loader)
    effective_updates = (total_steps + args.grad_accum - 1) // args.grad_accum
    log_message(f"Starting training: 1 Epoch | {total_steps} micro-batches | {effective_updates} optimizer updates", args.log_file)

    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    step_window_time = time.time()
    running_loss = 0.0
    accumulated_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            log_message(f"ERROR: NaN/Inf loss detected at step {step}! Loss: {loss.item()}", args.log_file)
            if device.type == "mps":
                log_message("Attempting CPU fallback due to numerical instability on MPS...", args.log_file)
                device = torch.device("cpu")
                model.to(device)
                outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
                loss = outputs.loss
            else:
                log_message("Aborting due to NaN/Inf on CPU.", args.log_file)
                sys.exit(1)

        # Scale loss for gradient accumulation
        scaled_loss = loss / args.grad_accum
        scaled_loss.backward()

        accumulated_loss += loss.item()
        running_loss += loss.item()

        # Optimizer step after grad_accum micro-batches or at end
        if step % args.grad_accum == 0 or step == total_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # Log progress every 20 steps
        if step % 20 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            window_duration = time.time() - step_window_time
            step_speed = window_duration / 20 if step % 20 == 0 else window_duration
            avg_loss_20 = running_loss / (20 if step % 20 == 0 else (step % 20))
            running_loss = 0.0
            step_window_time = time.time()

            est_total_time = (elapsed / step) * total_steps
            remaining_time = max(0.0, est_total_time - elapsed)

            msg = (
                f"Step {step:4d}/{total_steps} ({(step/total_steps)*100:5.1f}%) | "
                f"Batch Loss: {loss.item():.4f} | Avg Loss (last 20): {avg_loss_20:.4f} | "
                f"Speed: {step_speed:.3f}s/step | Elapsed: {elapsed/60:.1f}m | ETA: {remaining_time/60:.1f}m"
            )
            log_message(msg, args.log_file)

    train_duration = time.time() - start_time
    log_message(f"\nTraining completed in {train_duration/60:.2f} minutes ({train_duration:.1f}s).", args.log_file)

    # Validation
    log_message("\nRunning validation on held-out validation split...", args.log_file)
    model.eval()
    val_loss_total = 0.0
    val_steps = len(val_loader)

    with torch.no_grad():
        for val_batch in val_loader:
            v_input_ids = val_batch["input_ids"].to(device)
            v_attention_mask = val_batch["attention_mask"].to(device)
            v_labels = val_batch["labels"].to(device)

            v_outputs = model(input_ids=v_input_ids, attention_mask=v_attention_mask, labels=v_labels)
            val_loss_total += v_outputs.loss.item()

    avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
    log_message(f"Validation Loss: {avg_val_loss:.4f} over {len(val_df)} examples.", args.log_file)

    # Save model and tokenizer
    log_message(f"\nSaving model and tokenizer to {args.output_dir}...", args.log_file)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training summary metadata
    summary_meta = {
        "model_name": "t5-small",
        "fine_tuned_on": "CNN/DailyMail 3.0.0",
        "num_train_examples": args.num_train,
        "num_val_examples": args.num_val,
        "num_test_examples": args.num_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_source_len": args.max_source_len,
        "max_target_len": args.max_target_len,
        "training_duration_seconds": round(train_duration, 2),
        "avg_val_loss": round(avg_val_loss, 4),
        "device": str(device),
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    log_message(f"Fine-tuning pipeline finished successfully! Model ready in {args.output_dir}", args.log_file)


if __name__ == "__main__":
    main()
