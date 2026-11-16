"""
Reproducible Pretrained Model Fine-Tuning Pipeline.
Supports parameter-efficient / lightweight fine-tuning of sshleifer/distilbart-cnn-12-6
on local CNN/DailyMail dataset with gradient accumulation, evaluation checkpoints,
and safe memory management for Apple Silicon MPS and CPU environments.
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset_split
from src.pretrained_baseline import get_pretrained_device, DEFAULT_PRETRAINED_MODEL_NAME


class HFSummarizationDataset(Dataset):
    """Tokenizes article and summary text for Hugging Face Seq2Seq models."""
    def __init__(self, df, tokenizer, max_source_len=512, max_target_len=128):
        self.sources = df["source_text"].tolist()
        self.targets = df["target_summary"].tolist()
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = str(self.sources[idx])
        target = str(self.targets[idx])

        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze(0)
        # Replace padding token id's of the labels by -100 so CrossEntropy ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def run_finetuning_pilot(
    model_name: str = DEFAULT_PRETRAINED_MODEL_NAME,
    output_dir: str = "models/finetuned_distilbart",
    train_samples: int = 50,
    val_samples: int = 10,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 5e-5,
    max_steps: int = 20,
):
    print("=" * 70)
    print("PRETRAINED BASELINE FINE-TUNING PIPELINE")
    print(f"Model: {model_name} | Target Output: {output_dir}")
    print("=" * 70)

    device = get_pretrained_device()
    print(f"Active Compute Device: {device}")

    # Check RAM/hardware budget
    print(f"Batch size: {batch_size} | Gradient accumulation steps: {grad_accum_steps}")
    print(f"Training subset: {train_samples} | Validation subset: {val_samples} | Max steps: {max_steps}")

    print("\n[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    print("\n[2/4] Loading local dataset splits...")
    train_df = load_dataset_split("cnn_dailymail", "train", sample_size=train_samples)
    val_df = load_dataset_split("cnn_dailymail", "validation", sample_size=val_samples)

    train_dataset = HFSummarizationDataset(train_df, tokenizer, max_source_len=384, max_target_len=96)
    val_dataset = HFSummarizationDataset(val_df, tokenizer, max_source_len=384, max_target_len=96)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n[3/4] Executing Pilot Training (capped at {max_steps} optimization steps)...")
    model.train()
    step_count = 0
    total_loss = 0.0
    start_time = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                curr_loss = epoch_loss / (batch_idx + 1)
                print(f"  Step {step_count}/{max_steps} | Batch {batch_idx + 1} | Training Loss: {curr_loss:.4f}")

                if step_count >= max_steps:
                    break

        if step_count >= max_steps:
            break

    elapsed_s = time.perf_counter() - start_time
    print(f"\n[4/4] Pilot complete in {elapsed_s:.1f}s. Evaluating validation loss...")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            v_input_ids = val_batch["input_ids"].to(device)
            v_attention_mask = val_batch["attention_mask"].to(device)
            v_labels = val_batch["labels"].to(device)
            v_out = model(
                input_ids=v_input_ids,
                attention_mask=v_attention_mask,
                labels=v_labels,
            )
            val_loss += v_out.loss.item()
    val_loss /= max(1, len(val_loader))
    print(f"  Validation Loss: {val_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving pilot checkpoint artifacts to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    pilot_meta = {
        "base_model": model_name,
        "pilot_steps": step_count,
        "train_samples": train_samples,
        "final_train_loss": epoch_loss / max(1, step_count),
        "final_val_loss": val_loss,
        "elapsed_seconds": elapsed_s,
        "device": str(device),
        "status": "pilot_completed",
    }
    import json
    with open(os.path.join(output_dir, "pilot_metadata.json"), "w") as f:
        json.dump(pilot_meta, f, indent=2)

    print(f"Checkpoint successfully saved. Pilot metadata: {pilot_meta}")
    return pilot_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, help="Max pilot training steps")
    parser.add_argument("--samples", type=int, default=20, help="Train samples")
    args = parser.parse_args()
    run_finetuning_pilot(train_samples=args.samples, max_steps=args.steps)
