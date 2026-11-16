"""
PyTorch Dataset and DataLoader builder for Summarization tasks.
Handles tokenization, sequence truncation/padding, and conversion to torch.long tensors.
Ensures every target sequence properly includes <SOS> at start and <EOS> at end.
"""

from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocess import tokenize
from src.vocabulary import Vocabulary


class SummarizationDataset(Dataset):
    """
    PyTorch Dataset for text summarization.
    Preprocesses source texts and target summaries into PyTorch long tensors.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_source_len: int = 500,
        max_target_len: int = 60
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        source_text = row["source_text"]
        target_summary = row["target_summary"]

        # 1. Tokenize and encode source text
        src_tokens = tokenize(source_text)
        if len(src_tokens) > self.max_source_len:
            src_tokens = src_tokens[: self.max_source_len]

        src_indices = self.vocab.encode(src_tokens)
        if len(src_indices) < self.max_source_len:
            src_indices = src_indices + [self.vocab.pad_idx] * (self.max_source_len - len(src_indices))

        # 2. Tokenize and encode target summary with <SOS> and <EOS>
        tgt_tokens = tokenize(target_summary)
        
        # Truncate content tokens first to leave space for <SOS> and <EOS>
        max_content_len = self.max_target_len - 2
        if len(tgt_tokens) > max_content_len:
            tgt_tokens = tgt_tokens[:max_content_len]

        # Sandwich encoded content tokens between <SOS> (index 2) and <EOS> (index 3)
        tgt_indices = [self.vocab.sos_idx] + self.vocab.encode(tgt_tokens) + [self.vocab.eos_idx]

        # Pad remaining space up to max_target_len
        if len(tgt_indices) < self.max_target_len:
            tgt_indices = tgt_indices + [self.vocab.pad_idx] * (self.max_target_len - len(tgt_indices))

        # Convert to PyTorch Long Tensors
        source_tensor = torch.tensor(src_indices, dtype=torch.long)
        target_tensor = torch.tensor(tgt_indices, dtype=torch.long)

        return source_tensor, target_tensor


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocabulary,
    batch_size: int = 16,
    max_source_len: int = 500,
    max_target_len: int = 60,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    train_dataset = SummarizationDataset(
        train_df, vocab, max_source_len=max_source_len, max_target_len=max_target_len
    )
    val_dataset = SummarizationDataset(
        val_df, vocab, max_source_len=max_source_len, max_target_len=max_target_len
    )
    test_dataset = SummarizationDataset(
        test_df, vocab, max_source_len=max_source_len, max_target_len=max_target_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
