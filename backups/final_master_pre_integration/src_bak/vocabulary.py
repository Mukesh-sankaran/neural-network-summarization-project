"""
Vocabulary class for managing token-to-index and index-to-token mappings.
Builds vocabulary from training dataset tokens and saves/loads state.
"""

import os
import pickle
from collections import Counter
from typing import List, Dict, Optional

from src.preprocess import (
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX,
    SPECIAL_TOKENS
)


class Vocabulary:
    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.word2idx = {
            self.pad_token: self.pad_idx,
            self.unk_token: self.unk_idx,
            self.sos_token: self.sos_idx,
            self.eos_token: self.eos_idx,
        }
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

    def build_vocab(self, tokenized_texts: List[List[str]]):
        """
        Builds the vocabulary from a list of tokenized documents/summaries.
        Must be called ONLY on training data to avoid data leakage.
        """
        self.word_freq = Counter()
        for tokens in tokenized_texts:
            self.word_freq.update(tokens)

        # Reserve slots for special tokens
        num_special = len(SPECIAL_TOKENS)
        max_words = self.max_size - num_special

        # Select most common non-special tokens
        most_common = self.word_freq.most_common(max_words)

        self._init_special_tokens()
        current_idx = num_special

        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Converts a list of token strings to integer indices.
        Uses UNK_IDX for tokens not found in the vocabulary.
        """
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of integer indices back to a single text string.
        """
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in SPECIAL_TOKENS:
                    continue
                tokens.append(token)
        return " ".join(tokens)

    def save(self, filepath: str = "models/vocab.pkl"):
        """
        Saves the Vocabulary object to a pickle file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "models/vocab.pkl") -> "Vocabulary":
        """
        Loads a Vocabulary object from a pickle file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        with open(filepath, "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)
