"""
Decoder module for Seq2Seq summarization architecture.
Generates target tokens step-by-step using an LSTM, integrated with Bahdanau Attention.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.attention import BahdanauAttention


class Decoder(nn.Module):
    """
    Attention-Augmented LSTM Decoder generating target summary tokens one step at a time.
    
    Architecture (Single Time Step):
        Input Token
            ↓
        Word Embedding (128)
            ↓
        Concatenate [Embedding (128) + Attention Context Vector (512)]
            ↓
        LSTM Cell (Hidden: 256)
            ↓
        Concatenate [LSTM Output (256) + Attention Context (512) + Embedding (128)]
            ↓
        Linear Layer (896 -> Vocab Size 20,000)
            ↓
        Vocabulary Logits (Unnormalized Probabilities)
    """
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        decoder_hidden_dim: int = 256,
        encoder_hidden_dim: int = 256,
        dropout: float = 0.0,
        padding_idx: int = 0
    ):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_dim = encoder_hidden_dim * 2  # 512

        # 1. Word Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 2. Bahdanau Attention Module
        self.attention = BahdanauAttention(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=decoder_hidden_dim
        )

        # 3. Decoder LSTM: Input is Concatenated (Embedding + Context Vector) -> 128 + 512 = 640
        self.lstm = nn.LSTM(
            input_size=embedding_dim + self.encoder_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 4. Output Projection Layer: Concatenates (LSTM Output [256] + Context [512] + Embedding [128]) -> 896
        self.fc_out = nn.Linear(
            decoder_hidden_dim + self.encoder_dim + embedding_dim,
            vocab_size
        )

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single decoder time step.
        
        Args:
            input_token: Current target token indices of shape [batch_size].
            hidden_state: Previous decoder hidden state of shape [1, batch_size, 256].
            cell_state: Previous decoder cell state of shape [1, batch_size, 256].
            encoder_outputs: Encoder output sequence of shape [batch_size, source_length, 512].
            mask: Optional padding mask tensor of shape [batch_size, source_length].
            
        Returns:
            logits: Unnormalized vocabulary predictions of shape [batch_size, vocab_size].
            hidden_state: Updated decoder hidden state of shape [1, batch_size, 256].
            cell_state: Updated decoder cell state of shape [1, batch_size, 256].
            attn_weights: Attention distribution over source tokens of shape [batch_size, source_length].
        """
        # Step 1: Embed input token -> shape [batch_size, 128]
        embedded = self.dropout(self.embedding(input_token))

        # Step 2: Compute Attention Context Vector & Weights
        # hidden_state is [1, batch_size, 256] -> squeeze(0) becomes [batch_size, 256]
        context, attn_weights = self.attention(
            decoder_hidden=hidden_state.squeeze(0),
            encoder_outputs=encoder_outputs,
            mask=mask
        )

        # Step 3: Combine Embedding and Context Vector -> shape [batch_size, 1, 128 + 512 = 640]
        lstm_input = torch.cat((embedded, context), dim=1).unsqueeze(1)

        # Step 4: Pass through Decoder LSTM Cell
        # output shape: [batch_size, 1, 256]
        # hidden_state: [1, batch_size, 256]
        # cell_state:   [1, batch_size, 256]
        output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))

        # Step 5: Concatenate (Decoder LSTM Output [256] + Context [512] + Embedded [128]) -> [batch_size, 896]
        output_combined = torch.cat((output.squeeze(1), context, embedded), dim=1)

        # Step 6: Project to Vocabulary Logits -> shape [batch_size, vocab_size]
        logits = self.fc_out(output_combined)

        return logits, hidden_state, cell_state, attn_weights
