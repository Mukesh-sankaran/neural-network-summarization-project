"""
Encoder module for Seq2Seq summarization architecture.
Uses a trainable Word Embedding and a Bidirectional LSTM.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder for processing source document sequences.
    
    Architecture:
    1. Word Embedding Layer: Maps discrete token IDs -> continuous vectors.
    2. Bidirectional LSTM: Reads sequence in forward and backward directions.
    3. State Projection Linear Layers: Converts concatenated 2-directional
       hidden and cell states into a single directional vector for Decoder initialization.
    """
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        padding_idx: int = 0
    ):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers

        # 1. Embedding Layer: Maps token indices to embedding vectors of dimension 128
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 2. Bidirectional LSTM:
        # - forward hidden size: 256
        # - backward hidden size: 256
        # - total output dimension at each time step: 256 * 2 = 512
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. State Projections:
        # Convert forward and backward hidden/cell states [2 * encoder_hidden_dim = 512]
        # into decoder hidden/cell states [decoder_hidden_dim = 256].
        self.fc_hidden = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.fc_cell = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Encoder.
        
        Args:
            src: Tensor of shape [batch_size, source_length] containing word token indices.
            
        Returns:
            encoder_outputs: Tensor of shape [batch_size, source_length, 512] (256 forward + 256 backward).
            decoder_init_hidden: Tensor of shape [1, batch_size, 256] for initializing Decoder LSTM.
            decoder_init_cell: Tensor of shape [1, batch_size, 256] for initializing Decoder LSTM.
        """
        # Step 1: Embed source tokens -> shape [batch_size, source_length, embedding_dim]
        embedded = self.dropout(self.embedding(src))

        # Step 2: Pass through Bidirectional LSTM
        # encoder_outputs: [batch_size, source_length, 512]
        # hidden: [2 * num_layers, batch_size, 256] (hidden[0]=forward, hidden[1]=backward for 1 layer)
        # cell:   [2 * num_layers, batch_size, 256]
        encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # Step 3: Combine forward and backward states for Decoder initialization
        # Extract forward (hidden[0]) and backward (hidden[1]) states
        hidden_forward = hidden[0:1]   # [1, batch_size, 256]
        hidden_backward = hidden[1:2]  # [1, batch_size, 256]
        
        cell_forward = cell[0:1]       # [1, batch_size, 256]
        cell_backward = cell[1:2]      # [1, batch_size, 256]

        # Concatenate along state dimension: [1, batch_size, 512]
        combined_hidden = torch.cat((hidden_forward, hidden_backward), dim=2)
        combined_cell = torch.cat((cell_forward, cell_backward), dim=2)

        # Project 512 -> 256 via Linear layer + Tanh activation: [1, batch_size, 256]
        decoder_init_hidden = torch.tanh(self.fc_hidden(combined_hidden))
        decoder_init_cell = torch.tanh(self.fc_cell(combined_cell))

        return encoder_outputs, decoder_init_hidden, decoder_init_cell
