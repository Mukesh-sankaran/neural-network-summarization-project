"""
Bahdanau (Additive) Attention Mechanism module.
Calculates alignment scores between decoder hidden state and encoder outputs,
masking padding tokens and computing a dynamic context vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BahdanauAttention(nn.Module):
    """
    Bahdanau Additive Attention Mechanism.
    
    Equations:
        1. energy = tanh( W_encoder(encoder_outputs) + W_decoder(decoder_hidden) )
        2. scores = v^T * energy
        3. scores = scores.masked_fill(mask == 0, -1e9)  (Mask out <PAD> positions)
        4. attention_weights = softmax(scores, dim=1)
        5. context_vector = sum(attention_weights * encoder_outputs)
    """
    def __init__(
        self,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        attention_dim: int = 256
    ):
        super(BahdanauAttention, self).__init__()

        # Encoder outputs are bidirectional: dimension = encoder_hidden_dim * 2 = 512
        self.encoder_dim = encoder_hidden_dim * 2
        self.decoder_dim = decoder_hidden_dim
        self.attention_dim = attention_dim

        # Projection weights
        self.W_encoder = nn.Linear(self.encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(self.decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Bahdanau Attention.
        
        Args:
            decoder_hidden: Current/previous decoder hidden state of shape [batch_size, 256].
            encoder_outputs: Encoder outputs of shape [batch_size, source_length, 512].
            mask: Optional boolean or binary mask tensor of shape [batch_size, source_length],
                  where 1/True indicates valid tokens and 0/False indicates <PAD> tokens.
                  
        Returns:
            context_vector: Weighted sum of encoder outputs, shape [batch_size, 512].
            attention_weights: Attention distribution over source tokens, shape [batch_size, source_length].
        """
        # Step 1: Project Encoder Outputs -> shape [batch_size, source_length, attention_dim]
        proj_encoder = self.W_encoder(encoder_outputs)

        # Step 2: Project Decoder Hidden State -> shape [batch_size, 1, attention_dim]
        proj_decoder = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Step 3: Compute Additive Energy: tanh( W_encoder(enc) + W_decoder(dec) )
        # Broadcast addition across source_length dimension -> shape [batch_size, source_length, attention_dim]
        energy = torch.tanh(proj_encoder + proj_decoder)

        # Step 4: Compute Raw Attention Scores: v^T * energy -> shape [batch_size, source_length]
        scores = self.v(energy).squeeze(2)

        # Step 5: Mask <PAD> positions so the attention mechanism does NOT attend to padding
        if mask is not None:
            # Set scores for padding positions to a very large negative value (-1e9)
            # so softmax outputs approximately 0.0 probability for <PAD> tokens.
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 6: Softmax to obtain normalized Attention Weights -> shape [batch_size, source_length]
        attention_weights = F.softmax(scores, dim=1)

        # Step 7: Calculate Context Vector as weighted sum of encoder outputs
        # attention_weights: [batch_size, 1, source_length]
        # encoder_outputs:    [batch_size, source_length, 512]
        # context_vector:     [batch_size, 512]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context_vector, attention_weights
