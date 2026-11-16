"""
Neural Network Sanity Test Suite.
Verifies all dimensions, forward passes, loss calculation, backpropagation, optimizer updates,
device placement (MPS / CPU), parameter counts, and inference summary generation
using a tiny synthetic batch prior to training.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.encoder import Encoder
from src.attention import BahdanauAttention
from src.decoder import Decoder
from src.model import Seq2Seq, get_device
from src.vocabulary import Vocabulary
from src.preprocess import PAD_IDX, SOS_IDX, EOS_IDX


def run_sanity_tests():
    print("=" * 60)
    print("RUNNING NNDL NEURAL NETWORK SANITY TEST SUITE")
    print("=" * 60)

    # 1. Device Selection Test
    device = get_device()
    print(f"1. Device Detected & Selected: {device}")

    # Set hyperparameter shapes for test
    vocab_size = 20000
    embedding_dim = 128
    encoder_hidden_dim = 256
    decoder_hidden_dim = 256
    batch_size = 4
    source_len = 500  # Updated default long-document sequence length
    target_len = 60

    print(f"\nTest Config: Batch Size={batch_size}, Source Length={source_len}, Target Length={target_len}")

    # Create synthetic input tensors
    src = torch.randint(4, vocab_size, (batch_size, source_len), dtype=torch.long)
    src[:, -50:] = PAD_IDX  # Add padding to test attention masking

    tgt = torch.randint(4, vocab_size, (batch_size, target_len), dtype=torch.long)
    tgt[:, 0] = SOS_IDX   # First token <SOS>
    tgt[:, -1] = EOS_IDX  # Last token <EOS>

    src = src.to(device)
    tgt = tgt.to(device)

    # 2. Test Word Embedding Layer Dimensions
    print("\n2. Testing Word Embedding output dimensions...")
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX).to(device)
    embedded_src = embedding(src)
    assert embedded_src.shape == (batch_size, source_len, embedding_dim), \
        f"Embedding shape mismatch: Expected ({batch_size}, {source_len}, {embedding_dim}), got {embedded_src.shape}"
    print(f"   [PASS] Embedding output shape: {embedded_src.shape}")

    # 3. Test Encoder Module Output Dimensions & Bidirectional Hidden State Conversion
    print("\n3. Testing Encoder output & state conversion dimensions...")
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim
    ).to(device)

    encoder_outputs, init_hidden, init_cell = encoder(src)
    expected_enc_dim = encoder_hidden_dim * 2  # 512

    assert encoder_outputs.shape == (batch_size, source_len, expected_enc_dim), \
        f"Encoder outputs shape mismatch: Expected ({batch_size}, {source_len}, {expected_enc_dim}), got {encoder_outputs.shape}"
    assert init_hidden.shape == (1, batch_size, decoder_hidden_dim), \
        f"Init hidden shape mismatch: Expected (1, {batch_size}, {decoder_hidden_dim}), got {init_hidden.shape}"
    assert init_cell.shape == (1, batch_size, decoder_hidden_dim), \
        f"Init cell shape mismatch: Expected (1, {batch_size}, {decoder_hidden_dim}), got {init_cell.shape}"

    print(f"   [PASS] Encoder outputs shape: {encoder_outputs.shape}")
    print(f"   [PASS] Converted initial decoder hidden shape: {init_hidden.shape}")
    print(f"   [PASS] Converted initial decoder cell shape: {init_cell.shape}")

    # 4. Test Bahdanau Attention Output Dimensions & Masking
    print("\n4. Testing Bahdanau Attention dimensions & pad masking...")
    attention = BahdanauAttention(
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim
    ).to(device)

    mask = (src != PAD_IDX).to(device)
    context_vector, attn_weights = attention(
        decoder_hidden=init_hidden.squeeze(0),
        encoder_outputs=encoder_outputs,
        mask=mask
    )

    assert context_vector.shape == (batch_size, expected_enc_dim), \
        f"Context vector shape mismatch: Expected ({batch_size}, {expected_enc_dim}), got {context_vector.shape}"
    assert attn_weights.shape == (batch_size, source_len), \
        f"Attention weights shape mismatch: Expected ({batch_size}, {source_len}), got {attn_weights.shape}"

    padded_pos_weights = attn_weights[0, -50:].sum().item()
    assert padded_pos_weights < 1e-5, f"Attention padding mask failed! Weight on pad: {padded_pos_weights}"

    print(f"   [PASS] Context vector shape: {context_vector.shape}")
    print(f"   [PASS] Attention weights shape: {attn_weights.shape}")
    print(f"   [PASS] Attention pad masking verified (pad position weight sum = {padded_pos_weights:.6f}).")

    # 5. Test Single-Step Decoder Output Dimensions
    print("\n5. Testing Decoder single-step output dimensions...")
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim
    ).to(device)

    input_token = tgt[:, 0]  # <SOS> token
    step_logits, dec_hidden, dec_cell, step_attn = decoder(
        input_token=input_token,
        hidden_state=init_hidden,
        cell_state=init_cell,
        encoder_outputs=encoder_outputs,
        mask=mask
    )

    assert step_logits.shape == (batch_size, vocab_size), \
        f"Decoder logits shape mismatch: Expected ({batch_size}, {vocab_size}), got {step_logits.shape}"

    print(f"   [PASS] Decoder step logits shape: {step_logits.shape}")

    # 6. Test Complete Seq2Seq Model Forward Pass
    print("\n6. Testing complete Seq2Seq Model forward pass...")
    model = Seq2Seq(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX
    ).to(device)

    outputs = model(src, tgt, teacher_forcing_ratio=0.5)
    expected_output_shape = (batch_size, target_len - 1, vocab_size)
    assert outputs.shape == expected_output_shape, \
        f"Seq2Seq output shape mismatch: Expected {expected_output_shape}, got {outputs.shape}"
    print(f"   [PASS] Seq2Seq output logits shape: {outputs.shape}")

    # 7. Test CrossEntropyLoss Calculation & Backpropagation
    print("\n7. Testing CrossEntropyLoss calculation and backward pass...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = model.compute_loss(outputs, tgt, criterion)
    print(f"   [PASS] Loss computed successfully: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    print("   [PASS] Backpropagation (loss.backward()) executed cleanly.")

    # 8. Test Optimizer Step
    optimizer.step()
    print("   [PASS] Optimizer parameter update (optimizer.step()) executed successfully.")

    # 9. Test Inference summary generation utility
    print("\n8. Testing greedy inference generate_summary() method...")
    vocab = Vocabulary(max_size=vocab_size)
    dummy_tokens = [["apple", "inc", "reported", "profit", "increase", "."]]
    vocab.build_vocab(dummy_tokens)

    src_single = src[0]  # Shape [500]
    gen_text, gen_indices, attn_matrix = model.generate_summary(
        src_single, vocab, max_target_len=60, device=device
    )
    print(f"   [PASS] Greedy summary generation successful!")
    print(f"          Generated tokens count: {len(gen_indices)}")

    # 10. Parameter Count Summary
    param_counts = model.count_parameters()
    print("\n" + "=" * 60)
    print("MODEL PARAMETER BREAKDOWN")
    print("=" * 60)
    for component, count in param_counts.items():
        print(f"  - {component:30s}: {count:,} parameters")

    print("\n" + "=" * 60)
    print("ALL 10 NEURAL NETWORK SANITY TESTS PASSED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_tests()
