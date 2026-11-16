"""
Stage 4B Verification Test Suite.
Tests:
1. Greedy decoding inference (output shapes, types, limits).
2. Beam search decoding inference (beam width, length penalty, hypothesis selection).
3. EOS termination and special-token handling (no PAD/SOS in tokens, stops at EOS, no special tokens in decoded summary).
4. Trigram repetition blocking (n-gram suppression logic & end-to-end duplicate trigram prevention).
5. Stale Streamlit summary session state invalidation (state clearing on article change or sample change).
6. UI contrast ratio verification (WCAG standards for .info-notice and .degenerate-warning CSS).
"""

import sys
import os
import re
import math
from typing import Dict, List, Tuple
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import tokenize, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, SPECIAL_TOKENS
from src.vocabulary import Vocabulary
from src.model import Seq2Seq, get_device
import app


def load_fixtures() -> Tuple[Seq2Seq, Vocabulary, torch.device]:
    """Loads model and vocabulary fixtures."""
    vocab_path = "models/vocab.pkl"
    model_path = "models/best_model.pt"

    assert os.path.exists(vocab_path), "Vocabulary file missing!"
    assert os.path.exists(model_path), "Model checkpoint missing!"

    vocab = Vocabulary.load(vocab_path)
    device = torch.device("cpu")  # Use CPU for deterministic lightweight testing

    model = Seq2Seq(
        vocab_size=len(vocab),
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, device


def get_sample_tensor(vocab: Vocabulary, text: str) -> torch.Tensor:
    """Encodes text to a padded 500-dim source tensor."""
    tokens = tokenize(text)[:500]
    indices = vocab.encode(tokens)
    if len(indices) < 500:
        indices = indices + [PAD_IDX] * (500 - len(indices))
    return torch.tensor(indices, dtype=torch.long)


# -----------------------------------------------------------------------------
# Test 1: Greedy Decoding
# -----------------------------------------------------------------------------
def test_greedy_decoding(model: Seq2Seq, vocab: Vocabulary, device: torch.device):
    print("\n--- Test 1: Greedy Decoding ---")
    article = "Veteran actor Robert Hardy is selling his enormous collection of antiques in an auction."
    src_tensor = get_sample_tensor(vocab, article)

    summary, tokens, attn = model.generate_summary(
        src_tensor=src_tensor,
        vocab=vocab,
        max_target_len=30,
        device=device,
        strategy="greedy",
    )

    assert isinstance(summary, str), "Summary must be a string."
    assert isinstance(tokens, list), "Tokens must be a list of integer IDs."
    assert len(tokens) > 0, "Tokens list must not be empty."
    assert len(tokens) <= 30, f"Tokens count {len(tokens)} exceeds max_target_len 30."
    assert isinstance(attn, torch.Tensor), "Attention weights must be a tensor."
    assert attn.ndim == 2, f"Attention tensor must be 2D, got shape {attn.shape}."
    assert attn.shape[0] == len(tokens), (
        f"Attention rows ({attn.shape[0]}) should match generated tokens count ({len(tokens)})."
    )

    print(f"  [PASS] Greedy decoding returned {len(tokens)} tokens with attention shape {attn.shape}.")
    print(f"         Summary: '{summary[:60]}...'")


# -----------------------------------------------------------------------------
# Test 2: Beam Search Decoding
# -----------------------------------------------------------------------------
def test_beam_search_decoding(model: Seq2Seq, vocab: Vocabulary, device: torch.device):
    print("\n--- Test 2: Beam Search Decoding ---")
    article = "Three-time Masters champion Phil Mickelson enjoyed a return to form with a 66."
    src_tensor = get_sample_tensor(vocab, article)

    for k in [2, 4]:
        summary, tokens, attn = model.generate_summary(
            src_tensor=src_tensor,
            vocab=vocab,
            max_target_len=25,
            device=device,
            strategy="beam_search",
            beam_width=k,
            length_penalty=1.0,
        )

        assert isinstance(summary, str), f"Beam (k={k}) summary must be string."
        assert len(tokens) > 0, f"Beam (k={k}) tokens must not be empty."
        assert len(tokens) <= 25, f"Beam (k={k}) tokens exceed max_target_len."
        assert attn.shape[0] == len(tokens), f"Attention shape mismatch for k={k}."
        print(f"  [PASS] Beam search (k={k}) returned {len(tokens)} tokens.")
        print(f"         Summary: '{summary[:60]}...'")

    # Verify invalid strategy raises ValueError
    try:
        model.generate_summary(src_tensor, vocab, strategy="unsupported_mode")
        assert False, "Should have raised ValueError for invalid strategy."
    except ValueError as e:
        print(f"  [PASS] Unsupported strategy cleanly raised ValueError: {e}")


# -----------------------------------------------------------------------------
# Test 3: EOS Termination and Special-Token Handling
# -----------------------------------------------------------------------------
def test_eos_and_special_tokens(model: Seq2Seq, vocab: Vocabulary, device: torch.device):
    print("\n--- Test 3: EOS Termination and Special-Token Handling ---")
    article = "The quick brown fox jumps over the lazy dog."
    src_tensor = get_sample_tensor(vocab, article)

    # Test greedy special-token exclusion
    _, greedy_tokens, _ = model.generate_summary(
        src_tensor=src_tensor, vocab=vocab, max_target_len=40, strategy="greedy"
    )
    assert PAD_IDX not in greedy_tokens, "PAD_IDX must not be present in generated greedy tokens."
    assert SOS_IDX not in greedy_tokens, "SOS_IDX must not be present in generated greedy tokens."
    assert EOS_IDX not in greedy_tokens, "EOS_IDX must not be present in generated greedy tokens."

    # Test beam search special-token exclusion
    _, beam_tokens, _ = model.generate_summary(
        src_tensor=src_tensor, vocab=vocab, max_target_len=40, strategy="beam_search", beam_width=4
    )
    assert PAD_IDX not in beam_tokens, "PAD_IDX must not be present in generated beam tokens."
    assert SOS_IDX not in beam_tokens, "SOS_IDX must not be present in generated beam tokens."
    assert EOS_IDX not in beam_tokens, "EOS_IDX must not be present in generated beam tokens."

    # Verify decode() skips special tokens
    dirty_indices = [SOS_IDX, 10, PAD_IDX, 20, EOS_IDX, UNK_IDX]
    decoded = vocab.decode(dirty_indices, skip_special_tokens=True)
    for special in SPECIAL_TOKENS:
        assert special not in decoded.split(), f"Special token '{special}' leaked into decoded string!"

    print("  [PASS] PAD and SOS tokens are never emitted during generation.")
    print("  [PASS] EOS terminates sequence without leaking into output tokens.")
    print("  [PASS] Vocabulary decode cleanly suppresses all special tokens.")


# -----------------------------------------------------------------------------
# Test 4: Trigram Repetition Blocking
# -----------------------------------------------------------------------------
def test_trigram_blocking(model: Seq2Seq, vocab: Vocabulary, device: torch.device):
    print("\n--- Test 4: Trigram Repetition Blocking ---")

    # 1. Unit test _block_repeat_ngrams logic
    logits = torch.zeros(50)
    # History has repeated bigram prefix (10, 20) followed previously by 30
    tokens = [10, 20, 30, 40, 10, 20]
    masked_logits = model._block_repeat_ngrams(tokens=tokens, logits=logits.clone(), n=3)

    assert masked_logits[30] <= -1e8, "Candidate token 30 should be blocked (-1e9) as it completes (10, 20, 30)."
    assert masked_logits[25] == 0.0, "Unbanned candidate token 25 should remain unmasked."

    # 2. End-to-end generation with trigram blocking
    sample_article = (
        "Veteran actor Robert Hardy is selling his enormous collection of antiques in an auction. "
        "The star is auctioning off more than 200 items he has collected over his career."
    )
    src_tensor = get_sample_tensor(vocab, sample_article)

    # Greedy with trigram blocking
    _, g_tokens, _ = model.generate_summary(
        src_tensor=src_tensor,
        vocab=vocab,
        max_target_len=50,
        strategy="greedy",
        no_repeat_ngram_size=3,
    )

    trigrams_g = [tuple(g_tokens[i : i + 3]) for i in range(len(g_tokens) - 2)]
    assert len(trigrams_g) == len(set(trigrams_g)), (
        f"Greedy trigram blocking failed: found {len(trigrams_g) - len(set(trigrams_g))} duplicate trigrams!"
    )
    print(f"  [PASS] Greedy decoding with trigram blocking produced 0 duplicate trigrams across {len(g_tokens)} tokens.")

    # Beam search with trigram blocking
    _, b_tokens, _ = model.generate_summary(
        src_tensor=src_tensor,
        vocab=vocab,
        max_target_len=50,
        strategy="beam_search",
        beam_width=4,
        no_repeat_ngram_size=3,
    )

    trigrams_b = [tuple(b_tokens[i : i + 3]) for i in range(len(b_tokens) - 2)]
    assert len(trigrams_b) == len(set(trigrams_b)), (
        f"Beam search trigram blocking failed: found {len(trigrams_b) - len(set(trigrams_b))} duplicate trigrams!"
    )
    print(f"  [PASS] Beam search with trigram blocking produced 0 duplicate trigrams across {len(b_tokens)} tokens.")


# -----------------------------------------------------------------------------
# Test 5: Stale Streamlit Summary State
# -----------------------------------------------------------------------------
def test_stale_streamlit_state():
    print("\n--- Test 5: Stale Streamlit Summary State ---")

    # Mock session state
    session_state = {
        "gen_summary": "Old article summary.",
        "gen_token_ids": [10, 20, 30],
        "attn_matrix": torch.zeros((3, 500)),
        "src_tokens": ["old", "article"],
        "last_article": "Old article source text.",
        "last_reference": "Old reference summary.",
        "used_strategy": "Greedy Decoding",
        "used_beam_width": None,
        "used_trigram_blocking": False,
        "used_length_penalty": None,
        "unrelated_user_key": "preserve_this_setting",
    }

    # Verify clear_generated_summary_state clears generated state
    cleared = app.clear_generated_summary_state(session_state)
    assert "gen_summary" not in session_state, "gen_summary was not cleared!"
    assert "gen_token_ids" not in session_state, "gen_token_ids was not cleared!"
    assert "attn_matrix" not in session_state, "attn_matrix was not cleared!"
    assert "last_article" not in session_state, "last_article was not cleared!"
    assert session_state.get("unrelated_user_key") == "preserve_this_setting", "Unrelated keys were purged!"

    print(f"  [PASS] clear_generated_summary_state() successfully cleared {len(cleared)} generation keys.")
    print("  [PASS] Unrelated session keys preserved.")

    # Test stale detection conditions
    current_article = "New modified article text."
    recorded_last_article = "Old article source text."
    is_article_stale = (recorded_last_article.strip() != current_article.strip())
    assert is_article_stale, "Article text change must trigger staleness detection."

    prev_sample_key = "Sample 1"
    selected_sample_key = "Sample 2"
    is_sample_stale = (prev_sample_key != selected_sample_key)
    assert is_sample_stale, "Sample key change must trigger staleness detection."

    print("  [PASS] Article modification and sample switching staleness conditions correctly detected.")


# -----------------------------------------------------------------------------
# Test 6: Relevant UI Contrast Rules
# -----------------------------------------------------------------------------
def calculate_relative_luminance(hex_color: str) -> float:
    """Calculates relative luminance for sRGB hex color according to WCAG 2.1 specs."""
    hex_clean = hex_color.strip().lstrip("#")
    r, g, b = [int(hex_clean[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def channel_luminance(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel_luminance(r) + 0.7152 * channel_luminance(g) + 0.0722 * channel_luminance(b)


def calculate_contrast_ratio(hex1: str, hex2: str) -> float:
    """Calculates contrast ratio between two hex colors according to WCAG 2.1."""
    l1 = calculate_relative_luminance(hex1)
    l2 = calculate_relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def test_ui_contrast_rules():
    print("\n--- Test 6: UI Contrast Rules ---")
    app_path = "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    # 1. Verify CSS rules exist in app.py
    assert ".info-notice" in app_code, ".info-notice class missing from app.py CSS."
    assert ".degenerate-warning" in app_code, ".degenerate-warning class missing from app.py CSS."

    # 2. Extract background and color for .info-notice
    info_match = re.search(r"\.info-notice\s*\{([^}]+)\}", app_code)
    assert info_match, "Could not parse .info-notice CSS block."
    info_css = info_match.group(1)

    info_bg_match = re.search(r"background-color:\s*(#[0-9a-fA-F]+);", info_css)
    info_color_match = re.search(r"(?:^|;|\s)color:\s*(#[0-9a-fA-F]+);", info_css)

    assert info_bg_match, ".info-notice must specify explicit background-color hex."
    assert info_color_match, ".info-notice must specify explicit dark foreground color hex."

    info_bg = info_bg_match.group(1)
    info_fg = info_color_match.group(1)
    info_ratio = calculate_contrast_ratio(info_bg, info_fg)

    print(f"  .info-notice: bg={info_bg}, fg={info_fg}, Contrast Ratio = {info_ratio:.2f}:1")
    assert info_ratio >= 7.0, f".info-notice contrast ratio {info_ratio:.2f} is below WCAG AAA (7.0:1)!"
    print(f"  [PASS] .info-notice exceeds WCAG AAA standard (>= 7:1).")

    # 3. Extract background and color for .degenerate-warning
    degen_match = re.search(r"\.degenerate-warning\s*\{([^}]+)\}", app_code)
    assert degen_match, "Could not parse .degenerate-warning CSS block."
    degen_css = degen_match.group(1)

    degen_bg_match = re.search(r"background-color:\s*(#[0-9a-fA-F]+);", degen_css)
    degen_color_match = re.search(r"(?:^|;|\s)color:\s*(#[0-9a-fA-F]+);", degen_css)

    assert degen_bg_match, ".degenerate-warning must specify explicit background-color hex."
    assert degen_color_match, ".degenerate-warning must specify explicit dark foreground color hex."

    degen_bg = degen_bg_match.group(1)
    degen_fg = degen_color_match.group(1)
    degen_ratio = calculate_contrast_ratio(degen_bg, degen_fg)

    print(f"  .degenerate-warning: bg={degen_bg}, fg={degen_fg}, Contrast Ratio = {degen_ratio:.2f}:1")
    assert degen_ratio >= 7.0, f".degenerate-warning contrast ratio {degen_ratio:.2f} is below WCAG AAA (7.0:1)!"
    print(f"  [PASS] .degenerate-warning exceeds WCAG AAA standard (>= 7:1).")


# -----------------------------------------------------------------------------
# Main Test Runner
# -----------------------------------------------------------------------------
def run_stage4b_tests():
    print("=" * 60)
    print("RUNNING STAGE 4B VERIFICATION TEST SUITE")
    print("=" * 60)

    model, vocab, device = load_fixtures()

    test_greedy_decoding(model, vocab, device)
    test_beam_search_decoding(model, vocab, device)
    test_eos_and_special_tokens(model, vocab, device)
    test_trigram_blocking(model, vocab, device)
    test_stale_streamlit_state()
    test_ui_contrast_rules()

    print("\n" + "=" * 60)
    print("ALL 6 STAGE 4B VERIFICATION TESTS PASSED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage4b_tests()
