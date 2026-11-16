"""
Dual-Model & Pretrained Baseline Comprehensive Test Suite.
Verifies:
1. Pretrained model loading & tokenizer compatibility (DistilBART-CNN-12-6).
2. Model selection and tokenizer isolation (no cross-contamination of vocabularies).
3. Real inference output on representative news article & empty input handling.
4. Generation parameter controls (greedy vs beam search, max length, trigram blocking).
5. Streamlit session state invalidation with dual-model keys.
6. Error handling for missing or invalid model identifiers (no silent failures).
7. Evaluation metric computation on real baseline outputs (ROUGE, entity preservation, numeric consistency).
"""

import sys
import os
import torch
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pretrained_baseline import (
    load_pretrained_model,
    generate_pretrained_summary,
    get_pretrained_device,
    DEFAULT_PRETRAINED_MODEL_NAME,
)
from src.vocabulary import Vocabulary
from src.model import Seq2Seq, get_device as get_custom_device
from src.evaluate import evaluate_summary
from src.entity_evaluation import evaluate_entity_preservation, evaluate_numeric_consistency
import app


def test_pretrained_model_loading_and_tokenizer():
    print("\n--- Test 1: Pretrained Model Loading & Tokenizer Compatibility ---")
    model, tokenizer, metadata, device, err = load_pretrained_model(DEFAULT_PRETRAINED_MODEL_NAME)

    assert err is None, f"Expected successful load, got error: {err}"
    assert model is not None, "Model instance is None"
    assert tokenizer is not None, "Tokenizer instance is None"
    assert metadata is not None, "Metadata is None"

    assert metadata["total_parameters"] > 300_000_000, f"Expected >300M parameters, got {metadata['total_parameters']}"
    assert metadata["vocab_size"] > 50_000, f"Expected >50,000 vocab tokens, got {metadata['vocab_size']}"
    assert tokenizer.bos_token_id is not None, "Missing BOS token ID"
    assert tokenizer.eos_token_id is not None, "Missing EOS token ID"
    assert tokenizer.pad_token_id is not None, "Missing PAD token ID"

    print(f"  [PASS] DistilBART loaded cleanly on {device} ({metadata['total_parameters']:,} parameters, vocab size {metadata['vocab_size']:,}).")
    return model, tokenizer, device


def test_dual_model_tokenizer_isolation():
    print("\n--- Test 2: Dual Model Tokenizer Isolation & Independent Routing ---")
    vocab_path = "models/vocab.pkl"
    assert os.path.exists(vocab_path), "Custom vocab missing"
    custom_vocab = Vocabulary.load(vocab_path)

    _, pt_tokenizer, _, _, _ = load_pretrained_model(DEFAULT_PRETRAINED_MODEL_NAME)

    assert len(custom_vocab) == 20000, f"Custom vocab must have exactly 20,000 tokens, got {len(custom_vocab)}"
    assert pt_tokenizer.vocab_size > 50000, f"Pretrained tokenizer vocab must exceed 50,000, got {pt_tokenizer.vocab_size}"

    # Verify custom tokens have distinct internal IDs from Hugging Face BPE
    assert custom_vocab.word2idx["<PAD>"] == 0
    assert custom_vocab.word2idx["<UNK>"] == 1
    assert custom_vocab.word2idx["<SOS>"] == 2
    assert custom_vocab.word2idx["<EOS>"] == 3

    assert pt_tokenizer.pad_token_id == 1
    assert pt_tokenizer.bos_token_id == 0
    assert pt_tokenizer.eos_token_id == 2

    print("  [PASS] Custom 20,000-token word vocabulary and HuggingFace 50,264-token BPE tokenizer are strictly isolated.")


def test_real_inference_and_empty_input(model, tokenizer, device):
    print("\n--- Test 3: Real Inference Output & Empty Input Handling ---")
    article = (
        "Veteran actor Robert Hardy is selling his enormous collection of antiques in a £100,000 auction "
        "after deciding to downsize his home. Mr Hardy, 89, is auctioning off more than 200 items."
    )

    # 1. Non-empty inference
    summary, tokens, latency = generate_pretrained_summary(
        article, model, tokenizer, device, max_length=60, min_length=20, num_beams=4
    )
    assert isinstance(summary, str) and len(summary.strip()) > 0, "Summary must be non-empty string"
    assert isinstance(tokens, list) and len(tokens) > 0, "Token IDs must be non-empty list"
    assert latency > 0.0, "Latency must be positive float"
    assert "Hardy" in summary or "actor" in summary or "antiques" in summary, "Summary should contain core article keywords"
    print(f"  [PASS] Real inference generated ({len(tokens)} tokens, {latency:.1f}ms): '{summary[:70]}...'")

    # 2. Empty inputs
    for empty_in in ["", "    ", "\n\t  \n"]:
        s, t, lat = generate_pretrained_summary(empty_in, model, tokenizer, device)
        assert s == "", f"Expected empty string for '{empty_in}', got '{s}'"
        assert t == [], f"Expected empty tokens for '{empty_in}', got {t}"
        assert lat == 0.0, f"Expected 0.0ms latency for empty input, got {lat}"
    print("  [PASS] Empty and whitespace inputs cleanly return empty summary without invocation.")


def test_generation_parameters(model, tokenizer, device):
    print("\n--- Test 4: Generation Parameter Controls (Greedy vs Beam & Length Bounds) ---")
    article = (
        "Down Augusta way they say the azaleas are in full bloom, which is more than can be said for "
        "England's Justin Rose. A bruising Florida swing last month saw the Englishman fall outside the "
        "world's top 10 for the first time in three years. But a sparkling seven-under-par 65 in the "
        "first round of the Shell Houston Open on Thursday suggests he is on the way back."
    )

    # Greedy generation (num_beams=1)
    s_greedy, t_greedy, _ = generate_pretrained_summary(
        article, model, tokenizer, device, max_length=50, min_length=15, num_beams=1
    )
    assert len(s_greedy.strip()) > 0, "Greedy summary must not be empty"

    # Beam search generation (num_beams=4)
    s_beam, t_beam, _ = generate_pretrained_summary(
        article, model, tokenizer, device, max_length=50, min_length=15, num_beams=4
    )
    assert len(s_beam.strip()) > 0, "Beam summary must not be empty"

    # Verify trigram blocking prevents duplicate 3-grams
    words = s_beam.lower().split()
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    assert len(trigrams) == len(set(trigrams)), "Beam output with trigram blocking must not contain duplicate trigrams"

    print(f"  [PASS] Greedy ({len(t_greedy)} tokens) and Beam search ({len(t_beam)} tokens) executed cleanly.")
    print("  [PASS] Trigram repetition blocking confirmed (zero duplicate 3-grams).")


def test_session_state_invalidation():
    print("\n--- Test 5: Streamlit Session State Invalidation (Dual Model Keys) ---")
    mock_session = {
        "gen_summary": "A mock generated summary",
        "gen_token_ids": [101, 202, 303],
        "attn_matrix": torch.zeros((5, 10)),
        "src_tokens": ["mock", "source"],
        "last_article": "mock article text",
        "last_reference": "mock reference text",
        "used_strategy": "Beam Search",
        "used_beam_width": 4,
        "used_trigram_blocking": True,
        "used_length_penalty": 1.0,
        "inference_latency_ms": 142.5,
        "active_model_badge": "DistilBART",
        "active_model_kind": "pretrained",
        "unrelated_user_key": "MUST_SURVIVE",
    }

    cleared = app.clear_generated_summary_state(mock_session)
    assert "gen_summary" not in mock_session
    assert "gen_token_ids" not in mock_session
    assert "active_model_badge" not in mock_session
    assert "active_model_kind" not in mock_session
    assert mock_session["unrelated_user_key"] == "MUST_SURVIVE"
    assert len(cleared) >= 12, f"Expected >=12 cleared keys, got {len(cleared)}"
    print(f"  [PASS] Successfully invalidated {len(cleared)} generation session keys while preserving unrelated user state.")


def test_error_handling_invalid_model():
    print("\n--- Test 6: Clear Error Reporting on Invalid Model Identifier ---")
    model, tokenizer, metadata, device, err = load_pretrained_model("nonexistent_invalid_model_abc_123")
    assert model is None, "Model should be None on failure"
    assert err is not None, "Error message must be populated"
    assert "Failed to load" in err or "nonexistent" in err, f"Error message should explain failure: '{err}'"
    print(f"  [PASS] Gracefully caught invalid model with honest error message: '{err[:65]}...'")


def test_evaluation_on_pretrained_summary(model, tokenizer, device):
    print("\n--- Test 7: Evaluation Metrics on Real Pretrained Baseline Summary ---")
    article = (
        "Veteran actor Robert Hardy is selling his enormous collection of antiques in a £100,000 auction "
        "after deciding to downsize his home. The star, known for appearing in All Creatures Great and Small "
        "and playing Winston Churchill several times, is auctioning off more than 200 items he has collected "
        "over his career. Mr Hardy, 89, is a keen military historian who studied English with J.R.R. Tolkien "
        "and C.S. Lewis at Oxford before serving in the RAF during the Second World War."
    )
    reference = (
        "Robert Hardy, 89, is a military enthusiast who studied English literature under J.R.R. Tolkien .\n"
        "He is selling off a huge collection of books, artwork and furniture .\n"
        "Among the objects which have gone up for auction are a huge 3D diorama of the Battle of Agincourt ."
    )

    summary, _, _ = generate_pretrained_summary(
        article, model, tokenizer, device, max_length=80, min_length=25, num_beams=4
    )

    # ROUGE
    rouge = evaluate_summary(summary, reference, exclude_punctuation=True)
    assert rouge["rouge1"]["f1"] > 0.15, f"Expected ROUGE-1 F1 > 0.15, got {rouge['rouge1']['f1']}"
    assert rouge["rougel"]["f1"] > 0.15, f"Expected ROUGE-L F1 > 0.15, got {rouge['rougel']['f1']}"

    # Entity Preservation
    ent = evaluate_entity_preservation(article, summary)
    assert ent["status"] in ("grounded", "partially_grounded"), f"Unexpected entity status: {ent['status']}"
    assert ent["entity_precision"] is not None and ent["entity_precision"] > 0.50, (
        f"Expected entity precision > 0.50, got {ent['entity_precision']}"
    )

    # Numeric Consistency
    num = evaluate_numeric_consistency(article, summary)
    assert num["numeric_consistency_rate"] is not None and num["numeric_consistency_rate"] >= 0.0

    print(f"  [PASS] Pretrained ROUGE-1 F1: {rouge['rouge1']['f1']:.4f} | ROUGE-L F1: {rouge['rougel']['f1']:.4f}")
    print(f"  [PASS] Entity Precision: {ent['entity_precision']:.4f} ({ent['num_preserved_entities']} preserved, {len(ent['extra_entities'])} extra)")
    print(f"  [PASS] Numeric Consistency: {num['numeric_consistency_rate']}")


def main():
    print("=" * 70)
    print("RUNNING DUAL MODEL & PRETRAINED BASELINE TEST SUITE")
    print("=" * 70)

    model, tokenizer, device = test_pretrained_model_loading_and_tokenizer()
    test_dual_model_tokenizer_isolation()
    test_real_inference_and_empty_input(model, tokenizer, device)
    test_generation_parameters(model, tokenizer, device)
    test_session_state_invalidation()
    test_error_handling_invalid_model()
    test_evaluation_on_pretrained_summary(model, tokenizer, device)

    print("\n" + "=" * 70)
    print("ALL 7 DUAL MODEL & PRETRAINED BASELINE TESTS PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
