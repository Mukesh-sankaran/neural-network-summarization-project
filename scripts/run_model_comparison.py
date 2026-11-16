"""
Model Comparison and Scientific Evaluation Script.
Evaluates both the Pretrained Baseline (sshleifer/distilbart-cnn-12-6) and the
Custom Seq2Seq Research Model (best_model.pt) across 5 representative news articles.
Measures ROUGE-1/2/L, entity preservation, numeric consistency, repetition metrics, and latency.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import torch

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import tokenize, PAD_IDX, SOS_IDX, EOS_IDX
from src.vocabulary import Vocabulary
from src.model import Seq2Seq
from src.evaluate import evaluate_summary
from src.entity_evaluation import evaluate_entity_preservation, evaluate_numeric_consistency
from src.pretrained_baseline import load_pretrained_model, generate_pretrained_summary
from src.data_loader import load_dataset_split

ARTICLES_DATASET = [
    {
        "id": "sample_hardy",
        "title": "Robert Hardy Antiques Auction (CNN/DailyMail Example #10)",
        "article": (
            "Veteran actor Robert Hardy is selling his enormous collection of antiques in a £100,000 "
            "auction after deciding to downsize his home. The star, known for appearing in All Creatures "
            "Great and Small and playing Winston Churchill several times, is auctioning off more than "
            "200 items he has collected over his career. Mr Hardy, 89, is a keen military historian who "
            "studied English with J.R.R. Tolkien and C.S. Lewis at Oxford before serving in the RAF during "
            "the Second World War. Among the objects which have gone up for auction are a huge 3D diorama "
            "of the Battle of Agincourt, historic longbows, and rare antique furniture."
        ),
        "reference": (
            "Robert Hardy, 89, is a military enthusiast who studied English literature under J.R.R. Tolkien .\n"
            "He is selling off a huge collection of books, artwork and furniture .\n"
            "Among the objects which have gone up for auction are a huge 3D diorama of the Battle of Agincourt ."
        ),
    },
    {
        "id": "sample_golf",
        "title": "Golf: Justin Rose Houston Open (CNN/DailyMail Example #1)",
        "article": (
            "Down Augusta way they say the azaleas are in full bloom, which is more than can be said for "
            "England's Justin Rose. A bruising Florida swing last month saw the Englishman fall outside the "
            "world's top 10 for the first time in three years. But a sparkling seven-under-par 65 in the "
            "first round of the Shell Houston Open on Thursday suggests he is on the way back. Three-time "
            "Masters champion Phil Mickelson enjoyed a return to form with a 66, while Paul Casey celebrated "
            "his Masters invitation with a fine round of 68."
        ),
        "reference": (
            "Justin Rose bounced back from Florida misery by carding 69 in Houston .\n"
            "Three-time Masters champion Phil Mickelson enjoyed return to form .\n"
            "Paul Casey celebrated last-gasp Masters invitation with fine round of 68 ."
        ),
    },
    {
        "id": "sample_ambulance",
        "title": "Hoax Ambulance Calls (CNN/DailyMail Example #3)",
        "article": (
            "When emergency crews received a call saying 'someone' had been knocked down by a car in Essex, "
            "they immediately sent two ambulances to the rescue. But the vehicles returned to base empty after "
            "discovering the victim was in fact a squirrel. East of England ambulance service revealed the "
            "bizarre incident as they warned hoax and trivial calls are putting lives at risk. In another "
            "incident, a man dialled 999 to say he had dropped a burger which was bleeding. Ambulance bosses "
            "stressed that non-emergency calls delay response times for critical patients."
        ),
        "reference": (
            "Woman reported 'someone' had been run over, but victim was a squirrel .\n"
            "Another man dialled 999 to say he dropped a burger which was 'bleeding'\n"
            "East of England ambulance service warned hoax calls can cost lives ."
        ),
    },
]


def load_additional_cnn_articles() -> List[Dict[str, Any]]:
    """Loads 2 held-out articles from local CNN/DailyMail test parquet split."""
    test_df = load_dataset_split("cnn_dailymail", "test")
    samples = []
    # Index 0
    samples.append({
        "id": "cnn_test_0",
        "title": "Palestinian Authority ICC Membership (CNN Test #0)",
        "article": test_df["source_text"].iloc[0],
        "reference": test_df["target_summary"].iloc[0],
    })
    # Index 5
    samples.append({
        "id": "cnn_test_5",
        "title": "Trey Moses & Ellie Meredith Prom Date (CNN Test #5)",
        "article": test_df["source_text"].iloc[5],
        "reference": test_df["target_summary"].iloc[5],
    })
    return samples


def compute_repetition_rate(text: str) -> float:
    """Computes percentage of repeated unigrams in generated text."""
    words = text.lower().split()
    if not words:
        return 0.0
    from collections import Counter
    c = Counter(words)
    repeated = sum(count - 1 for count in c.values())
    return float(repeated / len(words))


def main():
    print("=" * 80)
    print("NNDL NEURAL NEWS SUMMARIZATION — DUAL MODEL SCIENTIFIC COMPARISON")
    print("=" * 80)

    articles = list(ARTICLES_DATASET) + load_additional_cnn_articles()
    print(f"Total evaluation test articles: {len(articles)}")

    # Load Custom Model Resources
    vocab_path = "models/vocab.pkl"
    custom_path = "models/best_model.pt"
    custom_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("\n[1/3] Loading Custom Research Model Checkpoint...")
    vocab = Vocabulary.load(vocab_path)
    custom_model = Seq2Seq(
        vocab_size=len(vocab),
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
    ).to(custom_device)
    chk = torch.load(custom_path, map_location=custom_device)
    custom_model.load_state_dict(chk["model_state_dict"])
    custom_model.eval()
    print(f"  Custom Seq2Seq loaded ({sum(p.numel() for p in custom_model.parameters()):,} params) on {custom_device}")

    # Load Pretrained DistilBART Baseline
    print("\n[2/3] Loading Pretrained Baseline (sshleifer/distilbart-cnn-12-6)...")
    pt_model, pt_tokenizer, pt_meta, pt_device, pt_err = load_pretrained_model()
    if pt_err:
        print(f"  ERROR loading pretrained model: {pt_err}")
        return
    print(f"  DistilBART loaded ({pt_meta['total_parameters']:,} params) on {pt_device}")

    results = []

    print("\n[3/3] Running Evaluation Across All 5 Test Articles...")
    for idx, item in enumerate(articles, 1):
        print(f"\n--- Article {idx}/{len(articles)}: {item['title']} ---")
        art_text = item["article"]
        ref_text = item["reference"]

        # --- A. Pretrained DistilBART ---
        pt_summary, pt_tokens, pt_latency = generate_pretrained_summary(
            art_text, pt_model, pt_tokenizer, pt_device,
            max_length=142, min_length=30, num_beams=4, no_repeat_ngram_size=3
        )
        pt_rouge = evaluate_summary(pt_summary, ref_text, exclude_punctuation=True)
        pt_entity = evaluate_entity_preservation(art_text, pt_summary)
        pt_num = evaluate_numeric_consistency(art_text, pt_summary)
        pt_rep = compute_repetition_rate(pt_summary)

        # --- B. Custom Seq2Seq (Greedy) ---
        raw_tokens = tokenize(art_text)[:500]
        src_indices = vocab.encode(raw_tokens)
        if len(src_indices) < 500:
            src_indices = src_indices + [PAD_IDX] * (500 - len(src_indices))
        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        t0 = time.perf_counter()
        with torch.no_grad():
            c_summary_gr, c_tokens_gr, _ = custom_model.generate_summary(
                src_tensor, vocab, max_target_len=60, device=custom_device, strategy="greedy"
            )
        c_lat_gr = (time.perf_counter() - t0) * 1000.0
        c_rouge_gr = evaluate_summary(c_summary_gr, ref_text, exclude_punctuation=True)
        c_entity_gr = evaluate_entity_preservation(art_text, c_summary_gr)
        c_num_gr = evaluate_numeric_consistency(art_text, c_summary_gr)
        c_rep_gr = compute_repetition_rate(c_summary_gr)

        # --- C. Custom Seq2Seq (Beam Search k=4 + Trigram Blocking) ---
        t0 = time.perf_counter()
        with torch.no_grad():
            c_summary_bm, c_tokens_bm, _ = custom_model.generate_summary(
                src_tensor, vocab, max_target_len=60, device=custom_device,
                strategy="beam_search", beam_width=4, no_repeat_ngram_size=3
            )
        c_lat_bm = (time.perf_counter() - t0) * 1000.0
        c_rouge_bm = evaluate_summary(c_summary_bm, ref_text, exclude_punctuation=True)
        c_entity_bm = evaluate_entity_preservation(art_text, c_summary_bm)
        c_num_bm = evaluate_numeric_consistency(art_text, c_summary_bm)
        c_rep_bm = compute_repetition_rate(c_summary_bm)

        print(f"  [Pretrained DistilBART] Latency: {pt_latency:.1f}ms | ROUGE-1 F1: {pt_rouge['rouge1']['f1']:.4f} | ROUGE-L F1: {pt_rouge['rougel']['f1']:.4f}")
        print(f"    Summary: {pt_summary[:120]}...")
        print(f"  [Custom Greedy]        Latency: {c_lat_gr:.1f}ms | ROUGE-1 F1: {c_rouge_gr['rouge1']['f1']:.4f} | ROUGE-L F1: {c_rouge_gr['rougel']['f1']:.4f}")
        print(f"    Summary: {c_summary_gr[:120]}...")
        print(f"  [Custom Beam+Trigram]  Latency: {c_lat_bm:.1f}ms | ROUGE-1 F1: {c_rouge_bm['rouge1']['f1']:.4f} | ROUGE-L F1: {c_rouge_bm['rougel']['f1']:.4f}")
        print(f"    Summary: {c_summary_bm[:120]}...")

        record = {
            "id": item["id"],
            "title": item["title"],
            "reference": ref_text,
            "distilbart": {
                "summary": pt_summary,
                "latency_ms": pt_latency,
                "rouge1": pt_rouge["rouge1"],
                "rouge2": pt_rouge["rouge2"],
                "rougel": pt_rouge["rougel"],
                "entity": pt_entity,
                "numeric": pt_num,
                "repetition_rate": pt_rep,
            },
            "custom_greedy": {
                "summary": c_summary_gr,
                "latency_ms": c_lat_gr,
                "rouge1": c_rouge_gr["rouge1"],
                "rouge2": c_rouge_gr["rouge2"],
                "rougel": c_rouge_gr["rougel"],
                "entity": c_entity_gr,
                "numeric": c_num_gr,
                "repetition_rate": c_rep_gr,
            },
            "custom_beam": {
                "summary": c_summary_bm,
                "latency_ms": c_lat_bm,
                "rouge1": c_rouge_bm["rouge1"],
                "rouge2": c_rouge_bm["rouge2"],
                "rougel": c_rouge_bm["rougel"],
                "entity": c_entity_bm,
                "numeric": c_num_bm,
                "repetition_rate": c_rep_bm,
            },
        }
        results.append(record)

    # Compute Averages
    def avg(vals):
        clean = [v for v in vals if v is not None]
        return float(sum(clean) / len(clean)) if clean else 0.0

    summary_metrics = {
        "distilbart": {
            "avg_latency_ms": avg([r["distilbart"]["latency_ms"] for r in results]),
            "avg_rouge1_f1": avg([r["distilbart"]["rouge1"]["f1"] for r in results]),
            "avg_rouge2_f1": avg([r["distilbart"]["rouge2"]["f1"] for r in results]),
            "avg_rougel_f1": avg([r["distilbart"]["rougel"]["f1"] for r in results]),
            "avg_entity_precision": avg([r["distilbart"]["entity"]["entity_precision"] for r in results]),
            "avg_entity_recall": avg([r["distilbart"]["entity"]["entity_recall"] for r in results]),
            "avg_numeric_rate": avg([r["distilbart"]["numeric"]["numeric_consistency_rate"] for r in results]),
            "avg_repetition_rate": avg([r["distilbart"]["repetition_rate"] for r in results]),
        },
        "custom_greedy": {
            "avg_latency_ms": avg([r["custom_greedy"]["latency_ms"] for r in results]),
            "avg_rouge1_f1": avg([r["custom_greedy"]["rouge1"]["f1"] for r in results]),
            "avg_rouge2_f1": avg([r["custom_greedy"]["rouge2"]["f1"] for r in results]),
            "avg_rougel_f1": avg([r["custom_greedy"]["rougel"]["f1"] for r in results]),
            "avg_entity_precision": avg([r["custom_greedy"]["entity"]["entity_precision"] for r in results]),
            "avg_entity_recall": avg([r["custom_greedy"]["entity"]["entity_recall"] for r in results]),
            "avg_numeric_rate": avg([r["custom_greedy"]["numeric"]["numeric_consistency_rate"] for r in results]),
            "avg_repetition_rate": avg([r["custom_greedy"]["repetition_rate"] for r in results]),
        },
        "custom_beam": {
            "avg_latency_ms": avg([r["custom_beam"]["latency_ms"] for r in results]),
            "avg_rouge1_f1": avg([r["custom_beam"]["rouge1"]["f1"] for r in results]),
            "avg_rouge2_f1": avg([r["custom_beam"]["rouge2"]["f1"] for r in results]),
            "avg_rougel_f1": avg([r["custom_beam"]["rougel"]["f1"] for r in results]),
            "avg_entity_precision": avg([r["custom_beam"]["entity"]["entity_precision"] for r in results]),
            "avg_entity_recall": avg([r["custom_beam"]["entity"]["entity_recall"] for r in results]),
            "avg_numeric_rate": avg([r["custom_beam"]["numeric"]["numeric_consistency_rate"] for r in results]),
            "avg_repetition_rate": avg([r["custom_beam"]["repetition_rate"] for r in results]),
        },
    }

    out_payload = {
        "date": "2026-10-29",
        "device": str(pt_device),
        "hardware": "Apple Silicon (MPS acceleration / CPU fallback)",
        "summary_averages": summary_metrics,
        "detailed_results": results,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/model_comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Metric':<25} | {'DistilBART-CNN-12-6':<20} | {'Custom Seq2Seq (Greedy)':<25} | {'Custom Seq2Seq (Beam)':<25}"
    print(header)
    print("-" * len(header))
    m = summary_metrics
    print(f"{'ROUGE-1 F1':<25} | {m['distilbart']['avg_rouge1_f1']:<20.4f} | {m['custom_greedy']['avg_rouge1_f1']:<25.4f} | {m['custom_beam']['avg_rouge1_f1']:<25.4f}")
    print(f"{'ROUGE-2 F1':<25} | {m['distilbart']['avg_rouge2_f1']:<20.4f} | {m['custom_greedy']['avg_rouge2_f1']:<25.4f} | {m['custom_beam']['avg_rouge2_f1']:<25.4f}")
    print(f"{'ROUGE-L F1':<25} | {m['distilbart']['avg_rougel_f1']:<20.4f} | {m['custom_greedy']['avg_rougel_f1']:<25.4f} | {m['custom_beam']['avg_rougel_f1']:<25.4f}")
    print(f"{'Entity Precision':<25} | {m['distilbart']['avg_entity_precision']:<20.4f} | {m['custom_greedy']['avg_entity_precision']:<25.4f} | {m['custom_beam']['avg_entity_precision']:<25.4f}")
    print(f"{'Entity Recall':<25} | {m['distilbart']['avg_entity_recall']:<20.4f} | {m['custom_greedy']['avg_entity_recall']:<25.4f} | {m['custom_beam']['avg_entity_recall']:<25.4f}")
    print(f"{'Numeric Consistency':<25} | {m['distilbart']['avg_numeric_rate']:<20.4f} | {m['custom_greedy']['avg_numeric_rate']:<25.4f} | {m['custom_beam']['avg_numeric_rate']:<25.4f}")
    print(f"{'Repetition Rate':<25} | {m['distilbart']['avg_repetition_rate']:<20.4f} | {m['custom_greedy']['avg_repetition_rate']:<25.4f} | {m['custom_beam']['avg_repetition_rate']:<25.4f}")
    print(f"{'Avg Latency (ms)':<25} | {m['distilbart']['avg_latency_ms']:<20.1f} | {m['custom_greedy']['avg_latency_ms']:<25.1f} | {m['custom_beam']['avg_latency_ms']:<25.1f}")
    print("=" * 80)
    print(f"Results saved cleanly to {out_path}")


if __name__ == "__main__":
    main()
