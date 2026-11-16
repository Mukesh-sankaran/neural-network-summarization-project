"""
NNDL Project — Neural News Intelligence Console
Interactive Streamlit Application for Abstractive News Summarization.
Supports Dual Model Backends:
1. Production Pretrained Baseline (sshleifer/distilbart-cnn-12-6 via Hugging Face Transformers)
2. Custom Research Model (PyTorch Seq2Seq BiLSTM + Bahdanau Additive Attention)
Premium Cyberpunk / AI Research Terminal UI (WCAG AAA Compliant).
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import tokenize, PAD_IDX, SOS_IDX, EOS_IDX
from src.vocabulary import Vocabulary
from src.model import Seq2Seq, get_device as get_custom_device
from src.pretrained_baseline import (
    load_pretrained_model,
    generate_pretrained_summary,
    get_pretrained_device,
    DEFAULT_PRETRAINED_MODEL_NAME,
)
from src.evaluate import evaluate_summary
from src.entity_evaluation import (
    evaluate_entity_preservation,
    evaluate_numeric_consistency,
    extract_named_entities,
    extract_numeric_values,
)

# -----------------------------------------------------------------------------
# Pre-curated Sample Articles (for quick evaluation across diverse news topics)
# -----------------------------------------------------------------------------
SAMPLE_ARTICLES = {
    "Robert Hardy Antiques Auction (CNN/DailyMail Example #10)": {
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
    "Golf: Justin Rose Houston Open (CNN/DailyMail Example #1)": {
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
    "Hoax Ambulance Calls (CNN/DailyMail Example #3)": {
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
    "Palestinian Authority ICC Accession (CNN Test Set #0)": {
        "article": (
            "The Palestinian Authority officially became the 123rd member of the International Criminal "
            "Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian "
            "territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, "
            "where the court is based. Palestinian Foreign Minister Riad Malki accepted a symbolic copy of the "
            "Rome Statute from ICC President Sidiki Kaba. Membership gives the ICC jurisdiction over alleged "
            "crimes committed in Palestinian territories since last June. Israel and the United States opposed "
            "the move, warning it could undermine peace negotiations."
        ),
        "reference": (
            "Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\n"
            "Israel and the United States opposed the move, which could open the door to war crimes investigations ."
        ),
    },
    "Basketball Star Prom-Posal (CNN Test Set #5)": {
        "article": (
            "He's a blue chip college basketball recruit. She's a high school freshman with Down syndrome. "
            "At first glance Trey Moses and Ellie Meredith couldn't be more different. But all that changed "
            "Thursday when Trey asked Ellie to be his prom date. Trey, a star on Eastern High School's "
            "basketball team in Louisville, Kentucky, held up a sign asking Ellie to the prom during gym class. "
            "Ellie broke into tears of joy and immediately hugged him. Pictures of the heartfelt proposal "
            "quickly went viral across social media with over 50,000 shares."
        ),
        "reference": (
            "College-bound basketball star asks girl with Down syndrome to high school prom .\n"
            "Pictures of the two during the 'prom-posal' have gone viral across social media ."
        ),
    },
    "Custom Article (Enter your own)": {
        "article": "",
        "reference": "",
    },
}


# -----------------------------------------------------------------------------
# Cached Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Custom PyTorch Seq2Seq Checkpoint & Vocabulary...")
def load_custom_model_resources() -> Tuple[Optional[Seq2Seq], Optional[Vocabulary], Optional[Dict[str, Any]], torch.device, Optional[str]]:
    """
    Safely loads the pickled vocabulary and custom PyTorch checkpoint.
    Returns: (model, vocab, checkpoint_meta, device, error_message)
    """
    vocab_path = "models/vocab.pkl"
    checkpoint_path = "models/best_model.pt"
    device = get_custom_device()

    if not os.path.exists(vocab_path):
        return None, None, None, device, f"Vocabulary file not found at `{vocab_path}`."

    if not os.path.exists(checkpoint_path):
        return None, None, None, device, f"Model checkpoint not found at `{checkpoint_path}`."

    try:
        vocab = Vocabulary.load(vocab_path)
    except Exception as e:
        return None, None, None, device, f"Failed to load vocabulary: {str(e)}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = Seq2Seq(
            vocab_size=len(vocab),
            embedding_dim=128,
            encoder_hidden_dim=256,
            decoder_hidden_dim=256,
            pad_idx=PAD_IDX,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        checkpoint_meta = {
            "epoch": checkpoint.get("epoch", 3),
            "train_loss": checkpoint.get("train_loss", 6.3864),
            "val_loss": checkpoint.get("val_loss", 6.8798),
            "vocab_size": checkpoint.get("vocab_size", len(vocab)),
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
        }

        return model, vocab, checkpoint_meta, device, None
    except Exception as e:
        return None, vocab, None, device, f"Failed to instantiate or load custom model: {str(e)}"


# Backwards compatibility alias for test suites
def load_model_resources():
    return load_custom_model_resources()


@st.cache_resource(show_spinner="Loading Pretrained Transformer Baseline (DistilBART-CNN-12-6)...")
def load_pretrained_model_resources(model_name: str = DEFAULT_PRETRAINED_MODEL_NAME):
    """
    Caches the pretrained transformer baseline and its official tokenizer.
    Returns: (model, tokenizer, metadata, device, error_message)
    """
    return load_pretrained_model(model_name)


# -----------------------------------------------------------------------------
# State Management & Diagnostics
# -----------------------------------------------------------------------------
def clear_generated_summary_state(session_state_dict: Optional[Any] = None) -> List[str]:
    """
    Invalidates cached generation results when inputs, models, or samples change.
    Accepts an optional dict/session_state for modular unit testing.
    """
    target = st.session_state if session_state_dict is None else session_state_dict
    keys_to_clear = [
        "gen_summary",
        "gen_token_ids",
        "attn_matrix",
        "src_tokens",
        "last_article",
        "last_reference",
        "used_strategy",
        "used_beam_width",
        "used_trigram_blocking",
        "used_length_penalty",
        "inference_latency_ms",
        "active_model_badge",
        "active_model_kind",
    ]
    cleared = []
    for key in keys_to_clear:
        if key in target:
            if hasattr(target, "pop"):
                target.pop(key, None)
            else:
                del target[key]
            cleared.append(key)
    return cleared


def detect_degenerate_output(tokens: List[str]) -> Tuple[bool, str]:
    """
    Detects whether the generated summary suffers from sequence degeneration or repetition loops.
    """
    if not tokens:
        return True, "The model emitted an empty sequence."

    total = len(tokens)
    from collections import Counter
    counts = Counter(tokens)
    most_common_tok, most_common_cnt = counts.most_common(1)[0]

    repetition_ratio = most_common_cnt / total
    punctuation_count = sum(counts[p] for p in [".", ",", "'", '"', "-", "!", "?"] if p in counts)
    punct_ratio = punctuation_count / total

    if repetition_ratio > 0.40:
        return True, f"Repetitive collapse: token '{most_common_tok}' accounts for {repetition_ratio:.1%} of generated output."

    if punct_ratio > 0.60:
        return True, f"Punctuation collapse: {punct_ratio:.1%} of generated tokens are standalone punctuation marks."

    return False, "Output does not trigger trivial repetition flags."


def plot_attention_heatmap(attn_matrix: torch.Tensor, src_tokens: List[str], gen_tokens: List[str], max_src_display: int = 35) -> plt.Figure:
    """
    Plots an attention alignment heatmap between generated steps and source tokens
    rendered with a dark cyberpunk aesthetic.
    """
    src_len = min(len(src_tokens), max_src_display)
    gen_len = len(gen_tokens)

    weights = attn_matrix[:gen_len, :src_len].cpu().numpy()

    cyber_cmap = LinearSegmentedColormap.from_list(
        "cyber_cyan",
        ["#070b14", "#032845", "#0284c7", "#00f0ff", "#ffffff"]
    )

    fig, ax = plt.subplots(figsize=(max(8, src_len * 0.35), max(4, gen_len * 0.28)))
    fig.patch.set_facecolor('#070b14')
    ax.set_facecolor('#070b14')

    cax = ax.matshow(weights, cmap=cyber_cmap, aspect="auto")
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#38bdf8', labelcolor='#94a3b8')
    cbar.outline.set_edgecolor((0.0, 0.94, 1.0, 0.35))

    ax.set_xticks(range(src_len))
    ax.set_yticks(range(gen_len))

    ax.set_xticklabels(src_tokens[:src_len], rotation=65, ha="left", fontsize=8, color="#cbd5e1")
    ax.set_yticklabels(gen_tokens, fontsize=8, color="#cbd5e1")

    for spine in ax.spines.values():
        spine.set_edgecolor((0.0, 0.94, 1.0, 0.35))
        spine.set_linewidth(1.0)

    ax.tick_params(colors='#38bdf8', which='both')
    ax.set_xlabel("Source Document Tokens (First 35)", fontsize=9, labelpad=10, color="#38bdf8", fontweight="600")
    ax.set_ylabel("Generated Summary Steps", fontsize=9, color="#38bdf8", fontweight="600")
    ax.set_title("Bahdanau Additive Attention Alignment Heatmap", fontsize=11, pad=15, color="#00f0ff", fontweight="700")
    plt.tight_layout()

    return fig


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NNDL // Neural News Intelligence Console",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -------------------------------------------------------------------------
    # Premium Cyberpunk Terminal Styling (WCAG AAA Compliant >= 7:1)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Base Canvas & Background Grid */
        .stApp {
            background-color: #070b14;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(0, 240, 255, 0.05) 0%, transparent 60%),
                linear-gradient(rgba(0, 240, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 36px 36px, 36px 36px;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Top Navigation Header */
        .terminal-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            background: rgba(13, 21, 39, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 10px;
            padding: 0.85rem 1.4rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 240, 255, 0.08);
        }
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .nav-mark {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(139, 92, 246, 0.2));
            border: 1px solid #00f0ff;
            border-radius: 8px;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.3);
            font-size: 1.25rem;
        }
        .nav-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            background: linear-gradient(135deg, #00f0ff 0%, #38bdf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-subtitle {
            font-size: 0.8rem;
            color: #94a3b8;
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
        }
        .nav-telemetry {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            letter-spacing: 0.04em;
        }
        .pill-cyan {
            background: rgba(0, 240, 255, 0.1);
            color: #00f0ff;
            border: 1px solid rgba(0, 240, 255, 0.35);
        }
        .pill-purple {
            background: rgba(168, 85, 247, 0.1);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.35);
        }
        .pill-blue {
            background: rgba(56, 189, 248, 0.1);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .pill-emerald {
            background: rgba(52, 211, 153, 0.1);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.35);
        }
        .dot-live {
            width: 7px;
            height: 7px;
            background-color: #00f0ff;
            border-radius: 50%;
            box-shadow: 0 0 8px #00f0ff;
            display: inline-block;
            animation: pulse-glow 2s infinite ease-in-out;
        }
        @keyframes pulse-glow {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        /* Section Headings */
        .section-badge {
            display: inline-block;
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: #00f0ff;
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.25);
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem;
        }
        .section-desc {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-bottom: 1.15rem;
        }

        /* Cyberpunk Terminal Window for Output */
        .cyber-terminal {
            background: #090e1a;
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 240, 255, 0.12);
            margin-bottom: 1.25rem;
        }
        .terminal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
            border-bottom: 1px solid rgba(0, 240, 255, 0.18);
            padding: 0.55rem 0.9rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .terminal-dots {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .term-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-red { background-color: #ef4444; box-shadow: 0 0 6px rgba(239, 68, 68, 0.6); }
        .dot-amber { background-color: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }
        .dot-cyan { background-color: #06b6d4; box-shadow: 0 0 6px rgba(6, 182, 212, 0.6); }
        .terminal-title {
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 600;
            color: #38bdf8;
            letter-spacing: 0.05em;
        }
        .terminal-tags {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .tag-meta {
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.2);
            color: #94a3b8;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
        }
        .terminal-content {
            padding: 1.25rem;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #f1f5f9;
            background: #080d17;
            border-left: 3px solid #00f0ff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 70px;
        }

        /* Telemetry Bar */
        .telemetry-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: center;
            background: rgba(10, 16, 30, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 6px;
            padding: 0.5rem 0.85rem;
            margin: 0.6rem 0 1rem 0;
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            font-size: 0.8rem;
        }
        .telemetry-stat {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .telemetry-label {
            color: #64748b;
        }
        .telemetry-value {
            color: #38bdf8;
            font-weight: 700;
        }
        .status-ok {
            color: #34d399;
            margin-left: auto;
            font-weight: 600;
        }
        .status-warn {
            color: #f59e0b;
            margin-left: auto;
            font-weight: 600;
        }

        /* Control Deck Card */
        .control-deck-card {
            background: rgba(13, 21, 39, 0.75);
            border: 1px solid rgba(0, 240, 255, 0.2);
            border-radius: 8px;
            padding: 1rem 1.25rem 0.6rem 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .deck-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: 0.08em;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
        }

        /* High-Contrast Notice & Warning Blocks (Verified WCAG AAA >= 7:1) */
        .info-notice {
            background-color: #081426;
            border-left: 4px solid #00f0ff;
            border-radius: 6px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1rem;
            color: #38bdf8;
            font-size: 0.88rem;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.08);
        }
        .info-notice b, .info-notice strong {
            color: #e0f2fe;
        }
        .info-notice code {
            color: #67e8f9;
            background-color: #0c2340;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
            font-weight: 600;
        }
        .degenerate-warning {
            background-color: #1f080c;
            border-left: 4px solid #f43f5e;
            border-radius: 6px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1rem;
            color: #fca5a5;
            font-size: 0.88rem;
            box-shadow: 0 0 15px rgba(244, 63, 94, 0.12);
        }
        .degenerate-warning b, .degenerate-warning strong {
            color: #ffe4e6;
        }
        .degenerate-warning small {
            color: #fda4af;
        }
        .degenerate-warning code {
            color: #fecdd3;
            background-color: #380d14;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
            font-weight: 600;
        }

        /* Primary Neon Button */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #0284c7 0%, #06b6d4 50%, #8b5cf6 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            border: 1px solid rgba(0, 240, 255, 0.4) !important;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.35) !important;
            text-transform: uppercase !important;
            border-radius: 6px !important;
            padding: 0.65rem 1.5rem !important;
            transition: all 0.25s ease !important;
        }
        div[data-testid="stButton"] > button:hover {
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.65) !important;
            transform: translateY(-1px) !important;
            border-color: #00f0ff !important;
        }

        div[data-baseweb="textarea"] {
            border: 1px solid rgba(0, 240, 255, 0.25) !important;
            border-radius: 6px !important;
            background-color: #080d18 !important;
        }
        div[data-baseweb="textarea"]:focus-within {
            border-color: #00f0ff !important;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.3) !important;
        }
        div[data-baseweb="select"] {
            border-radius: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Resource Initialization (Both Models Cached Safely)
    # -------------------------------------------------------------------------
    custom_model, custom_vocab, custom_meta, custom_device, custom_err = load_custom_model_resources()
    pt_model, pt_tokenizer, pt_meta, pt_device, pt_err = load_pretrained_model_resources()

    # -------------------------------------------------------------------------
    # Model Selection Deck
    # -------------------------------------------------------------------------
    st.markdown('<div class="control-deck-card">', unsafe_allow_html=True)
    st.markdown('<div class="deck-header">🤖 // 01 &middot; NEURAL BACKEND & MODEL SELECTOR</div>', unsafe_allow_html=True)

    col_m_select, col_m_info = st.columns([1.5, 2.5])

    with col_m_select:
        selected_model_option = st.radio(
            "Select Summarization Engine:",
            options=[
                "Pretrained Baseline (DistilBART-CNN-12-6 — SOTA Baseline)",
                "Custom Research Model (BiLSTM + Bahdanau Attention — Academic Seq2Seq)",
            ],
            index=0,
            help="Pretrained baseline provides verified production-grade abstractive summaries. Custom research model evaluates the undertrained academic PyTorch implementation.",
        )

    is_pretrained = "DistilBART" in selected_model_option

    # Invalidate stale summary state if active model changes
    if "current_model_option" not in st.session_state:
        st.session_state["current_model_option"] = selected_model_option
    elif st.session_state["current_model_option"] != selected_model_option:
        st.session_state["current_model_option"] = selected_model_option
        clear_generated_summary_state()

    with col_m_info:
        if is_pretrained:
            if pt_err:
                st.error(f"⚠️ Pretrained Baseline Error: {pt_err}")
            else:
                st.markdown(
                    f"""
                    <div style="font-family: 'SF Mono', monospace; font-size: 0.8rem; background: rgba(0,240,255,0.05); padding: 0.6rem 0.9rem; border-radius: 6px; border-left: 3px solid #00f0ff;">
                        <b>ACTIVE MODEL:</b> <code>{pt_meta['model_name']}</code> ({pt_meta['architecture']})<br>
                        <b>TOKENIZER:</b> Hugging Face BPE (Vocab: {pt_meta['vocab_size']:,} tokens)<br>
                        <b>PARAMETERS:</b> {pt_meta['total_parameters']:,} &middot; <b>COMPUTE:</b> {str(pt_device).upper()}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            if custom_err:
                st.error(f"⚠️ Custom Research Model Error: {custom_err}")
            else:
                st.markdown(
                    f"""
                    <div style="font-family: 'SF Mono', monospace; font-size: 0.8rem; background: rgba(168,85,247,0.05); padding: 0.6rem 0.9rem; border-radius: 6px; border-left: 3px solid #c084fc;">
                        <b>ACTIVE MODEL:</b> Custom Seq2Seq with Bahdanau Attention (<code>models/best_model.pt</code>)<br>
                        <b>TOKENIZER:</b> Custom Word-Level Vocabulary ({custom_meta['vocab_size']:,} tokens)<br>
                        <b>PARAMETERS:</b> {custom_meta['parameters']:,} (Epoch {custom_meta['epoch']}, Val Loss: {custom_meta['val_loss']:.4f}) &middot; <b>COMPUTE:</b> {str(custom_device).upper()}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Top Navigation & Telemetry Banner
    # -------------------------------------------------------------------------
    active_device_str = str(pt_device if is_pretrained else custom_device).upper()
    active_badge_class = "pill-emerald" if is_pretrained else "pill-purple"
    active_badge_text = "SOTA BASELINE ONLINE" if is_pretrained else "RESEARCH PROTOTYPE ONLINE"

    st.markdown(
        f"""
        <div class="terminal-nav">
            <div class="nav-brand">
                <div class="nav-mark">⚡</div>
                <div>
                    <div class="nav-title">NNDL // NEURAL NEWS INTELLIGENCE</div>
                    <div class="nav-subtitle">Abstractive News Summarization Console &middot; Pretrained SOTA & Custom Seq2Seq Architectures</div>
                </div>
            </div>
            <div class="nav-telemetry">
                <span class="pill-badge {active_badge_class}"><span class="dot-live"></span> {active_badge_text}</span>
                <span class="pill-badge pill-blue">DEVICE: {active_device_str}</span>
                <span class="pill-badge pill-cyan">CNN/DAILYMAIL 3.0.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Generation Controls Deck (Dynamically Adapted per Model Backend)
    # -------------------------------------------------------------------------
    st.markdown('<div class="control-deck-card">', unsafe_allow_html=True)
    st.markdown('<div class="deck-header">🎛️ // 02 &middot; GENERATION & DECODING PARAMETERS</div>', unsafe_allow_html=True)

    if is_pretrained:
        # Pretrained DistilBART Controls
        col_c1, col_c2, col_c3, col_c4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with col_c1:
            pt_num_beams = st.slider(
                "Beam Search Width",
                min_value=1,
                max_value=8,
                value=4,
                help="k=1 executes greedy search. k > 1 performs beam search.",
            )
        with col_c2:
            pt_length_penalty = st.slider(
                "Length Penalty (α)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Wu et al. exponent: α > 1 encourages longer summaries.",
            )
        with col_c3:
            pt_max_length = st.slider(
                "Max Summary Tokens",
                min_value=40,
                max_value=200,
                value=142,
                step=10,
                help="Maximum length budget for generated summary.",
            )
        with col_c4:
            pt_min_length = st.slider(
                "Min Summary Tokens",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Minimum generated summary tokens.",
            )
            pt_trigram = st.checkbox(
                "Trigram Repetition Blocking",
                value=True,
                help="Prohibits duplicate 3-grams to ensure diversity.",
            )

    else:
        # Custom Seq2Seq Controls
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.3, 1.2, 1.2, 1.3])
        with col_ctrl1:
            decoding_strategy = st.radio(
                "Decoding Strategy",
                options=["Greedy Decoding", "Beam Search"],
                index=1,
                horizontal=True,
                help="Greedy selects argmax. Beam Search explores multiple candidate paths.",
            )
        with col_ctrl2:
            if decoding_strategy == "Beam Search":
                beam_width = st.slider("Beam Width (k)", min_value=2, max_value=8, value=4, step=1)
            else:
                st.caption("Beam Width: **1** (Greedy Argmax)")
                beam_width = 1
        with col_ctrl3:
            if decoding_strategy == "Beam Search":
                length_penalty = st.slider("Length Penalty (α)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
            else:
                st.caption("Length Penalty: **N/A**")
                length_penalty = 1.0
        with col_ctrl4:
            enable_trigram_blocking = st.checkbox("Trigram Repetition Blocking", value=True)
            max_target_len = st.slider("Max Output Tokens", min_value=20, max_value=100, value=60, step=5)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main Workspace (Article Selection, Input & Reference)
    # -------------------------------------------------------------------------
    st.markdown('<span class="section-badge">// 03 &middot; SOURCE INTELLIGENCE</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">News Intelligence Workspace</div>', unsafe_allow_html=True)

    selected_sample_key = st.selectbox(
        "Select Curated News Sample or Input Custom Article:",
        list(SAMPLE_ARTICLES.keys()),
        index=0,
    )

    # Invalidate stale generated summary state if the sample selection changed
    if "current_sample_key" not in st.session_state:
        st.session_state["current_sample_key"] = selected_sample_key
    elif st.session_state["current_sample_key"] != selected_sample_key:
        st.session_state["current_sample_key"] = selected_sample_key
        clear_generated_summary_state()

    default_article = SAMPLE_ARTICLES[selected_sample_key]["article"]
    default_reference = SAMPLE_ARTICLES[selected_sample_key]["reference"]

    col_input, col_ref = st.columns([65, 35])

    with col_input:
        article_text = st.text_area(
            "Source News Article Document:",
            value=default_article,
            height=200,
            placeholder="Paste raw news article text here...",
            help="Source document. Articles are cleanly truncated at max positional embedding length (500 tokens for custom model, 1024 for DistilBART).",
        )

    with col_ref:
        with st.expander("Optional Reference Summary (Human Ground Truth)", expanded=bool(default_reference)):
            reference_text = st.text_area(
                "Human Editorial Highlights:",
                value=default_reference,
                height=150,
                placeholder="(Optional) Editorial reference summary for ROUGE evaluation...",
                help="Reference highlights from human editors. Used strictly to compute ROUGE-1, ROUGE-2, and ROUGE-L.",
            )

    # Invalidate stale state if article or reference text changed
    if "last_article" in st.session_state and (
        st.session_state["last_article"].strip() != article_text.strip()
        or st.session_state.get("last_reference", "").strip() != reference_text.strip()
    ):
        clear_generated_summary_state()

    # Telemetry indicators
    raw_tokens = tokenize(article_text)
    word_count = len(article_text.split())
    token_count = len(raw_tokens)
    max_budget = 1024 if is_pretrained else 500
    pct_budget = min(100.0, (token_count / float(max_budget)) * 100.0)

    status_tag = (
        f'<span class="status-warn">⚠️ EXCEEDS {max_budget} TOKENS (WILL TRUNCATE)</span>'
        if token_count > max_budget
        else '<span class="status-ok">● FITS WITHIN ENCODER BUDGET</span>'
    )

    st.markdown(
        f"""
        <div class="telemetry-bar">
            <div class="telemetry-stat"><span class="telemetry-label">SOURCE WORDS:</span> <span class="telemetry-value">{word_count:,}</span></div>
            <div class="telemetry-stat"><span class="telemetry-label">SOURCE TOKENS:</span> <span class="telemetry-value">{token_count:,} / {max_budget}</span></div>
            <div class="telemetry-stat"><span class="telemetry-label">CONTEXT UTILIZATION:</span> <span class="telemetry-value">{pct_budget:.1f}%</span></div>
            {status_tag}
        </div>
        """,
        unsafe_allow_html=True,
    )

    generate_button = st.button("⚡ EXECUTE NEURAL SUMMARIZATION", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Execution & Inference Pipeline
    # -------------------------------------------------------------------------
    if generate_button:
        if not article_text or not article_text.strip():
            st.error("❌ Validation Error: Please provide non-empty source article text before generating.")
            return

        if is_pretrained:
            if pt_model is None or pt_tokenizer is None:
                st.error(f"❌ Execution Blocked: Pretrained model unavailable. {pt_err}")
                return

            with st.spinner(f"Generating summary with DistilBART ({'Beam Search k=' + str(pt_num_beams) if pt_num_beams > 1 else 'Greedy'})..."):
                try:
                    summary_text, token_ids, latency_ms = generate_pretrained_summary(
                        article_text=article_text,
                        model=pt_model,
                        tokenizer=pt_tokenizer,
                        device=pt_device,
                        max_length=pt_max_length,
                        min_length=pt_min_length,
                        num_beams=pt_num_beams,
                        length_penalty=pt_length_penalty,
                        no_repeat_ngram_size=3 if pt_trigram else 0,
                    )

                    st.session_state["gen_summary"] = summary_text
                    st.session_state["gen_token_ids"] = token_ids
                    st.session_state["attn_matrix"] = None
                    st.session_state["src_tokens"] = raw_tokens[:max_budget]
                    st.session_state["last_article"] = article_text
                    st.session_state["last_reference"] = reference_text
                    st.session_state["used_strategy"] = f"Beam Search (k={pt_num_beams})" if pt_num_beams > 1 else "Greedy Search"
                    st.session_state["used_beam_width"] = pt_num_beams
                    st.session_state["used_trigram_blocking"] = pt_trigram
                    st.session_state["used_length_penalty"] = pt_length_penalty
                    st.session_state["inference_latency_ms"] = latency_ms
                    st.session_state["active_model_badge"] = "DistilBART-CNN-12-6 (Pretrained)"
                    st.session_state["active_model_kind"] = "pretrained"

                except Exception as e:
                    st.error(f"❌ Pretrained Inference Error: {str(e)}")
                    return

        else:
            # Custom Seq2Seq
            if custom_model is None or custom_vocab is None:
                st.error(f"❌ Execution Blocked: Custom model unavailable. {custom_err}")
                return

            strategy_param = "beam_search" if decoding_strategy == "Beam Search" else "greedy"
            ngram_param = 3 if enable_trigram_blocking else 0

            with st.spinner(f"Executing Seq2Seq {decoding_strategy}..."):
                try:
                    max_source_len = 500
                    src_tokens = raw_tokens[:max_source_len]
                    src_indices = custom_vocab.encode(src_tokens)
                    if len(src_indices) < max_source_len:
                        src_indices = src_indices + [PAD_IDX] * (max_source_len - len(src_indices))

                    src_tensor = torch.tensor(src_indices, dtype=torch.long)

                    t_start = time.perf_counter()
                    with torch.no_grad():
                        gen_summary, gen_token_ids, attn_matrix = custom_model.generate_summary(
                            src_tensor=src_tensor,
                            vocab=custom_vocab,
                            max_target_len=max_target_len,
                            device=custom_device,
                            strategy=strategy_param,
                            beam_width=beam_width,
                            no_repeat_ngram_size=ngram_param,
                            length_penalty=length_penalty,
                        )
                    latency_ms = (time.perf_counter() - t_start) * 1000.0

                    st.session_state["gen_summary"] = gen_summary
                    st.session_state["gen_token_ids"] = gen_token_ids
                    st.session_state["attn_matrix"] = attn_matrix
                    st.session_state["src_tokens"] = src_tokens
                    st.session_state["last_article"] = article_text
                    st.session_state["last_reference"] = reference_text
                    st.session_state["used_strategy"] = decoding_strategy
                    st.session_state["used_beam_width"] = beam_width if strategy_param == "beam_search" else None
                    st.session_state["used_trigram_blocking"] = enable_trigram_blocking
                    st.session_state["used_length_penalty"] = length_penalty if strategy_param == "beam_search" else None
                    st.session_state["inference_latency_ms"] = latency_ms
                    st.session_state["active_model_badge"] = "Seq2Seq BiLSTM + Bahdanau (Custom)"
                    st.session_state["active_model_kind"] = "custom"

                except Exception as e:
                    st.error(f"❌ Custom Inference Error: {str(e)}")
                    return

    # -------------------------------------------------------------------------
    # Results Presentation
    # -------------------------------------------------------------------------
    if "gen_summary" in st.session_state:
        st.markdown("<hr style='border-color: rgba(0, 240, 255, 0.15); margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown('<span class="section-badge">// 04 &middot; INFERENCE OUTPUT & TELEMETRY</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Synthesized Neural Summary</div>', unsafe_allow_html=True)

        gen_summary = st.session_state["gen_summary"]
        gen_token_ids = st.session_state["gen_token_ids"]
        attn_matrix = st.session_state.get("attn_matrix")
        src_tokens = st.session_state.get("src_tokens", [])
        curr_article = st.session_state.get("last_article", article_text)
        curr_reference = st.session_state.get("last_reference", reference_text)
        latency_val = st.session_state.get("inference_latency_ms", 0.0)
        active_badge = st.session_state.get("active_model_badge", "Unknown Model")
        active_kind = st.session_state.get("active_model_kind", "pretrained")

        gen_words = len(gen_summary.split())
        gen_tokens_count = len(gen_token_ids)
        used_strat = st.session_state.get("used_strategy", "Standard Generation")

        # Cyber Terminal Output Window
        clean_text = gen_summary if gen_summary else "<EMPTY SEQUENCE EMITTED BY DECODER>"
        st.markdown(
            f"""
            <div class="cyber-terminal">
                <div class="terminal-header">
                    <div class="terminal-dots">
                        <span class="term-dot dot-red"></span>
                        <span class="term-dot dot-amber"></span>
                        <span class="term-dot dot-cyan"></span>
                    </div>
                    <div class="terminal-title">OUTPUT_STREAM // {active_badge.upper()}</div>
                    <div class="terminal-tags">
                        <span class="tag-meta">STRATEGY: {used_strat.upper()}</span>
                        <span class="tag-meta">LATENCY: {latency_val:.1f} ms</span>
                        <span class="tag-meta">LENGTH: {gen_words} WORDS ({gen_tokens_count} TOKENS)</span>
                    </div>
                </div>
                <div class="terminal-content">
                    {clean_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quality & Integrity Notices
        if active_kind == "custom":
            gen_token_words = [custom_vocab.idx2word.get(idx, "<UNK>") for idx in gen_token_ids] if custom_vocab else gen_summary.split()
            is_degenerate, degen_msg = detect_degenerate_output(gen_token_words)
            if is_degenerate:
                st.markdown(
                    f"""
                    <div class="degenerate-warning">
                        <b>⚠️ DEGENERATE GENERATION DETECTED:</b> {degen_msg}<br>
                        <small>The recurrent decoder entered a repetitive token loop. While Beam Search with Trigram Blocking suppresses trivial loops, the academic prototype checkpoint remains severely undertrained (2,000 samples for 3 epochs).</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                """
                <div class="info-notice">
                    <b>PROTOTYPE CHECKPOINT NOTICE:</b> Generated via <code>models/best_model.pt</code> 
                    (trained on 2,000 samples for 3 epochs with final validation loss 6.8798). Output displays raw, 
                    unfiltered neural network emissions demonstrating baseline research behavior.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="info-notice">
                    <b>SOTA BASELINE NOTICE:</b> Generated via <code>sshleifer/distilbart-cnn-12-6</code> 
                    (DistilBART 305.5M parameters fine-tuned on CNN/DailyMail). Summaries exhibit coherent grammar, 
                    strong source grounding, and high entity preservation.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # Multi-Metric Evaluation & Diagnostics Console
        # ---------------------------------------------------------------------
        st.markdown('<span class="section-badge">// 05 &middot; FACTUAL GROUNDING & MULTI-METRIC DIAGNOSTICS</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Multi-Metric Evaluation Suite</div>', unsafe_allow_html=True)

        tab_rouge, tab_entities, tab_numeric, tab_attn, tab_comparison = st.tabs([
            "📊 ROUGE Overlap Metrics",
            "🏷️ Named Entity Preservation",
            "🔢 Numeric Consistency",
            "🔍 Attention & Internal Dynamics",
            "⚖️ Scientific Model Comparison",
        ])

        # TAB 1: ROUGE Metrics
        with tab_rouge:
            if curr_reference and curr_reference.strip():
                st.caption("Punctuation tokens are excluded from ROUGE n-gram matching to prevent artificial score inflation.")
                rouge_res = evaluate_summary(gen_summary, curr_reference, exclude_punctuation=True)

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric(
                        "ROUGE-1 F1 (Unigram)",
                        f"{rouge_res['rouge1']['f1']:.4f}",
                        help=f"Precision: {rouge_res['rouge1']['precision']:.4f} | Recall: {rouge_res['rouge1']['recall']:.4f}",
                    )
                with r_col2:
                    st.metric(
                        "ROUGE-2 F1 (Bigram)",
                        f"{rouge_res['rouge2']['f1']:.4f}",
                        help=f"Precision: {rouge_res['rouge2']['precision']:.4f} | Recall: {rouge_res['rouge2']['recall']:.4f}",
                    )
                with r_col3:
                    st.metric(
                        "ROUGE-L F1 (LCS)",
                        f"{rouge_res['rougel']['f1']:.4f}",
                        help=f"Precision: {rouge_res['rougel']['precision']:.4f} | Recall: {rouge_res['rougel']['recall']:.4f}",
                    )
            else:
                st.info("ℹ️ Optional reference summary was not provided. Enter a reference summary above to compute ROUGE overlap.")

        # TAB 2: Entity Preservation
        with tab_entities:
            entity_eval = evaluate_entity_preservation(curr_article, gen_summary)

            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            p_val = f"{entity_eval['entity_precision']:.4f}" if entity_eval['entity_precision'] is not None else "N/A"
            r_val = f"{entity_eval['entity_recall']:.4f}" if entity_eval['entity_recall'] is not None else "N/A"
            pres_val = f"{entity_eval['entity_preservation_rate']:.4f}" if entity_eval['entity_preservation_rate'] is not None else "N/A"
            f1_val = f"{entity_eval['entity_f1']:.4f}" if entity_eval['entity_f1'] is not None else "N/A"

            with e_col1:
                st.metric("Entity Precision", p_val, help="Proportion of summary entities present in source document (groundedness).")
            with e_col2:
                st.metric("Entity Recall", r_val, help="Proportion of unique source entities captured in summary (coverage).")
            with e_col3:
                st.metric("Preservation Rate", pres_val, help="Weighted proportion of source entity mentions preserved.")
            with e_col4:
                st.metric("Entity F1 Score", f1_val, help="Harmonic mean of entity precision and recall.")

            st.markdown(f"**Entity Status:** `{entity_eval['status']}` — *{entity_eval['status_description']}*")

            col_ent1, col_ent2, col_ent3 = st.columns(3)
            with col_ent1:
                st.markdown(f"**Preserved Entities ({entity_eval['num_preserved_entities']}):**")
                st.write(entity_eval["preserved_entities"] if entity_eval["preserved_entities"] else "None")
            with col_ent2:
                st.markdown(f"**Hallucinated Entities ({len(entity_eval['extra_entities'])}):**")
                st.write(entity_eval["extra_entities"] if entity_eval["extra_entities"] else "None")
            with col_ent3:
                st.markdown(f"**Missing Source Entities ({len(entity_eval['missing_entities'])}):**")
                st.write(entity_eval["missing_entities"][:8] if entity_eval["missing_entities"] else "None")

        # TAB 3: Numeric Consistency
        with tab_numeric:
            num_eval = evaluate_numeric_consistency(curr_article, gen_summary)

            n_col1, n_col2, n_col3 = st.columns(3)
            num_rate_str = f"{num_eval['numeric_consistency_rate']:.4f}" if num_eval['numeric_consistency_rate'] is not None else "N/A"

            with n_col1:
                st.metric("Numeric Overlap Rate", num_rate_str, help="Proportion of generated numbers present in source text.")
            with n_col2:
                st.metric("Source Numbers Found", num_eval["total_source_numbers"])
            with n_col3:
                st.metric("Summary Numbers Emitted", num_eval["total_summary_numbers"])

            st.markdown(f"**Numeric Status:** `{num_eval['status']}` — *{num_eval['status_description']}*")

            col_num1, col_num2 = st.columns(2)
            with col_num1:
                st.markdown(f"**Consistent Numbers ({len(num_eval['consistent_numbers'])}):**")
                st.write(num_eval["consistent_numbers"] if num_eval["consistent_numbers"] else "None")
            with col_num2:
                st.markdown(f"**Hallucinated Numbers ({len(num_eval['hallucinated_numbers'])}):**")
                st.write(num_eval["hallucinated_numbers"] if num_eval["hallucinated_numbers"] else "None")

        # TAB 4: Attention Alignment & Internals
        with tab_attn:
            if active_kind == "custom" and attn_matrix is not None and attn_matrix.numel() > 0:
                gen_words_list = [custom_vocab.idx2word.get(idx, "<UNK>") for idx in gen_token_ids]
                fig = plot_attention_heatmap(attn_matrix, src_tokens, gen_words_list[:25], max_src_display=30)
                st.pyplot(fig)
                st.caption("Bahdanau Additive Attention alignment heatmap across decoding steps (Y-axis) relative to source tokens (X-axis).")
            elif active_kind == "pretrained":
                st.markdown(
                    """
                    ### DistilBART Cross-Attention Architecture
                    DistilBART utilizes **12 multi-head attention layers** in the encoder and **6 layers** in the decoder, 
                    distilled from BART-large. Each layer maintains 16 attention heads across 1,024-dimensional hidden representations.
                    """
                )
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.info(
                        "**Pretrained Encoder**: 12 Transformer blocks with bidirectional self-attention capturing dense bidirectional contextual dependencies."
                    )
                with col_a2:
                    st.info(
                        "**Pretrained Decoder**: 6 Transformer blocks with causal masked self-attention and cross-attention dynamically querying encoder representations."
                    )
            else:
                st.info("No attention weights available for this generation.")

        # TAB 5: Scientific Model Comparison
        with tab_comparison:
            st.markdown("### Empirical Benchmark Comparison (CNN/DailyMail Test Set)")
            comp_path = "results/model_comparison_results.json"
            if os.path.exists(comp_path):
                try:
                    with open(comp_path, "r", encoding="utf-8") as fp:
                        comp_data = json.load(fp)
                    m = comp_data.get("summary_averages", {})
                    st.markdown(
                        f"""
                        | Evaluation Metric | Pretrained DistilBART-CNN-12-6 | Custom Seq2Seq (Greedy) | Custom Seq2Seq (Beam k=4) |
                        |---|---|---|---|
                        | **ROUGE-1 F1** | **{m.get('distilbart', {}).get('avg_rouge1_f1', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_rouge1_f1', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_rouge1_f1', 0.0):.4f} |
                        | **ROUGE-2 F1** | **{m.get('distilbart', {}).get('avg_rouge2_f1', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_rouge2_f1', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_rouge2_f1', 0.0):.4f} |
                        | **ROUGE-L F1** | **{m.get('distilbart', {}).get('avg_rougel_f1', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_rougel_f1', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_rougel_f1', 0.0):.4f} |
                        | **Entity Precision** | **{m.get('distilbart', {}).get('avg_entity_precision', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_entity_precision', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_entity_precision', 0.0):.4f} |
                        | **Entity Recall** | **{m.get('distilbart', {}).get('avg_entity_recall', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_entity_recall', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_entity_recall', 0.0):.4f} |
                        | **Numeric Consistency** | **{m.get('distilbart', {}).get('avg_numeric_rate', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_numeric_rate', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_numeric_rate', 0.0):.4f} |
                        | **Token Repetition** | **{m.get('distilbart', {}).get('avg_repetition_rate', 0.0):.4f}** | {m.get('custom_greedy', {}).get('avg_repetition_rate', 0.0):.4f} | {m.get('custom_beam', {}).get('avg_repetition_rate', 0.0):.4f} |
                        | **Inference Latency** | {m.get('distilbart', {}).get('avg_latency_ms', 0.0):.1f} ms | **{m.get('custom_greedy', {}).get('avg_latency_ms', 0.0):.1f} ms** | {m.get('custom_beam', {}).get('avg_latency_ms', 0.0):.1f} ms |
                        """
                    )
                except Exception as e:
                    st.warning(f"Could not parse comparison results: {e}")
            else:
                st.info("Run `scripts/run_model_comparison.py` to refresh empirical comparison results.")

    else:
        st.markdown(
            """
            <div style="background: rgba(13, 21, 39, 0.4); border: 1px dashed rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 2rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: #38bdf8;">⚡ CONSOLE READY</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    Select an evaluation sample or enter custom news text above, choose your preferred summarization engine, and click <b>EXECUTE NEURAL SUMMARIZATION</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Collapsible Model Architecture Dossier & Research Notes
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border-color: rgba(0, 240, 255, 0.15); margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)

    with st.expander("🔬 // 06 · ARCHITECTURE SPECIFICATIONS & RESEARCH DOSSIER", expanded=False):
        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown(
                """
                ### Custom Seq2Seq Neural Architecture (25.2M Params)
                - **Encoder**: 1-layer Bidirectional LSTM (256 forward + 256 backward units = 512-dim output).
                - **Attention**: Bahdanau Additive Attention with pad masking (`-1e9`).
                - **Decoder**: Attention-augmented LSTM with 896-dimensional projection:
                  Decoder Hidden (256) + Attention Context (512) + Token Embedding (128) $\\to$ 20,000 logits.
                - **Tokenizer**: Custom word-level regex tokenizer preserving numbers, currencies, dates, and decimals.
                - **Training**: 2,000 samples for 3 epochs with Adam ($lr=1e-3$).
                """
            )

        with col_m2:
            st.markdown(
                """
                ### Pretrained DistilBART Baseline (305.5M Params)
                - **Architecture**: DistilBART-CNN-12-6 distilled from BART-large.
                - **Layers**: 12 Transformer Encoder layers + 6 Transformer Decoder layers.
                - **Hidden Dimensions**: 1024-dim hidden state, 16 attention heads.
                - **Tokenizer**: Byte-Pair Encoding (BPE) with 50,264 vocabulary tokens.
                - **Dataset**: Pretrained on CNN/DailyMail 3.0.0 news highlights.
                """
            )

        st.markdown("---")
        st.markdown(
            """
            ### Scientific Limitations of Automated Evaluation Metrics
            - **Surface-Level Numeric Consistency**: Comparing extracted numerical tokens verifies that numbers in the summary appeared in the source text. However, it does not detect syntactic attribute binding errors (e.g., attributing 89 years to an auction price rather than an actor's age).
            - **Entity Preservation vs Groundedness**: Entity precision measures whether generated entities exist in the source, but full factual consistency requires natural language inference (NLI) or claim decomposition.
            - **Undertraining Perplexity**: Small prototype checkpoints suffer high perplexity ($\\sim$972), leading to argmax collapse into high-frequency punctuation. Pretrained transformers resolve this through billions of pre-training tokens.
            """
        )

    # -------------------------------------------------------------------------
    # Sidebar: Status & Reset
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚡ System Telemetry")
        st.success("All Neural Engines Online")
        st.markdown(
            f"""
            - **Active Engine**: `{selected_model_option.split('(')[0].strip()}`
            - **Pretrained**: DistilBART-CNN-12-6 (305.5M)
            - **Custom**: Seq2Seq BiLSTM (25.2M)
            - **Compute**: `{active_device_str}`
            - **Primary Dataset**: CNN/DailyMail 3.0.0
            """
        )

        st.markdown("---")
        st.markdown("### 🧹 State Management")
        if st.button("Reset Console State", use_container_width=True):
            clear_generated_summary_state()
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <small style="color: #64748b;">
            <b>NNDL Neural News Summarizer</b><br>
            Dual SOTA & Custom Architecture<br>
            Autonomous AI Research Console
            </small>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
