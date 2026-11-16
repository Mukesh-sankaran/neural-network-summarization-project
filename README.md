# NNDL Neural News Intelligence — Dual-Engine Abstractive Summarization Console

[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.21.0-FFD21E.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-black.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Verified-00f0ff.svg)]()

Interactive neural abstractive news summarization system featuring a **dual-model architecture**:
1. **Pretrained Baseline**: `sshleifer/distilbart-cnn-12-6` (305.5M parameters) for production-grade, factually grounded news summaries.
2. **Custom Research Model**: Scratch-built PyTorch Seq2Seq neural network with a Bidirectional LSTM Encoder, manual Bahdanau Additive Attention with padding masking, and an Attention-Augmented LSTM Decoder (25.2M parameters) paired with custom greedy/beam search and trigram repetition blocking.

Delivered within a premium **Cyberpunk AI Research Console** designed to WCAG AAA contrast standards.

---

## Table of Contents
- [1. Research Contribution & Novelty](#1-research-contribution--novelty)
- [2. Dual-Engine Architecture](#2-dual-engine-architecture)
- [3. Datasets & Pipeline](#3-datasets--pipeline)
- [4. Empirical Benchmark Comparison](#4-empirical-benchmark-comparison)
- [5. Fine-Tuning Feasibility Analysis](#5-fine-tuning-feasibility-analysis)
- [6. Cyberpunk Console UI](#6-cyberpunk-console-ui)
- [7. Installation & Quick Start](#7-installation--quick-start)
- [8. Test Suite & Verification](#8-test-suite--verification)
- [9. Repository Structure](#9-repository-structure)

---

## 1. Research Contribution & Novelty

This project implements and analyzes transparent neural sequence-to-sequence dynamics built entirely from PyTorch primitives (`torch.nn`), rather than relying solely on opaque pretrained transformer wrappers.

### Key Research Elements Implemented:
- **Scratch BiLSTM Encoder**: 128-dimensional trainable word embedding + 1-layer Bidirectional LSTM ($256 \times 2 = 512$-dim representation) with bidirectional hidden/cell state linear projections into decoder dimensions.
- **Bahdanau Additive Attention**: Manual additive alignment mechanism scoring decoder states against all 500 source encoder hidden states, using unnormalized projection vectors and $-1\text{e}9$ padding masking over `<PAD>` tokens.
- **Attention-Augmented Decoder**: LSTM cell receiving an 896-dimensional concatenated projection ($\text{Decoder Hidden } [256] + \text{Attention Context } [512] + \text{Input Embedding } [128] = 896$) projected into a 20,000-logit output vocabulary.
- **Zero-Leakage Vocabulary**: 20,000 unique word tokens constructed strictly from the CNN/DailyMail training set. Special tokens: `<PAD>` (0), `<UNK>` (1), `<SOS>` (2), `<EOS>` (3).
- **Empirical `<EOS>` Truncation Diagnosis**: Discovered that standard sequence truncation in early experiments inadvertently stripped `<EOS>` tokens for 39% of target highlights, causing the decoder to collapse into infinite period loops (`. . . .`). Resolved by pre-truncating target text to 58 tokens *prior* to sandwiching between `<SOS>` and `<EOS>`.
- **Custom Decoding Experiments**: Implemented and benchmarked greedy argmax decoding, beam search ($k=2$ to $8$) with Wu et al. length penalties ($\alpha$), and trigram repetition blocking.

---

## 2. Dual-Engine Architecture

| Attribute | SOTA Pretrained Baseline | Custom Research Model |
|---|---|---|
| **Model Checkpoint** | `sshleifer/distilbart-cnn-12-6` | `models/best_model.pt` |
| **Model Type** | DistilBART Transformer | BiLSTM + Bahdanau Additive Attention + LSTM |
| **Parameters** | 305,510,400 trainable | 25,229,600 trainable |
| **Tokenizer** | Hugging Face BPE (50,265 tokens) | Custom Word-Level Regex (20,000 tokens) |
| **Max Source Tokens** | 1,024 tokens | 500 tokens |
| **Max Target Tokens** | 142 tokens (configurable 40–200) | 60 tokens (configurable 20–100) |
| **Decoding Options** | Beam Search ($k=1$ to $8$), Length Penalty, Trigram Block | Greedy Decoding, Beam Search ($k=2$ to $8$), Trigram Block |
| **Primary Purpose** | Production-quality, fluent, factually grounded news summaries | Empirical study of Seq2Seq mechanics, attention heatmaps, and failure modes |

*Note: Tokenizers and vocabularies are strictly isolated. Custom token indices are never fed into DistilBART, and DistilBART BPE subwords are never fed into the custom word-level embedding.*

---

## 3. Datasets & Pipeline

- **Primary Dataset**: CNN/DailyMail 3.0.0 (311,971 samples: 287,113 Train, 13,368 Val, 11,490 Test).
- **Secondary Dataset**: XSum (226,711 samples: 204,045 Train, 11,332 Val, 11,334 Test).
- **Storage**: Local partitioned Apache Parquet shards under `datasets/cnn_dailymail/3.0.0/` and `datasets/xsum/data/`.
- **Preprocess Pipeline**: `src/preprocess.py` normalizes whitespace and isolates punctuation while strictly preserving numeric quantities, decimals (`3.14`), dates (`2026-08-21`), percentages (`15.5%`), and currencies (`£100,000`, `$500`).

---

## 4. Empirical Benchmark Comparison

Evaluated on held-out news articles from the CNN/DailyMail test set using `scripts/run_model_comparison.py`:

| Metric | DistilBART-CNN-12-6 (Baseline) | Custom Seq2Seq (Greedy) | Custom Seq2Seq (Beam $k=4$ + Trigram) |
|---|:---:|:---:|:---:|
| **ROUGE-1 F1** | **0.3459** | 0.0314 | 0.1325 |
| **ROUGE-2 F1** | **0.1814** | 0.0000 | 0.0080 |
| **ROUGE-L F1** | **0.3001** | 0.0314 | 0.1082 |
| **Entity Precision** | **0.8750** | 0.0000 | 0.0000 |
| **Entity Recall** | **0.2266** | 0.0000 | 0.0000 |
| **Numeric Consistency** | **1.0000** | 0.0000 | 0.0000 |
| **Token Repetition Rate** | **0.1570** | 0.8835 | 0.5667 |
| **Average Latency** | 2,961.5 ms | **253.1 ms** | 842.8 ms |

### Scientific Findings:
1. **Pretrained DistilBART** generates grammatical, informative, and factually grounded summaries with 87.5% entity precision and 100% numeric consistency against source articles.
2. **Custom Prototype Checkpoint** (`models/best_model.pt`) was trained on 2,000 examples for 3 epochs (final validation loss: 6.8798, perplexity $\sim$972). Under greedy argmax, high perplexity leads to repetitive punctuation collapse. Beam search with trigram blocking successfully breaks repetitive loops (boosting ROUGE-1 from 0.0314 to 0.1325), but semantic coherence requires large-scale training on the full 287k dataset.

---

## 5. Fine-Tuning Feasibility Analysis

A lightweight fine-tuning pipeline was implemented in `scripts/finetune_baseline.py` and evaluated on local hardware (macOS ARM64, 8 GB unified memory, Apple Silicon MPS):
- **Throughput**: ~30 seconds per batch on Apple Silicon MPS with gradient accumulation.
- **Compute Estimation**: Training a single epoch across 287,113 samples would require $>2,000$ GPU hours on this machine.
- **Numerical Stability**: Autoregressive cross-entropy backpropagation with dynamic padding masks on Apple Silicon MPS exhibited gradient overflow (`loss: nan`) at step 2.
- **Engineering Decision**: The untouched pretrained checkpoint (`sshleifer/distilbart-cnn-12-6`) is promoted as the primary baseline. The fine-tuning pipeline is provided for multi-GPU CUDA environments.

---

## 6. Cyberpunk Console UI

The user interface (`app.py`) has been upgraded to a dark cyberpunk research console:
- **Palette**: Deep void canvas (`#070b14`), terminal cards (`#090e1a`, `#0d1527`), neon cyan (`#00f0ff`), electric violet (`#8b5cf6`), and soft coral warnings (`#f43f5e`).
- **Telemetry Bar**: Live word and token counters, encoder budget utilization indicators, and model parameter badges.
- **Dynamic Control Deck**: Automatically toggles controls based on selected backend (Beam search width, length penalty, trigram blocking, max output length).
- **Multi-Metric Grounding Console**:
  - **Tab 1: ROUGE Overlap** (Punctuation-normalized unigram, bigram, and LCS matching).
  - **Tab 2: Entity Preservation** (spaCy `en_core_web_sm` groundedness, coverage, and missing/hallucinated entity breakdowns).
  - **Tab 3: Numeric Consistency** (Dates, currencies, percentages, and cardinal number preservation).
  - **Tab 4: Attention & Internals** (Dynamic Bahdanau alignment heatmap for custom Seq2Seq; multi-head cross-attention breakdown for DistilBART).
  - **Tab 5: Empirical Comparison** (Side-by-side benchmark table from test set runs).

---

## 7. Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- macOS (Apple Silicon MPS supported) or Linux / Windows (CPU / CUDA)

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd "NNDL summarization project"

# Activate the local virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 8. Test Suite & Verification

The project includes 4 test suites:

```bash
# 1. Neural Network Sanity Tests (Encoder, Attention, Decoder, Dimensions, Shapes)
python test_model_sanity.py

# 2. Stage 2 Verification (Model loading, empty input rejection, numeric/entity edge cases)
python test_stage2_verification.py

# 3. Stage 4B Verification (Greedy/Beam decoding, trigram blocking, UI contrast, state invalidation)
python test_stage4b_verification.py

# 4. Dual Model & Pretrained Baseline Tests (DistilBART loading, tokenizer isolation, inference)
python test_pretrained_and_dual_model.py
```

### Standalone Evaluation Scripts
```bash
# Run 5-article empirical benchmark comparison
python scripts/run_model_comparison.py

# Verify bounded Streamlit application startup and health check
python scripts/verify_app_startup.py

# Run fine-tuning pilot pipeline
python scripts/finetune_baseline.py --steps 5 --samples 10
```

---

## 9. Repository Structure

```
├── app.py                               # Cyberpunk Streamlit application
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive project documentation
├── test_model_sanity.py                 # Neural network architecture sanity tests
├── test_stage2_verification.py          # Data pipeline & evaluation tests
├── test_stage4b_verification.py         # Decoding strategy & UI contrast tests
├── test_pretrained_and_dual_model.py    # Pretrained baseline & dual-model test suite
├── test_training_diagnostics.py         # Target sequence & EOS truncation tests
├── src/
│   ├── model.py                         # PyTorch Seq2Seq architecture & generate_summary()
│   ├── encoder.py                       # Bidirectional LSTM Encoder
│   ├── attention.py                     # Bahdanau Additive Attention with pad mask
│   ├── decoder.py                       # Attention-augmented LSTM Decoder
│   ├── vocabulary.py                    # Custom word-level vocabulary
│   ├── preprocess.py                    # Number-preserving tokenizer & text cleaner
│   ├── pretrained_baseline.py           # Pretrained Hugging Face baseline loader & inference
│   ├── evaluate.py                      # ROUGE-1/2/L evaluation with punctuation exclusion
│   ├── entity_evaluation.py             # spaCy NER & numeric consistency metrics
│   ├── data_loader.py                   # Local Parquet dataset loader
│   └── dataset.py                       # PyTorch SummarizationDataset
├── scripts/
│   ├── run_model_comparison.py          # Dual-model benchmark comparison script
│   ├── verify_app_startup.py            # Bounded Streamlit health check verification
│   └── finetune_baseline.py             # Pretrained model fine-tuning pipeline
├── models/
│   ├── best_model.pt                    # Custom Seq2Seq checkpoint (Epoch 3, 25.2M params)
│   ├── vocab.pkl                        # Pickled 20,000-token vocabulary
│   └── finetuned_distilbart/            # Pilot fine-tuning metadata & configs
├── datasets/                            # Local CNN/DailyMail and XSum Parquet files
├── results/                             # Benchmark JSONs and evaluation charts
└── backups/                             # Timestamped backups of original files
```
