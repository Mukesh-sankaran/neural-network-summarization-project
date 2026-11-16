# Comprehensive Project Audit Report: NNDL Neural News Summarization

**Audit Date**: October 31, 2026 / November 1, 2026  
**Auditor**: Antigravity Autonomous Research Assistant (Read-Only Audit)  
**Project Directory**: `/Users/mukeshsankaran/Documents/projects/NNDL summarization project`  
**Audit Scope**: Stage 1 Read-Only Complete Audit (Code, Artifacts, Hardware, Verification, Benchmarks)

---

## 1. Environment Summary

### 1.1 System Hardware & Operating System
- **Operating System**: macOS 27.0 (ProductVersion `27.0`, BuildVersion `26A428`)
- **Kernel / Architecture**: Darwin ARM64 (Apple Silicon)
- **Total Physical Memory**: 8,589,934,592 bytes (8.00 GB unified memory)
  - *Evidence*: `sysctl -n hw.memsize` returned `8589934592`.
- **Antigravity CLI Version**: `agy 1.4.12`

### 1.2 Python Environment & Package Versions
- **Active Python Interpreter**: `/Users/mukeshsankaran/Documents/projects/NNDL summarization project/.venv/bin/python`
- **Python Version**: `3.14.7`
- **PyTorch & Accelerator**:
  - `torch.__version__`: `2.13.0`
  - `torch.backends.mps.is_available()`: `True`
  - `torch.backends.mps.is_built()`: `True`
- **Core Framework Versions**:
  - `transformers`: `5.21.0`
  - `streamlit`: `1.66.1`
  - `safetensors`: `0.8.0`
  - `spacy`: `3.8.13` (with model `en_core_web_sm` 3.8.0)
  - `datasets`: `5.0.1`
  - `huggingface_hub`: `2.3.0`
  - `numpy`: `2.5.1`
  - `pandas`: `3.0.5`
- **Test Runner Check**:
  - `pytest`: **NOT INSTALLED** (`No module named pytest`). Tests are executed via individual Python scripts invoking their respective test runners.

### 1.3 Dependency Conflict Audit (`pip check`)
- **Status**: **FAILED (Exit Code 1)**
- **Raw Command Output**:
  ```text
  datasets 5.0.1 has requirement huggingface-hub<2.0,>=0.25.0, but you have huggingface-hub 2.3.0.
  ```
- **Analysis**: The installed `datasets` package (v5.0.1) specifies a strict upper bound `<2.0` on `huggingface-hub`, but `huggingface-hub` v2.3.0 is installed. While local parquet dataset loading remains functional through PyArrow, HF Hub programmatic dataset fetching may experience compatibility errors.

---

## 2. Actual File Tree and Artifact Checksums

### 2.1 File Tree Inventory (Excluding `.venv` and `__pycache__`)
```text
.
├── .DS_Store (6.0 KB)
├── .streamlit/
│   └── config.toml (227 B)
├── ARCHITECTURE_DIAGRAM.png (300 KB)
├── EVALUATION_RESULTS.png (125 KB)
├── PROJECT_FLOWCHART.png (409 KB)
├── PROJECT_PRESENTATION.md (14 KB)
├── PROJECT_REPORT.md (20 KB)
├── PROJECT_STATUS.txt (3.6 KB)
├── README.md (12 KB)
├── TRAINING_RESULTS.png (121 KB)
├── app.py (61 KB)
├── current_desktop.png (7.0 KB)
├── current_mobile.png (4.4 KB)
├── demo.py (9.9 KB)
├── download_dataset.py (242 B)
├── download_xsum.py (197 B)
├── generate_diagrams_and_plots.py (6.7 KB)
├── project_dashboard.py (16 KB)
├── requirements.txt (547 B)
├── run_pipeline_validation.py (11 KB)
├── test_model_sanity.py (7.8 KB)
├── test_pretrained_and_dual_model.py (11 KB)
├── test_stage2_verification.py (10 KB)
├── test_stage4b_verification.py (16 KB)
├── test_training_diagnostics.py (4.4 KB)
├── _audit_stage1/
│   ├── artifact_checksums.txt (234 B)
│   ├── test_results.txt (13 KB)
│   └── AUDIT_REPORT.md
├── backups/
│   ├── app_pre_cyberpunk_master.py (47 KB)
│   ├── final_master_pre_integration/ (app.py, requirements.txt, src_bak/)
│   ├── stage2_backup_original/ (demo.py, entity_evaluation.py, evaluate.py)
│   ├── stage4b_backup/ (app.py.bak, model.py.bak)
│   ├── stage4b_cyberpunk_redesign_backup/ (app.py.bak)
│   ├── stage4b_premium_cyberpunk_backup/ (app.py.bak)
│   └── stage4b_session2_backup/ (app.py.bak, model.py.bak)
├── datasets/
│   ├── cnn_dailymail/
│   │   ├── 3.0.0/ (train, val, test parquet shards)
│   │   └── .cache/huggingface/
│   └── xsum/
│       ├── data/ (train, val, test parquet shards)
│       └── .cache/huggingface/
├── models/
│   ├── best_model.pt (288.75 MB / 302,782,517 B)
│   ├── vocab.pkl (910.45 KB / 932,302 B)
│   └── finetuned_distilbart/
│       ├── config.json (1.6 KB)
│       ├── generation_config.json (379 B)
│       ├── pilot_metadata.json (237 B)
│       ├── tokenizer.json (3.4 MB)
│       └── tokenizer_config.json (389 B)
├── results/
│   ├── entity_preservation.png (75 KB)
│   ├── evaluation_report.json (61 KB)
│   ├── evaluation_results.png (125 KB)
│   ├── factual_consistency.png (64 KB)
│   ├── model_comparison_results.json (35 KB)
│   ├── rouge_scores.png (84 KB)
│   ├── sequence_lengths.png (173 KB)
│   ├── training_history.json (270 B)
│   └── training_validation_loss.png (121 KB)
├── scripts/
│   ├── finetune_baseline.py (6.9 KB)
│   ├── run_model_comparison.py (16 KB)
│   └── verify_app_startup.py (2.6 KB)
└── src/
    ├── attention.py (3.8 KB)
    ├── data_loader.py (3.4 KB)
    ├── dataset.py (3.6 KB)
    ├── decoder.py (4.8 KB)
    ├── encoder.py (4.2 KB)
    ├── entity_evaluation.py (11 KB)
    ├── evaluate.py (15 KB)
    ├── model.py (16 KB)
    ├── preprocess.py (1.7 KB)
    ├── pretrained_baseline.py (6.0 KB)
    ├── train.py (13 KB)
    └── vocabulary.py (3.5 KB)
```

### 2.2 Model Artifact Checksums
Computed via `shasum -a 256` and saved in `_audit_stage1/artifact_checksums.txt`:
- **`models/best_model.pt`**:
  - **SHA-256**: `0963e1d7eb342c72e3af9d73d88a49174845626604c47296ad4c3e340ec2af96`
  - **Exact Byte Size**: `302,782,517` bytes (~288.75 MB)
- **`models/vocab.pkl`**:
  - **SHA-256**: `a5ad782774b3330ef5d340f4ff8624c5d9a642d774b4a83fb571a33be1b24645`
  - **Exact Byte Size**: `932,302` bytes (~910.45 KB)

### 2.3 Hugging Face Cache Audit (`~/.cache/huggingface/hub`)
- **Directory**: `/Users/mukeshsankaran/.cache/huggingface/hub/models--sshleifer--distilbart-cnn-12-6`
- **Total Dereferenced Disk Size**: `4.6 GB`
- **Cached Snapshots**:
  - `a4f8f3ea906ed274767e9906dbaede7531d660ff` (symlinked to PyTorch binary blob `3bac65d1...`, 1.1 GB)
  - `e9f0c28b06171c3fc52cd02cc8481cb7350fd32e` (symlinked to safetensors blob `c37a3dec...`, 1.1 GB)
- **Status**: The pretrained model `sshleifer/distilbart-cnn-12-6` is fully cached locally; zero external download is required for local inference.

### 2.4 Real Locations of Expected Files
| Target File | Real Location | Project Root Presence | Notes |
|---|---|:---:|---|
| `app.py` | `./app.py` | **YES** | Primary Streamlit application entry point |
| `pretrained_baseline.py` | `./src/pretrained_baseline.py` | **NO** | Moved into `src/` modular directory |
| `run_model_comparison.py` | `./scripts/run_model_comparison.py` | **NO** | Moved into `scripts/` directory |
| `finetune_baseline.py` | `./scripts/finetune_baseline.py` | **NO** | Moved into `scripts/` directory |
| `verify_app_startup.py` | `./scripts/verify_app_startup.py` | **NO** | Moved into `scripts/` directory |
| `test_model_sanity.py` | `./test_model_sanity.py` | **YES** | 10 unit checks |
| `test_pretrained_and_dual_model.py` | `./test_pretrained_and_dual_model.py` | **YES** | 7 dual-model test suites |
| `test_stage2_verification.py` | `./test_stage2_verification.py` | **YES** | 6 test suites |
| `test_stage4b_verification.py` | `./test_stage4b_verification.py` | **YES** | 6 test suites |
| `test_training_diagnostics.py` | `./test_training_diagnostics.py` | **YES** | Diagnostic inspection script |
| `requirements.txt` | `./requirements.txt` | **YES** | Root configuration file |
| `README.md` | `./README.md` | **YES** | Complete project documentation |

---

## 3. Claim-by-Claim Verification Table

| # | Claim | Status | Evidence (Command + Output Reference) |
|---|---|:---:|---|
| 1 | Custom Seq2Seq uses a Bidirectional LSTM Encoder | **VERIFIED** | `src/encoder.py:51-58`: `nn.LSTM(..., bidirectional=True, hidden_size=256)`. Output dimension: $256 \times 2 = 512$. |
| 2 | Attention is Bahdanau Additive Attention | **VERIFIED** | `src/attention.py:67-72`: `energy = torch.tanh(W_enc(enc) + W_dec(dec))`; `scores = v(energy)`. Softmax over masked positions (-1e9). |
| 3 | Decoder is an Attention-Augmented LSTM | **VERIFIED** | `src/decoder.py:63-68`: `nn.LSTM(input_size=128+512, hidden_size=256, num_layers=1)`. Output projection concatenates `[lstm_out (256) + context (512) + emb (128)]` = 896 -> 20,000. |
| 4 | Custom vocabulary contains exactly 20,000 word-level tokens | **VERIFIED** | `models/vocab.pkl`: `len(vocab) == 20000`. Special tokens: `<PAD>`:0, `<UNK>`:1, `<SOS>`:2, `<EOS>`:3. Matching embedding shapes `[20000, 128]` in checkpoint. |
| 5 | Custom model checkpoint has 25.23M parameters | **VERIFIED** | `torch.load("models/best_model.pt")`: sum of elements in `model_state_dict` is **25,229,600** parameters (25.2296M rounds to 25.23M). |
| 6 | DistilBART baseline has 305.5M parameters | **VERIFIED** | `AutoModelForSeq2SeqLM.from_pretrained(...)`: sum of parameters is **305,510,400** parameters (305.5104M rounds to 305.5M). |
| 7 | Checkpoint was trained for 3 epochs with high validation loss | **VERIFIED** | `models/best_model.pt` metadata: `epoch: 3`, `train_loss: 6.3864`, `val_loss: 6.8798`. |
| 8 | Test suites have pass counts (10, 6, 6, 7) | **VERIFIED** | `_audit_stage1/test_results.txt`: `test_model_sanity.py` (10 passed), `test_stage2_verification.py` (6 passed), `test_stage4b_verification.py` (6 passed), `test_pretrained_and_dual_model.py` (7 passed). |
| 9 | Pytest is the test framework | **CONTRADICTED** | `python -m pytest`: `No module named pytest`. Pytest is not installed; tests are executed as standalone scripts. |
| 10 | Pretrained baseline was fine-tuned locally | **CONTRADICTED** | `scripts/finetune_baseline.py` pilot failed with `loss: NaN` at step 2 (`pilot_metadata.json`); `models/finetuned_distilbart/` contains no weights. The active engine runs unmodified `sshleifer/distilbart-cnn-12-6`. |
| 11 | Files `pretrained_baseline.py`, `run_model_comparison.py`, `finetune_baseline.py`, `verify_app_startup.py` reside at root | **CONTRADICTED** | Files reside in `src/` and `scripts/`. |
| 12 | Tokenizer isolation: no cross-contamination between custom and DistilBART tokens | **VERIFIED** | `app.py:915-1004` and `test_pretrained_and_dual_model.py:Test 2`: separate code branches, distinct session state keys, and automatic cache invalidation. |
| 13 | DistilBART achieves 100% numeric consistency across test articles | **CONTRADICTED (FLAWED)** | In `scripts/run_model_comparison.py`, only 2 of 5 summaries contained numbers (both 1.0). The other 3 returned `None`. The script averaged only non-`None` values, presenting an inflated 1.0 (100%) average. |
| 14 | Apple Silicon MPS acceleration is available and functional | **VERIFIED** | `torch.backends.mps.is_available() == True`. Verified via tensor allocations and inference runs on `mps`. |
| 15 | Pip packages have zero dependency conflicts | **CONTRADICTED** | `pip check` reports conflict: `datasets 5.0.1` requires `huggingface-hub<2.0,>=0.25.0`, but `huggingface-hub 2.3.0` is installed. |
| 16 | Streamlit headless app starts and passes health probe | **VERIFIED** | `http://localhost:8599/_stcore/health` returned HTTP 200 `ok` within 2.3 seconds, and process terminated cleanly. |
| 17 | Custom model produces coherent summaries without training defects | **CONTRADICTED** | Due to training on only 2,000 samples for 3 epochs (val loss 6.88), greedy decoding collapses into repetitive punctuation loops (`. . . .`). Beam search + trigram blocking stops loops but produces severe hallucinations. |

---

## 4. Architecture as Actually Implemented

### 4.1 Custom Seq2Seq Neural Architecture (`src/`)

```
Source Tokens [batch, seq_len] (Max 500)
    │
    ▼
[Word Embedding] ────────────────────── 20,000 Vocab -> 128-dim
    │
    ▼
[Bidirectional LSTM] ────────────────── 1 Layer, Hidden 256 (Forward + Backward)
    │
    ├─► Encoder Outputs [batch, seq_len, 512]
    │
    └─► Combined States [1, batch, 512] ──► [fc_hidden, fc_cell + tanh] ──► Initial Decoder States [1, batch, 256]

Decoder Step t:
Input Token (t-1) ──► [Word Embedding (128)]
                               │
[Bahdanau Additive Attention] ◄┼── Decoder Hidden (t-1) [batch, 256] & Encoder Outputs [batch, seq_len, 512]
    │                          │   (Masked Softmax over padding tokens, score = v^T * tanh(W_e*enc + W_d*dec))
    ▼                          │
Context Vector [batch, 512]    │
    │                          │
    ├─────────► Concatenate [Embedding (128) + Context (512)] = 640-dim
    │                          │
    │                          ▼
    │                   [Decoder LSTM Cell] ──► Updated Hidden & Cell [1, batch, 256]
    │                          │
    └─────────► Concatenate [Hidden (256) + Context (512) + Embedding (128)] = 896-dim
                               │
                               ▼
                        [fc_out Linear] ──────► Vocabulary Logits [batch, 20,000]
```

- **Encoder (`src/encoder.py:11-103`)**:
  - Embedding: `nn.Embedding(20000, 128, padding_idx=0)`
  - LSTM: `nn.LSTM(128, 256, num_layers=1, batch_first=True, bidirectional=True)` (lines 51-58)
  - Hidden projection: `self.fc_hidden = nn.Linear(512, 256)` (line 63)
  - Cell projection: `self.fc_cell = nn.Linear(512, 256)` (line 64)
  - Forward: `decoder_init_hidden = torch.tanh(self.fc_hidden(combined_hidden))` (lines 96-101)
- **Bahdanau Additive Attention (`src/attention.py:13-89`)**:
  - Projection layers: `W_encoder = nn.Linear(512, 256, bias=False)`, `W_decoder = nn.Linear(256, 256, bias=False)`, `v = nn.Linear(256, 1, bias=False)` (lines 38-40)
  - Energy: `energy = torch.tanh(proj_encoder + proj_decoder)` (line 69)
  - Scoring & Masking: `scores = self.v(energy).squeeze(2)`; `scores.masked_fill(mask == 0, -1e9)` (lines 72-78)
  - Attention distribution & Context: `attention_weights = F.softmax(scores, dim=1)`; `context_vector = torch.bmm(...)` (lines 81-87)
- **Decoder (`src/decoder.py:12-126`)**:
  - Embedding: `nn.Embedding(20000, 128, padding_idx=0)`
  - LSTM: `nn.LSTM(input_size=128 + 512, hidden_size=256, num_layers=1, batch_first=True)` (lines 63-68)
  - Output Linear: `nn.Linear(256 + 512 + 128, 20000)` (lines 71-74)
- **Vocabulary (`src/vocabulary.py:18-112`)**:
  - Size: 20,000 tokens built strictly on training corpus unigrams.
  - Special tokens: `<PAD>`: 0, `<UNK>`: 1, `<SOS>`: 2, `<EOS>`: 3 (lines 38-45).
- **Inference Decoding & Heuristics (`src/model.py:161-395`)**:
  - Special token suppression: Logits for `<PAD>` and `<SOS>` are clamped to `-1e9` at every generation step (lines 270-271, lines 319-320).
  - Greedy decoding: Argmax per step until `<EOS>` or `max_target_len` (lines 249-289).
  - Beam search: Configurable beam width $k$, active candidate tracking, early termination when $k$ beams finish (lines 293-374).
  - Length Penalty: Wu et al. length normalization: $\text{factor} = ((5 + L) / 6)^\alpha$ applied to log-likelihood scores (lines 359, 382).
  - Trigram Blocking (`_block_repeat_ngrams`): Masks candidate vocabulary tokens that would create a repeated $n$-gram to `-1e9` (lines 161-197).

### 4.2 Pretrained Transformer Baseline (`src/pretrained_baseline.py`)
- **Model Identifier**: `sshleifer/distilbart-cnn-12-6` (line 12).
- **Tokenizer**: Hugging Face fast BPE tokenizer (`AutoTokenizer.from_pretrained`), vocabulary size 50,265.
- **Device Management**: `get_pretrained_device()` checks `torch.backends.mps.is_available()`, tests allocation on `mps`, and falls back to CPU if an error occurs.
- **Generation Parameters**: `max_length=142`, `min_length=30`, `num_beams=4`, `length_penalty=1.0`, `no_repeat_ngram_size=3`, `early_stopping=True` (lines 86-93, 132-139).
- **Local Checkpoint Loading**: Unused. The module exclusively loads the upstream Hugging Face checkpoint.

### 4.3 Interactive Application Architecture (`app.py`)
- **Model Selection Deck**: Radio selector allows hot-switching between DistilBART and Custom Seq2Seq (lines 673-681).
- **State Invalidation & Isolation**:
  - `current_model_option` and `current_sample_key` are tracked in `st.session_state` (lines 686-691, 843-847).
  - When either changes, `clear_generated_summary_state()` clears 13 session keys (`gen_summary`, `gen_token_ids`, `attn_matrix`, `src_tokens`, `last_article`, `last_reference`, `used_strategy`, `used_beam_width`, `used_trigram_blocking`, `used_length_penalty`, `inference_latency_ms`, `active_model_badge`, `active_model_kind`).
  - Text area input change detection: lines 874-878 check if `last_article` or `last_reference` changed and clear stale summaries.
  - Reset button: lines 1314-1316 invoke `clear_generated_summary_state()` and `st.rerun()`.

---

## 5. Bugs and Risks Found (Ranked by Severity)

### Severity 1: HIGH — Flawed Metric Aggregation in Model Comparison
- **File**: `scripts/run_model_comparison.py:245-247`, `257`
- **Issue**:
  ```python
  def avg(vals):
      clean = [v for v in vals if v is not None]
      return float(sum(clean) / len(clean)) if clean else 0.0
  ```
  In `src/entity_evaluation.py:223-236`, when a generated summary contains no numbers, `evaluate_numeric_consistency` returns `None` (precision undefined). In `scripts/run_model_comparison.py`, `avg()` filters out all `None` values.
  For DistilBART, only 2 of the 5 evaluation articles produced summaries containing numbers (both scored 1.0). The remaining 3 returned `None`. Filtering `None` caused the benchmark to report an overall numeric consistency of **1.0 (100.0%)**, masking the fact that 60% of test summaries had no numbers evaluated.
  For the custom model, all 5 summaries produced `None`, so `clean = []` returned 0.0.
- **Risk**: Overstates factual grounding reliability and misrepresents empirical benchmarks.

### Severity 2: HIGH — Fine-Tuning Pipeline Instability (NaN Loss on MPS)
- **File**: `scripts/finetune_baseline.py:121-137`, `models/finetuned_distilbart/pilot_metadata.json`
- **Issue**: The pilot run of `scripts/finetune_baseline.py` on Apple Silicon MPS experienced gradient explosion/overflow, resulting in:
  ```json
  {
    "final_train_loss": NaN,
    "final_val_loss": NaN,
    "elapsed_seconds": 299.59890770795755,
    "device": "mps",
    "status": "pilot_completed"
  }
  ```
  Consequently, `models/finetuned_distilbart/` contains tokenizer and config files, but zero model weights (`model.safetensors` or `pytorch_model.bin`).
- **Risk**: Attempting to train or fine-tune DistilBART without fp16/bf16 gradient scaling, lower learning rates, or CPU fallback on Apple Silicon causes complete divergence.

### Severity 3: MEDIUM — Missing Device Synchronization in Latency Benchmarks
- **File**: `src/pretrained_baseline.py:153-156`, `scripts/run_model_comparison.py:176-181`, `188-194`
- **Issue**: Latency is measured using `t0 = time.perf_counter() ... latency = time.perf_counter() - t0` without calling `torch.mps.synchronize()` (or `torch.cuda.synchronize()`).
- **Risk**: PyTorch operations on Apple Silicon MPS are executed asynchronously. Wall-clock timing without synchronization measures host dispatch time rather than complete GPU kernel execution, underestimating true inference latency.

### Severity 4: MEDIUM — Dependency Conflict (`pip check`)
- **File**: Project virtual environment (`.venv`)
- **Issue**: `pip check` fails with:
  `datasets 5.0.1 has requirement huggingface-hub<2.0,>=0.25.0, but you have huggingface-hub 2.3.0.`
- **Risk**: Potential API incompatibility during future dataset downloads or Hugging Face Hub operations.

### Severity 5: MEDIUM — Inaccurate UI Text on Model Provenance
- **File**: `app.py:1080`
- **Issue**: UI text states:
  `<b>SOTA BASELINE NOTICE:</b> Generated via <code>sshleifer/distilbart-cnn-12-6</code> (DistilBART 305.5M parameters fine-tuned on CNN/DailyMail).`
- **Risk**: May mislead evaluators into believing that the model running in the UI was fine-tuned locally in this project, whereas it is the standard upstream Hugging Face checkpoint.

### Severity 6: LOW — Lack of Inference Warm-Up in Benchmarks
- **File**: `scripts/run_model_comparison.py:154-206`
- **Issue**: No warm-up pass is performed before benchmarking. The first article (`sample_hardy`) absorbs MPS shader compilation overhead, logging 3,895 ms latency versus ~2,700 ms for subsequent runs.

### Severity 7: LOW — Security Warning in Streamlit Configuration
- **File**: `.streamlit/config.toml:11`
- **Issue**: `enableCORS = false` triggers Streamlit's security notice regarding unrestricted cross-origin requests.

---

## 6. Metric-Implementation Review

### 6.1 ROUGE Metrics (`src/evaluate.py:30-146`)
- **What It Measures**: N-gram precision, recall, and F1 overlap between candidate summary and human reference summary (ROUGE-1 for unigrams, ROUGE-2 for bigrams, ROUGE-L for longest common subsequence).
- **Implementation**: Pure Python implementation using `collections.Counter` and dynamic programming LCS table (`compute_lcs_length`).
- **Soundness**: Sound and mathematically accurate.
- **Normalization Feature**: `normalize_rouge_tokens()` (lines 97-114) strips standalone punctuation tokens (e.g. `.`, `,`, `'`, `"`). This is a critical safeguard: without this filter, the custom model's degenerate punctuation loops (`. . . . .`) artificially scored a high ROUGE-1/L recall against punctuation marks in human references.

### 6.2 Entity Preservation Metrics (`src/entity_evaluation.py:44-162`)
- **What It Measures**: Groundedness and coverage of named entities between source document and generated summary.
  - *Entity Precision*: Proportion of generated summary entities that exist in the source document ($|E_{preserved}| / |E_{summary}|$).
  - *Entity Recall*: Proportion of unique source document entities preserved in the summary ($|E_{preserved}| / |E_{source}|$).
  - *Preservation Rate*: Mention-frequency-weighted source coverage.
  - *Entity F1*: Harmonic mean of entity precision and recall.
- **Implementation**: Uses spaCy `en_core_web_sm` pipeline targeting 9 named entity categories (`PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `MONEY`, `PERCENT`, `CARDINAL`, `QUANTITY`).
- **Soundness**: Methodologically sound. Distinguishes between unique entity set overlap and raw mention frequency. Accurately flags `None` when zero entities are present.

### 6.3 Numeric Consistency Metric (`src/entity_evaluation.py:165-268`)
- **What It Measures**: Surface-level verification that numbers, dates, currencies, and percentages appearing in the summary originate in the source document.
- **Implementation**: Regex extraction covering currencies (`$100`, `£100,000`), percentages (`15.5%`), ISO dates (`2026-08-21`), slash dates (`08/21/2026`), decimals (`3.14`), and integers.
- **Flaw**: While the per-sample function correctly returns `None` when the summary has no numbers, the aggregation function in `scripts/run_model_comparison.py:245-247` drops `None` values, leading to a misleading 100% score when only a minority of articles contained numbers.

### 6.4 Repetition Rate (`scripts/run_model_comparison.py:103-111`, `app.py:230-245`)
- **What It Measures**: Proportion of unigram tokens that represent repeated occurrences:
  $$\text{Repetition Rate} = \frac{\sum_{w} (\text{count}(w) - 1)}{N}$$
- **Soundness**: Sound as a lexical diversity and degeneration indicator. Accurately exposes greedy decoding collapse (88.3% repetition) versus beam search with trigram blocking (56.7%).

### 6.5 Latency Timing (`src/pretrained_baseline.py:153-157`, `scripts/run_model_comparison.py`)
- **What It Measures**: End-to-end execution time in milliseconds.
- **Flaw**:
  1. DistilBART timing measures only `model.generate()`, omitting tokenizer encoding and decoding. Custom Seq2Seq timing encompasses token encoding, generation, and decoding.
  2. No GPU synchronization (`torch.mps.synchronize()`), introducing timing jitter on Apple Silicon.
  3. No warm-up pass, inflating first-run latency by 30-40%.

---

## 7. UI Text Claims That Are Inaccurate or Unsupported

1. **`app.py:1080`**:
   - *Text*: `<b>SOTA BASELINE NOTICE:</b> Generated via <code>sshleifer/distilbart-cnn-12-6</code> (DistilBART 305.5M parameters fine-tuned on CNN/DailyMail). Summaries exhibit coherent grammar, strong source grounding, and high entity preservation.`
   - *Issue*: Ambiguous wording. While `sshleifer` originally fine-tuned this model on CNN/DailyMail on Hugging Face, presenting it under the project banner without clarification implies local fine-tuning. The local fine-tuning attempt failed (`loss: NaN`), and no local fine-tuned weights exist.
2. **`app.py:744`**:
   - *Text*: `<span class="pill-badge pill-cyan">CNN/DAILYMAIL 3.0.0</span>`
   - *Issue*: While the custom model vocabulary and training subsets were sourced from CNN/DailyMail, the custom checkpoint in `models/best_model.pt` was trained on only 2,000 samples for 3 epochs. Labeling the overall system with CNN/DailyMail 3.0.0 could suggest training on the full 287k dataset.
3. **`app.py:1223`**:
   - *Text*: `| Numeric Consistency | 100.0% | 0.0% | 0.0% |` (in benchmark table)
   - *Issue*: Directly presents the flawed aggregation from `results/model_comparison_results.json` where DistilBART is credited with 100.0% numeric consistency despite 3 out of 5 summaries having zero numbers evaluated.

---

## 8. Items That Could Not Be Verified and Why

1. **Historical Training Run that Created `models/best_model.pt`**:
   - *Reason*: Under the read-only audit constraints (Rule 2: "Do NOT run any training or fine-tuning script"), `src/train.py` was not executed. Verification was restricted to loading and inspecting the saved checkpoint metadata (`epoch: 3`, `train_loss: 6.3864`, `val_loss: 6.8798`, `vocab_size: 20000`).
2. **CUDA Device Acceleration**:
   - *Reason*: Audit environment is a local Apple Silicon Mac (`Darwin ARM64`). CUDA hardware is physically absent; CUDA-specific code paths (`torch.cuda.is_available()`) could not be dynamically executed.
3. **Training on the Full 287k CNN/DailyMail Dataset**:
   - *Reason*: Read-only rule prohibits executing training jobs. Furthermore, local hardware constraints (8 GB RAM) would require hundreds of hours.

---

## 9. Proposed (NOT Applied) Changes

In accordance with Rule 1 ("Record it in the report instead of making it"), the following changes are proposed for future stages:

| Proposed Change | File | Rationale |
|---|---|---|
| **1. Fix Numeric Consistency Aggregation** | `scripts/run_model_comparison.py:245-257` | Report numeric consistency as both (a) average over articles containing numbers, and (b) fraction of articles where numbers were present. Explicitly display `N/A (no numbers)` rather than collapsing `None` into 0.0 or silently dropping it. |
| **2. Add Device Synchronization to Benchmarking** | `src/pretrained_baseline.py:153-157`, `scripts/run_model_comparison.py:176,188` | Insert `if device.type == 'mps': torch.mps.synchronize()` before `time.perf_counter()` start and stop calls to ensure accurate GPU kernel timing. |
| **3. Add Benchmark Warm-Up Iteration** | `scripts/run_model_comparison.py:153` | Run 1 throwaway warm-up summary through both models before starting timed benchmarking to eliminate shader compilation latency spikes. |
| **4. Clarify UI Provenance Text** | `app.py:1080` | Update string to: `Generated via Hugging Face pretrained baseline sshleifer/distilbart-cnn-12-6 (305.5M parameters). Pretrained upstream on CNN/DailyMail.` to prevent misunderstanding. |
| **5. Resolve Dependency Conflict** | `requirements.txt:10-11` | Align `datasets` and `huggingface_hub` versions (e.g. `huggingface_hub>=0.25.0,<2.0.0` or upgrade `datasets` to compatible version) to satisfy `pip check`. |
| **6. Add Pytest to Dependencies** | `requirements.txt:28` | Add `pytest>=8.0.0` so standard test runners can be used in CI/CD environments. |
| **7. Fix Fine-Tuning Numerical Stability on MPS** | `scripts/finetune_baseline.py:104,126` | Set `torch.amp.autocast('mps', dtype=torch.bfloat16)`, reduce base learning rate to `1e-5`, and add gradient clipping checks to prevent MPS NaN loss divergence. |
| **8. Modernize Streamlit CORS Config** | `.streamlit/config.toml:11` | Remove `enableCORS = false` or configure trusted origins explicitly to eliminate security warnings. |

---

## 10. Audit Execution Telemetry

- **Antigravity CLI Behavior**:
  - `agy` version: 1.4.12
  - Process Stability: 100% reliable. Zero crashes, zero unhandled tool exceptions, zero deadlocks.
  - Background Task Handling: 4 background tasks managed and resolved cleanly (`task-98`, `task-113`, `task-126`, `task-133`).
- **Audit Deliverables Generated**:
  1. `./_audit_stage1/artifact_checksums.txt` (SHA-256 and byte sizes of model weights and vocabulary)
  2. `./_audit_stage1/test_results.txt` (Full raw test suite execution log)
  3. `./_audit_stage1/AUDIT_REPORT.md` (This document)
