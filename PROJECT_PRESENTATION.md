# Long Document Summarization with Factual/Entity Preservation
**Academic Progress Presentation (12 Slides)**

---

## Slide 1: Project Title & Overview

### Slide Title
**Long Document Summarization with Factual/Entity Preservation**

### Bullet Points
* **Student Name / Project ID**: NNDL Academic Progress Demonstration
* **Core Objective**: Build a custom, non-pretrained Sequence-to-Sequence (Seq2Seq) neural network for long-document text summarization.
* **Key Innovations**:
  * Scratch PyTorch implementation (Embedding + BiLSTM Encoder + Bahdanau Attention + LSTM Decoder).
  * Explicit factual and named entity preservation evaluation framework.
  * Local data loading pipeline handling Parquet datasets (CNN/DailyMail and XSum).

### Suggested Diagram / Image
`ARCHITECTURE_DIAGRAM.png`

### Verbal Script for Student
*"Good morning, Professor. Today I am presenting our NNDL academic project titled 'Long Document Summarization with Factual/Entity Preservation'. In this project, we built a complete Sequence-to-Sequence neural network from scratch using PyTorch primitives. Our goal is not just to generate fluent summaries, but to ensure that critical factual information—such as names, dates, numbers, and monetary figures—is accurately preserved without hallucination."*

---

## Slide 2: Problem Statement

### Slide Title
**The Challenge of Long Document Summarization**

### Bullet Points
* **Document Compression Gap**: Articles in CNN/DailyMail average **813+ tokens**, while target summaries average **58 tokens** (over 14x compression).
* **Information Loss**: Standard seq2seq architectures struggle to maintain long-range dependencies, often dropping key named entities.
* **Factual Distortion**: Neural generators tend to distort numerical quantities, dates, and proper nouns during auto-regressive decoding.
* **Sequence Termination Failures**: Inconsistent end-of-sequence (`<EOS>`) targets can cause models to collapse into un-terminated token repetition.

### Suggested Diagram / Image
`PROJECT_FLOWCHART.png` (Steps 1–2)

### Verbal Script for Student
*"When compressing an 800-word news article into a 3-sentence summary, standard neural networks face two major challenges. First, key details like person names and numerical statistics are frequently dropped or distorted. Second, if sequence termination is not handled correctly during preprocessing, the decoder can get stuck in repetitive token loops. Our project systematically addresses both modeling and evaluation challenges."*

---

## Slide 3: Why Existing Summarization Evaluation Is Not Enough

### Slide Title
**Beyond ROUGE: The Need for Entity & Factual Metrics**

### Bullet Points
* **ROUGE Limitations**: Standard ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) measures n-gram and surface string overlap, not factual accuracy.
* **False Positives**: A summary can achieve high ROUGE scores while swapping crucial numbers or misattributing quotes.
* **Proposed Multi-Metric Evaluation**:
  1. **ROUGE Overlap**: ROUGE-1, ROUGE-2, and ROUGE-L F1/Precision/Recall.
  2. **spaCy Named Entity Metrics**: Entity Precision, Entity Recall, and Entity Preservation Rate.
  3. **Factual/Numeric Consistency Rate**: Verification of numbers, percentages, dates, and currency values.

### Suggested Diagram / Image
`EVALUATION_RESULTS.png`

### Verbal Script for Student
*"Traditional summarization projects rely solely on ROUGE scores. However, ROUGE only measures word overlap and cannot tell if a date or financial figure was hallucinated. To solve this, we implemented a dedicated evaluation layer using spaCy to measure Entity Preservation Rate alongside a Numeric Consistency metric that checks whether every generated number exists in the original source document."*

---

## Slide 4: Dataset Analysis & Statistics

### Slide Title
**Dataset Overview & Local Parquet Pipeline**

### Bullet Points
* **Primary Dataset**: CNN/DailyMail Version 3.0.0 (311,971 total samples).
  * Train: **287,113** examples (736.65 MB across 3 Parquet shards).
  * Validation: **13,368** examples (33.06 MB).
  * Test: **11,490** examples (28.61 MB).
* **Secondary Dataset**: XSum (226,711 total samples).
* **Local Memory Efficiency**: Parquet files are streamed and sampled deterministically using pandas/pyarrow without loading full datasets into RAM.
* **Prototype Configuration**: 2,000 Train / 300 Validation / 100 Test samples.

### Suggested Diagram / Image
`PROJECT_FLOWCHART.png` (Step 1)

### Verbal Script for Student
*"We use the benchmark CNN/DailyMail dataset containing 311,971 news articles paired with human-written highlight summaries. To handle this large dataset efficiently on local hardware, we built a data loader using PyArrow and Pandas that reads Parquet files directly from disk with deterministic random sampling, ensuring zero data leakage between splits."*

---

## Slide 5: Data Preprocessing & Vocabulary

### Slide Title
**Text Normalization & Custom Tokenization**

### Bullet Points
* **Text Preprocessing**: Lowercasing, whitespace normalization, and smart punctuation separation.
* **Factual Unit Preservation**: Preserves decimals (`3.14`), dates (`2026-08-21`), percentages (`15.5%`), and currencies (`$100`, `£795`).
* **Custom Tokenizer**: Pure Python word-level tokenizer (NO pretrained tokenizers or Hugging Face dependency).
* **Vocabulary Construction**:
  * Size: **20,000** unique tokens.
  * Built **strictly from training data** (`train_df`) to prevent data leakage.
  * Fixed Special Tokens: `<PAD>` (0), `<UNK>` (1), `<SOS>` (2), `<EOS>` (3).

### Suggested Diagram / Image
`PROJECT_FLOWCHART.png` (Steps 3–5)

### Verbal Script for Student
*"Our preprocessing pipeline cleans and normalizes text while carefully protecting numerical values, dates, and percentages. We built a custom word-level tokenizer and constructed a 20,000-word vocabulary exclusively from the training dataset. We enforce strict indices for four special tokens: PAD at index 0, UNK at index 1, SOS at index 2, and EOS at index 3."*

---

## Slide 6: Proposed Neural Network Architecture

### Slide Title
**Custom Seq2Seq Neural Network (25.2M Parameters)**

### Bullet Points
* **Scratch Implementation**: Built entirely using PyTorch `torch.nn` primitives.
* **Total Parameters**: **25,229,600** trainable parameters.
* **Sequence Constraints**: `MAX_SOURCE_LENGTH = 500` tokens, `MAX_TARGET_LENGTH = 60` tokens.
* **Core Components**:
  1. Trainable Word Embedding ($128$-dim).
  2. Bidirectional LSTM Encoder ($256$-dim forward + $256$-dim backward = $512$-dim output).
  3. Manual Bahdanau Additive Attention with Padding Mask ($512$-dim context).
  4. Attention-Augmented LSTM Decoder ($256$-dim hidden).
  5. Linear Vocabulary Output Projection ($896 \rightarrow 20,000$).

### Suggested Diagram / Image
`ARCHITECTURE_DIAGRAM.png`

### Verbal Script for Student
*"Here is our core neural network architecture. It contains 25.2 million trainable parameters. The model passes source tokens through a 128-dimensional embedding, encodes them using a Bidirectional LSTM, applies Bahdanau Additive Attention to compute a dynamic context vector, and feeds that vector into an LSTM Decoder to project 20,000 vocabulary logits step-by-step."*

---

## Slide 7: BiLSTM Encoder & Bahdanau Attention

### Slide Title
**Bidirectional Encoding & Masked Additive Attention**

### Bullet Points
* **BiLSTM Encoder**:
  * Reads source document in forward and backward directions simultaneously.
  * Concatenated output dimension: $256 + 256 = 512$.
  * Linear projection layers (`fc_hidden`, `fc_cell`) convert 512-dim encoder states into 256-dim decoder initialization states.
* **Bahdanau Additive Attention**:
  * Score equation: $\text{energy} = \tanh(\mathbf{W}_{\text{enc}} \cdot h_{\text{enc}} + \mathbf{W}_{\text{dec}} \cdot s_{\text{dec}})$.
  * **Padding Masking**: Masked with $-1e9$ before Softmax so the model never attends to `<PAD>` positions.
  * Produces a dynamic $512$-dimensional context vector $c_t$ at every generation step.

### Suggested Diagram / Image
`ARCHITECTURE_DIAGRAM.png` (Encoder & Attention Blocks)

### Verbal Script for Student
*"The Encoder uses a Bidirectional LSTM so that every word has context from both past and future tokens. For the attention mechanism, we manually implemented Bahdanau Additive Attention. Crucially, we apply a padding mask that sets attention scores for PAD tokens to negative one billion before Softmax, ensuring the decoder focuses exclusively on actual article content."*

---

## Slide 8: Training Pipeline & Optimizer

### Slide Title
**Training Strategy & Hardware Acceleration**

### Bullet Points
* **Loss Function**: `torch.nn.CrossEntropyLoss(ignore_index=0)` operating on raw unnormalized logits.
* **Teacher Forcing**: Applied during training with `teacher_forcing_ratio = 0.5`.
* **Optimizer**: Adam ($\text{lr} = 0.001$) with Gradient Clipping ($\text{max\_norm} = 1.0$) to stabilize BPTT.
* **Hardware Acceleration**: Apple Silicon MPS GPU (`mps` device).
* **Checkpointing**: Best model saved automatically to `models/best_model.pt` based on validation loss.

### Suggested Diagram / Image
`TRAINING_RESULTS.png`

### Verbal Script for Student
*"We train the network using the Adam optimizer with a learning rate of 0.001 and gradient clipping set to 1.0. We use Cross-Entropy Loss ignoring PAD tokens, and apply Teacher Forcing 50% of the time during training. The pipeline automatically leverages Apple Silicon MPS GPU acceleration and saves the best model checkpoint whenever validation loss improves."*

---

## Slide 9: Development Problem & Debugging

### Slide Title
**Empirical Diagnosis & Sequence Termination Fix**

### Bullet Points
* **Problem in Experiment 1**: Initial 2-epoch run produced repeated period summaries (`. . . . .`) during greedy decoding.
* **Diagnostic Investigation**: Created `test_training_diagnostics.py` to inspect target preprocessing.
* **Root Cause Discovered**: `<EOS>` was appended *before* slicing target tokens (`tgt_indices[:60]`). Because **39.00%** of target summaries exceeded 60 tokens, **`<EOS>` was cut off for 39% of training samples**!
* **Correction Applied**: Pre-truncated content tokens to 58 tokens *before* adding `<SOS>` and `<EOS>`.
* **Verification Result**: **`100.00%` of target sequences now contain `<EOS>`**, restoring real content word generation.

### Suggested Diagram / Image
`PROJECT_FLOWCHART.png` (Steps 8–11)

### Verbal Script for Student
*"An important highlight of our development journey was debugging a sequence termination issue. In our first experiment, greedy decoding generated repeated periods. By building a diagnostic script, we discovered that simple truncation had cut off the EOS token in 39% of target sequences! Once we fixed the pre-truncation logic so that 100% of targets ended with EOS, the model successfully learned sequence termination and began generating real content words."*

---

## Slide 10: Multi-Metric Evaluation Framework

### Slide Title
**Comprehensive Evaluation Layer**

### Bullet Points
* **ROUGE Metrics** (`src/evaluate.py`):
  * ROUGE-1 (Unigram F1, Precision, Recall).
  * ROUGE-2 (Bigram F1, Precision, Recall).
  * ROUGE-L (Longest Common Subsequence F1, Precision, Recall).
* **Entity Metrics** (`src/entity_evaluation.py`):
  * spaCy `en_core_web_sm` NER covering 9 entity types (`PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `MONEY`, `PERCENT`, `CARDINAL`, `QUANTITY`).
  * Case-insensitive & whitespace-normalized Entity Precision, Recall, and Preservation Rate.
* **Factual Consistency**: Numeric Consistency Rate for numbers, percentages, dates, and currencies.

### Suggested Diagram / Image
`EVALUATION_RESULTS.png`

### Verbal Script for Student
*"Our evaluation framework, implemented in `evaluate.py` and `entity_evaluation.py`, measures performance across three complementary dimensions: standard ROUGE n-gram overlap, spaCy-based Named Entity Preservation, and Numeric Consistency. This allows us to track both fluency and factual accuracy."*

---

## Slide 11: Current Baseline Results

### Slide Title
**Empirical Performance Summary (20 Test Examples)**

### Bullet Points
* **Training Loss Progression**:
  * Epoch 1: Train = **7.3243** | Val = **6.9023**
  * Epoch 2: Train = **6.6933** | Val = **6.9266**
  * Epoch 3: Train = **6.3864** | Val = **6.8798**
* **ROUGE Baseline**: ROUGE-1 F1 = **0.1479**, ROUGE-2 F1 = **0.0107**, ROUGE-L F1 = **0.1246**.
* **Entity Preservation Rate**: **`0.0000`** (baseline on 2k sample prototype).
* **Numeric Consistency Rate**: **`1.0000`** (**`100.00%`** — zero numeric hallucinations introduced).
* **EOS Generation Success**: **`100.00%`** across test samples.

### Suggested Diagram / Image
`EVALUATION_RESULTS.png` & `TRAINING_RESULTS.png`

### Verbal Script for Student
*"On our 2,000-sample prototype trained for 3 epochs, the model achieved a training loss of 6.3864 and a validation loss of 6.8798. It achieved a ROUGE-1 score of 0.1479 and a 100% Numeric Consistency rate, meaning zero hallucinated numbers were introduced. This establishes a clean, verified baseline for our project."*

---

## Slide 12: Future Work & Next Steps

### Slide Title
**Next Steps & Planned Enhancements**

### Bullet Points
1. **Full-Scale Dataset Training**: Expand training from 2,000 prototype samples to 50,000+ examples across 10–20 epochs.
2. **Entity-Weighted Cross-Entropy Loss**: Introduce loss multipliers for named entity tokens to explicitly boost the Entity Preservation Rate.
3. **Beam Search Decoding**: Replace greedy decoding with Beam Search ($k=4$) and length normalization to improve generation fluency.
4. **Cross-Dataset Generalization**: Evaluate model performance on the XSum dataset.
5. **Interactive Dashboard**: Maintain the local project dashboard (`project_dashboard.py`) for live demonstrations.

### Suggested Diagram / Image
`PROJECT_FLOWCHART.png` (Step 14)

### Verbal Script for Student
*"Looking ahead, our next steps are: first, scaling up training from 2,000 samples to over 50,000 articles; second, introducing an entity-weighted loss function to directly penalize entity omissions; and third, implementing Beam Search decoding to improve summary coherence. Thank you, Professor. I am ready for your questions."*
