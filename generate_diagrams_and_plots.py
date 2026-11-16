"""
Script to generate all project visualization PNG files using matplotlib:
- ARCHITECTURE_DIAGRAM.png
- PROJECT_FLOWCHART.png
- TRAINING_RESULTS.png
- EVALUATION_RESULTS.png

All values are loaded directly from results/training_history.json and results/evaluation_report.json.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------
# 1. GENERATE TRAINING_RESULTS.png
# ---------------------------------------------------------
with open("results/training_history.json", "r") as f:
    th = json.load(f)

epochs = th["epochs"]
train_loss = th["train_loss"]
val_loss = th["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Training Loss")
plt.plot(epochs, val_loss, marker="s", linewidth=2, label="Validation Loss")
plt.title("Seq2Seq Model Training & Validation Loss (3 Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("TRAINING_RESULTS.png", dpi=300)
plt.savefig("results/training_validation_loss.png", dpi=300)
plt.close()
print("Saved TRAINING_RESULTS.png")


# ---------------------------------------------------------
# 2. GENERATE EVALUATION_RESULTS.png
# ---------------------------------------------------------
with open("results/evaluation_report.json", "r") as f:
    ev = json.load(f)

oa = ev["overall_averages"]

fig, ax = plt.subplots(figsize=(10, 5))
metrics = [
    "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1",
    "Entity Precision", "Entity Recall", "Entity Pres. Rate", "Numeric Cons."
]
values = [
    oa["rouge1_f1"], oa["rouge2_f1"], oa["rougel_f1"],
    oa["entity_precision"], oa["entity_recall"], oa["entity_preservation_rate"],
    oa["numeric_consistency_rate"]
]

bars = ax.bar(metrics, values, width=0.5)
ax.set_ylabel("Score (0.0 to 1.0)")
ax.set_title("Overall Evaluation Metrics Baseline (20 Test Examples)")
ax.set_ylim(0, 1.1)
ax.grid(True, axis="y")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)

plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("EVALUATION_RESULTS.png", dpi=300)
plt.savefig("results/evaluation_results.png", dpi=300)
plt.close()
print("Saved EVALUATION_RESULTS.png")


# ---------------------------------------------------------
# 3. GENERATE ARCHITECTURE_DIAGRAM.png
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

boxes = [
    ("CNN/DailyMail Parquet Dataset", 0.5, 0.95),
    ("Text Preprocessing (Clean, Lowercase, Whitespace)", 0.5, 0.88),
    ("Custom Word Tokenization", 0.5, 0.81),
    ("20,000 Word Vocabulary (models/vocab.pkl)", 0.5, 0.74),
    ("Trainable Word Embedding (128-dim)", 0.5, 0.67),
    ("Bidirectional LSTM Encoder (256-dim fwd + 256-dim bwd = 512-dim)", 0.5, 0.60),
    ("Bahdanau Additive Attention (Masked Padding, Context 512-dim)", 0.5, 0.53),
    ("LSTM Decoder (256-dim hidden, Step-by-Step)", 0.5, 0.46),
    ("Linear Output Layer (896 -> 20,000 Vocab Logits)", 0.5, 0.39),
    ("Generated Summary Output", 0.5, 0.32),
    ("Evaluation Layer:\nROUGE (1, 2, L) + Entity Preservation + Factual Consistency", 0.5, 0.23),
]

for text, x, y in boxes:
    ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"), fontsize=10)

# Connect boxes with arrows
for i in range(len(boxes) - 1):
    x1, y1 = boxes[i][1], boxes[i][2] - 0.02
    x2, y2 = boxes[i+1][1], boxes[i+1][2] + 0.02
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

# Add Training Loop feedback arrow
ax.text(0.85, 0.32, "Cross Entropy Loss\n(ignore_index=PAD_IDX)", ha="center", va="center", bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="black"), fontsize=9)
ax.annotate("", xy=(0.85, 0.39), xytext=(0.58, 0.32), arrowprops=dict(arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=-0.2"))

ax.text(0.85, 0.60, "Backpropagation &\nAdam Optimizer Step", ha="center", va="center", bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="black"), fontsize=9)
ax.annotate("", xy=(0.85, 0.55), xytext=(0.85, 0.37), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.58, 0.67), xytext=(0.85, 0.65), arrowprops=dict(arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=-0.2"))

plt.title("Custom NNDL Seq2Seq Neural Network Architecture & Training Loop", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("ARCHITECTURE_DIAGRAM.png", dpi=300)
plt.close()
print("Saved ARCHITECTURE_DIAGRAM.png")


# ---------------------------------------------------------
# 4. GENERATE PROJECT_FLOWCHART.png
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 12))
ax.axis("off")

steps = [
    "1. Local Dataset Loading\n(CNN/DailyMail & XSum Parquet files)",
    "2. Exploratory Data Analysis (EDA)\n(Length distributions & statistics)",
    "3. Text Preprocessing & Cleaning\n(Whitespace, punctuation, case normalization)",
    "4. Custom Word-Level Vocabulary\n(20,000 size, built ONLY from training set)",
    "5. PyTorch Dataset & DataLoader\n(Source=500, Target=60, torch.long tensors)",
    "6. PyTorch Neural Network Implementation\n(Embedding -> BiLSTM -> Bahdanau Attention -> Decoder)",
    "7. Pre-Training Sanity Testing\n(10-step verification of shapes & backward pass)",
    "8. First Training Experiment\n(500 train, 2 epochs - Revealed period repetition problem)",
    "9. EOS & Alignment Diagnosis\n(Discovered truncation cut off <EOS> in 39% target seqs)",
    "10. Pipeline Bug Fix\n(Pre-truncation to 58 tokens so 100% target seqs end with <EOS>)",
    "11. Second Training Experiment\n(2,000 train, 3 epochs - 100% EOS generation & real words)",
    "12. Multi-Metric Evaluation Framework\n(ROUGE-1/2/L + Entity Preservation + Factual Consistency)",
    "13. Current Baseline Results & Visualization\n(100% Numeric consistency, ROUGE-1 0.1479)",
    "14. Future Work & Enhancements\n(Scale training, entity-aware loss, beam search)"
]

y_positions = np.linspace(0.96, 0.04, len(steps))

for i, text in enumerate(steps):
    y = y_positions[i]
    ax.text(0.5, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"), fontsize=9)
    if i < len(steps) - 1:
        y_next = y_positions[i+1]
        ax.annotate("", xy=(0.5, y_next + 0.018), xytext=(0.5, y - 0.018), arrowprops=dict(arrowstyle="->", lw=1.2))

plt.title("Iterative Development Journey Flowchart", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("PROJECT_FLOWCHART.png", dpi=300)
plt.close()
print("Saved PROJECT_FLOWCHART.png")
