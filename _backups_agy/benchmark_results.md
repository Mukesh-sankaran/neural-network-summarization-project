# Empirical Benchmark Evaluation Report

- **Dataset**: CNN/DailyMail 3.0.0 (Held-out Test Set, seed 42)
- **Sample Size (n)**: 100 articles
- **Generation Parameters**: `num_beams=4`, `no_repeat_ngram_size=3`, `max_length=128`
- **Evaluation Timestamp**: 2026-11-01 02:05:38

## Benchmark Comparison Table

| Model | Parameters | n | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Words | Avg Latency (ms) |
|---|---|---|---|---|---|---|---|
| **T5-small (Zero-Shot)** | 60.5M | 100 | **0.3203** | **0.1361** | **0.2312** | 42.5 | 1786.8 ms |
| **T5-small (Fine-Tuned)** | 60.5M | 100 | **0.3447** | **0.1492** | **0.2499** | 41.2 | 1526.5 ms |
| **DistilBART-CNN-12-6 (Pretrained)** | 305.5M | 100 | **0.3871** | **0.1889** | **0.2808** | 58.9 | 5576.7 ms |
| **Custom Seq2Seq BiLSTM (Beam k=4 + Trigram Block)** | 25.2M | 100 | **0.1204** | **0.0040** | **0.1055** | 31.0 | 1180.0 ms |

## Findings & Model Comparison

1. **T5-small Fine-Tuning Result**: Zero-shot ROUGE-L was **0.2312**; fine-tuned ROUGE-L reached **0.2499** (an improvement of **+0.0187** points).
2. **DistilBART Production Baseline**: Delivers the highest grammatical fluency and ROUGE performance (ROUGE-L 0.2808) due to 305.5M parameters and pretraining.
3. **Custom Seq2Seq Research Model**: Undertrained academic model with high repetition and low ROUGE-L (0.1055), useful as an educational prototype of LSTM + Bahdanau attention.

## Honest Scientific Limitations

- **Custom Model Undertrained**: Trained for only 3 epochs on 2,000 samples, exhibiting repetition loops without large pretraining.
- **Small Evaluation Set**: Evaluated on n=100 held-out test articles for tractable demonstration runtime.
- **No Human Evaluation**: Automated ROUGE metrics assess n-gram recall against reference highlights, which can favor extractiveness over nuance; factual consistency requires human annotators or NLI models.
