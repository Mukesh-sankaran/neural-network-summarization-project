"""
Pretrained Transformer Summarization Baseline Module.
Integrates Hugging Face pretrained models (default: sshleifer/distilbart-cnn-12-6)
for production-quality abstractive news summarization alongside the custom research model.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_PRETRAINED_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def get_pretrained_device(prefer_mps: bool = True) -> torch.device:
    """
    Determines optimal compute device for pretrained transformer inference.
    Prefers MPS on Apple Silicon if available and requested, with safe CPU fallback.
    """
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            # Test small allocation to confirm MPS is operational
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pretrained_model(
    model_name_or_path: str = DEFAULT_PRETRAINED_MODEL_NAME,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer], Optional[Dict[str, Any]], torch.device, Optional[str]]:
    """
    Safely loads pretrained Hugging Face tokenizer and Seq2Seq model.
    Never fails silently: returns descriptive error messages upon failure.

    Returns:
        (model, tokenizer, metadata, active_device, error_message)
    """
    target_device = device if device is not None else get_pretrained_device()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        return None, None, None, target_device, f"Failed to load tokenizer for '{model_name_or_path}': {str(e)}"

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        try:
            model = model.to(target_device)
        except Exception as dev_err:
            # Fall back to CPU if device placement fails (e.g., MPS OOM or unsupported kernel)
            target_device = torch.device("cpu")
            model = model.to(target_device)

        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metadata = {
            "model_name": model_name_or_path,
            "architecture": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(target_device),
            "vocab_size": getattr(tokenizer, "vocab_size", len(tokenizer)),
            "max_position_embeddings": getattr(model.config, "max_position_embeddings", 1024),
        }

        return model, tokenizer, metadata, target_device, None

    except Exception as e:
        return None, tokenizer, None, target_device, f"Failed to load model weights for '{model_name_or_path}': {str(e)}"


def generate_pretrained_summary(
    article_text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 142,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[str, List[int], float]:
    """
    Generates an abstractive summary using the pretrained transformer model.

    Args:
        article_text: Raw input news article text.
        model: Pretrained AutoModelForSeq2SeqLM instance.
        tokenizer: Pretrained AutoTokenizer instance.
        device: Active torch device (CPU/MPS/CUDA).
        max_length: Maximum summary token length.
        min_length: Minimum summary token length (automatically clamped <= max_length).
        num_beams: Number of beams for beam search (1 = greedy argmax).
        length_penalty: Exponential length penalty (Wu et al.).
        no_repeat_ngram_size: Size of n-grams prohibited from repeating (0 to disable).
        do_sample: Whether to sample probabilistically.
        temperature: Sampling temperature when do_sample=True.
        top_p: Nucleus sampling probability when do_sample=True.

    Returns:
        (summary_string, generated_token_ids, latency_milliseconds)
    """
    if not article_text or not article_text.strip():
        return "", [], 0.0

    # Ensure length constraints are mathematically consistent
    effective_max = max(10, max_length)
    effective_min = min(min_length, effective_max - 2) if effective_max > 12 else 5
    if effective_min < 1:
        effective_min = 1

    inputs = tokenizer(
        article_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Build generation kwargs
    gen_kwargs: Dict[str, Any] = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_length": effective_max,
        "min_length": effective_min,
        "no_repeat_ngram_size": no_repeat_ngram_size if no_repeat_ngram_size > 0 else 0,
        "early_stopping": True if num_beams > 1 else False,
    }

    if num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["length_penalty"] = float(length_penalty)
        gen_kwargs["do_sample"] = False
    elif do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["num_beams"] = 1

    t0 = time.perf_counter()
    with torch.no_grad():
        summary_tensor = model.generate(**gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    token_ids: List[int] = summary_tensor[0].tolist()
    decoded = tokenizer.decode(summary_tensor[0], skip_special_tokens=True).strip()

    return decoded, token_ids, latency_ms
