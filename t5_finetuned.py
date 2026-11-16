"""
T5 Fine-tuned Summarization Engine Module.
Loads and runs inference on the local fine-tuned t5-small model (models/t5_small_finetuned).
Provides CPU fallback, generate(), and info() methods in the same style as pretrained_baseline.py.
Handles missing checkpoints gracefully with honest UI error messages.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_T5_MODEL_PATH = "models/t5_small_finetuned"
PREFIX = "summarize: "


def get_t5_device(prefer_mps: bool = True) -> torch.device:
    """
    Determines optimal compute device for T5 inference.
    Prefers MPS on Apple Silicon if available with safe CPU fallback.
    """
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        try:
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_t5_model(
    model_path: str = DEFAULT_T5_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer], Optional[Dict[str, Any]], torch.device, Optional[str]]:
    """
    Safely loads fine-tuned T5 tokenizer and model from local path.
    If the directory is missing or incomplete, returns a descriptive error instead of crashing.

    Returns:
        (model, tokenizer, metadata, active_device, error_message)
    """
    target_device = device if device is not None else get_t5_device()

    if not os.path.exists(model_path):
        return (
            None,
            None,
            None,
            target_device,
            f"Fine-tuned T5 checkpoint folder not found at '{model_path}'. "
            f"Run finetune_t5.py to train and save the model before selecting this engine."
        )

    # Check for presence of essential files
    has_weights = (
        os.path.exists(os.path.join(model_path, "model.safetensors"))
        or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
    )
    if not has_weights:
        return (
            None,
            None,
            None,
            target_device,
            f"Checkpoint weights missing in '{model_path}'. Ensure fine-tuning completed successfully."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        return None, None, None, target_device, f"Failed to load T5 tokenizer from '{model_path}': {str(e)}"

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        try:
            model = model.to(target_device)
        except Exception:
            target_device = torch.device("cpu")
            model = model.to(target_device)

        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Load optional training summary
        train_summary = {}
        summary_file = os.path.join(model_path, "training_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    train_summary = json.load(f)
            except Exception:
                pass

        metadata = {
            "model_name": "t5-small (fine-tuned)",
            "model_path": model_path,
            "architecture": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(target_device),
            "vocab_size": getattr(tokenizer, "vocab_size", len(tokenizer)),
            "training_summary": train_summary,
            "description": "t5-small fine-tuned on CNN/DailyMail (1 epoch, fp32)",
        }

        return model, tokenizer, metadata, target_device, None

    except Exception as e:
        return None, tokenizer, None, target_device, f"Failed to load T5 model from '{model_path}': {str(e)}"


def generate_t5_summary(
    article_text: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 128,
    min_length: int = 20,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    prefix: str = PREFIX,
) -> Tuple[str, List[int], float]:
    """
    Generates an abstractive summary using the fine-tuned T5 model.

    Returns:
        (summary_string, generated_token_ids, latency_milliseconds)
    """
    if not article_text or not article_text.strip():
        return "", [], 0.0

    input_text = article_text.strip()
    if not input_text.startswith(prefix.strip()):
        input_text = prefix + input_text

    effective_max = max(10, max_length)
    effective_min = min(min_length, effective_max - 2) if effective_max > 12 else 5
    if effective_min < 1:
        effective_min = 1

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

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


# Module-level convenience wrapper conforming to generate() and info() requirement
class T5Engine:
    """Wrapper providing direct generate() and info() methods."""
    def __init__(self, model_path: str = DEFAULT_T5_MODEL_PATH):
        self.model_path = model_path
        self.model, self.tokenizer, self.meta, self.device, self.error = load_t5_model(model_path)

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def info(self) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "available": False,
                "error": self.error,
                "model_path": self.model_path,
            }
        return {
            "available": True,
            **self.meta
        }

    def generate(
        self,
        article_text: str,
        max_length: int = 128,
        min_length: int = 20,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        **kwargs
    ) -> Tuple[str, List[int], float]:
        if not self.is_available():
            raise RuntimeError(f"T5 Engine unavailable: {self.error}")
        return generate_t5_summary(
            article_text=article_text,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs
        )


# Global instances / aliases
_default_engine: Optional[T5Engine] = None

def get_engine() -> T5Engine:
    global _default_engine
    if _default_engine is None:
        _default_engine = T5Engine()
    return _default_engine

def generate(article_text: str, **kwargs) -> Tuple[str, List[int], float]:
    return get_engine().generate(article_text, **kwargs)

def info() -> Dict[str, Any]:
    return get_engine().info()
