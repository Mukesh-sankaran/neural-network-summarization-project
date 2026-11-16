"""
Pretrained Transformer Summarization Baseline Root Wrapper.
Re-exports from src.pretrained_baseline for root-level script compatibility.
"""

from src.pretrained_baseline import (
    DEFAULT_PRETRAINED_MODEL_NAME,
    get_pretrained_device,
    load_pretrained_model,
    generate_pretrained_summary,
)

__all__ = [
    "DEFAULT_PRETRAINED_MODEL_NAME",
    "get_pretrained_device",
    "load_pretrained_model",
    "generate_pretrained_summary",
]
