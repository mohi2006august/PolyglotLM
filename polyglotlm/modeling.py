"""Thin helpers around torch / transformers shared by surgery, training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name_or_path: str | Path) -> Any:
    """Load a causal LM. bf16 on GPU, fp32 on CPU."""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(str(name_or_path), dtype=dtype)


def save_model(model: Any, tokenizer: Any, out_dir: Path) -> None:
    """Save weights and the matching tokenizer side by side (llama.cpp needs both)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def embeddings_tied(model: Any) -> bool:
    """True when the LM head shares its weight with the input embeddings (or has no head)."""
    head = model.get_output_embeddings()
    return head is None or head.weight is model.get_input_embeddings().weight
