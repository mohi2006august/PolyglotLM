"""FR-01 tokenizer extension and the fertility (token tax) report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polyglotlm.config import TrainingRecipe


def load_tokenizer(name_or_path: str | Path) -> Any:
    """Load a fast Hugging Face tokenizer from a local directory or the Hub."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(name_or_path))


def resolve_tokenizer(name: str, recipe: TrainingRecipe) -> Any:
    """`base` and `extended` are recipe-relative aliases; anything else is a path or Hub id."""
    if name == "base":
        return load_tokenizer(recipe.base_model)
    if name == "extended":
        return load_tokenizer(recipe.paths.extended_tokenizer)
    return load_tokenizer(name)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
