"""Grow the vocabulary dimension of a causal LM without touching existing rows."""

from __future__ import annotations

from typing import Any

from polyglotlm.logs import get_logger
from polyglotlm.modeling import embeddings_tied

log = get_logger(__name__)


def vocab_rows(model: Any) -> tuple[int, int | None]:
    """Row counts of (input embeddings, LM head). The head is None for headless models."""
    head = model.get_output_embeddings()
    return model.get_input_embeddings().weight.shape[0], (
        None if head is None else head.weight.shape[0]
    )


def resize_embeddings(model: Any, new_vocab_size: int) -> tuple[int, int]:
    """Resize input embeddings and the LM head to `new_vocab_size`. Returns (old, new) sizes.

    Both matrices must grow: an untied head left at the old size would make the model unable to
    ever predict a new token. Tied models share one matrix, so a single resize covers both.
    """
    old_size, _ = vocab_rows(model)
    if new_vocab_size < old_size:
        raise ValueError(f"cannot shrink the vocabulary ({old_size} -> {new_vocab_size})")
    tied = embeddings_tied(model)
    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    rows_in, rows_out = vocab_rows(model)
    if rows_in != new_vocab_size or rows_out not in (None, new_vocab_size):
        raise RuntimeError(
            f"resize left input={rows_in}, lm_head={rows_out}; expected {new_vocab_size} for both"
        )
    if tied and not embeddings_tied(model):
        raise RuntimeError("resize broke the weight tying between embeddings and LM head")
    model.config.vocab_size = new_vocab_size
    log.info("embeddings_resized", old=old_size, new=new_vocab_size, tied=tied)
    return old_size, new_vocab_size
