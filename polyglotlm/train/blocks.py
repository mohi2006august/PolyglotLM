"""Packed sequences: documents are tokenized once, joined with EOS and cut into fixed blocks.

Token IDs live in a flat int32 file that is memory-mapped, so sampling a block touches only that
block. Packing wastes no positions on padding, which is what makes CPT throughput predictable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from polyglotlm.logs import get_logger

log = get_logger(__name__)

_FLUSH_TOKENS = 1 << 20


def _fingerprint(corpus: Path, tokenizer: Any) -> dict[str, Any]:
    stat = corpus.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "vocab": len(tokenizer)}


def tokenize_corpus(tokenizer: Any, corpus: Path, cache: Path) -> np.ndarray:
    """Tokenize a one-document-per-line corpus into a memory-mapped int32 array.

    The cache is reused only while the corpus file and tokenizer size are unchanged.
    """
    meta_path = cache.with_suffix(".json")
    fingerprint = _fingerprint(corpus, tokenizer)
    if cache.exists() and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) == fingerprint:
            return np.memmap(cache, dtype=np.int32, mode="r")
    cache.parent.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS token; packed sequences need a document separator")
    buffer: list[int] = []
    with corpus.open(encoding="utf-8") as src, cache.open("wb") as out:
        for line in src:
            if not (text := line.strip()):
                continue
            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            buffer.append(eos)
            if len(buffer) >= _FLUSH_TOKENS:
                np.asarray(buffer, dtype=np.int32).tofile(out)
                buffer.clear()
        np.asarray(buffer, dtype=np.int32).tofile(out)
    meta_path.write_text(json.dumps(fingerprint), encoding="utf-8")
    tokens = np.memmap(cache, dtype=np.int32, mode="r")
    log.info("corpus_tokenized", corpus=str(corpus), tokens=len(tokens))
    return tokens


class BlockSet:
    """Fixed-length blocks over one language's token stream."""

    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        if len(tokens) < 2:
            raise ValueError("a block set needs at least two tokens")
        self.tokens = tokens
        # A stream shorter than one block still yields a single (shorter) block.
        self.block_size = min(block_size, len(tokens))
        self.n_blocks = len(tokens) // self.block_size

    def __len__(self) -> int:
        return self.n_blocks

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def block(self, index: int) -> torch.Tensor:
        start = index * self.block_size
        return torch.from_numpy(np.array(self.tokens[start : start + self.block_size], np.int64))

    def batch(self, indices: list[int]) -> torch.Tensor:
        return torch.stack([self.block(i) for i in indices])

    def head(self, max_blocks: int) -> torch.Tensor:
        """The first `max_blocks` blocks: a fixed evaluation set."""
        return self.batch(list(range(min(max_blocks, self.n_blocks))))
