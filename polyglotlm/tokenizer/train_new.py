"""Train a target-language SentencePiece model and count how often each piece is used.

The SentencePiece model is only a source of candidate subwords. It never tokenizes text for the
LLM; `merge_vocab` decides which candidates are appended to the base tokenizer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import sentencepiece as spm

from polyglotlm.logs import get_logger

log = get_logger(__name__)

WORD_START = "▁"  # SentencePiece marks a piece that begins a word with this character


def candidate_vocab_size(new_tokens: int) -> int:
    """Train more pieces than we intend to keep: many will already exist in the base vocab."""
    return max(2 * new_tokens, new_tokens + 500)


def train_candidates(
    corpus: Path,
    model_prefix: Path,
    vocab_size: int,
    model_type: Literal["unigram", "bpe"] = "unigram",
) -> Path:
    """Train SentencePiece on one document-per-line NFC text. Returns the `.model` path."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name="identity",  # text is already NFC; never NFKC-fold scripts
        hard_vocab_limit=False,  # small corpora cannot fill the requested vocabulary
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        num_threads=1,  # single-threaded EM keeps the candidate set reproducible
        minloglevel=2,
    )
    model = model_prefix.with_suffix(".model")
    log.info("candidates_trained", corpus=str(corpus), model=str(model), vocab_size=vocab_size)
    return model


def count_pieces(model: Path, corpus: Path) -> Counter[str]:
    """Frequency of every piece when the corpus is segmented with the candidate model."""
    processor = spm.SentencePieceProcessor(model_file=str(model))
    counts: Counter[str] = Counter()
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            counts.update(processor.encode(line.strip(), out_type=str))
    return counts
