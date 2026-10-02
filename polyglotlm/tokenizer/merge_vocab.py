"""Select new tokens by tokens saved and append them to the base tokenizer.

Ranking follows TECHNICAL.md: `gain = freq * (base_tokens(t) - 1)`, the number of tokens the
corpus would save if `t` were a single token. The top-N candidates are appended after the last
existing ID, so every base ID keeps its meaning (append-only vocabulary).

New tokens are registered as non-special added tokens. The tokenizer matches them
leftmost-longest before running its subword model, and only on text in the target script, so
English tokenization is untouched.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from tokenizers import AddedToken

from polyglotlm.config import TrainingRecipe
from polyglotlm.logs import get_logger
from polyglotlm.text import in_script, nfc
from polyglotlm.tokenizer import count_tokens
from polyglotlm.tokenizer.train_new import (
    WORD_START,
    candidate_vocab_size,
    count_pieces,
    train_candidates,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class Candidate:
    text: str  # string added to the tokenizer; a leading space marks a word-initial token
    lang: str
    freq: int
    base_tokens: int

    @property
    def gain(self) -> int:
        return self.freq * (self.base_tokens - 1)


def piece_to_text(piece: str) -> str | None:
    """Map a SentencePiece piece to the raw string it covers, or None if it is unusable."""
    core = piece.removeprefix(WORD_START)
    if not core or WORD_START in core:
        return None
    return (" " + core) if piece.startswith(WORD_START) else core


def select_new_tokens(
    piece_counts: Counter[str], base_tokenizer: Any, lang: str, script: str, top_n: int
) -> list[Candidate]:
    """Rank in-script pieces by tokens saved and return the best `top_n` with positive gain."""
    candidates = []
    for piece, freq in piece_counts.items():
        text = piece_to_text(piece)
        if text is None or not in_script(text.strip(), script) or text != nfc(text):
            continue
        candidate = Candidate(text, lang, freq, count_tokens(base_tokenizer, text))
        if candidate.gain > 0:
            candidates.append(candidate)
    candidates.sort(key=lambda c: (-c.gain, c.text))
    return candidates[:top_n]


def extend_tokenizer(base_tokenizer: Any, candidates: list[Candidate]) -> Any:
    """Copy of the base tokenizer with the candidates appended; the base is not modified."""
    extended = copy.deepcopy(base_tokenizer)
    known = extended.get_vocab()
    new = list(dict.fromkeys(c.text for c in candidates if c.text not in known))
    extended.add_tokens([AddedToken(text, normalized=False, special=False) for text in new])
    verify_append_only(base_tokenizer, extended, expected_new=len(new))
    return extended


def verify_append_only(base_tokenizer: Any, extended_tokenizer: Any, expected_new: int) -> None:
    """Raise if any base ID changed or the new IDs are not a contiguous block at the end."""
    base_vocab, extended_vocab = base_tokenizer.get_vocab(), extended_tokenizer.get_vocab()
    moved = [tok for tok, idx in base_vocab.items() if extended_vocab.get(tok) != idx]
    if moved:
        raise AssertionError(f"{len(moved)} base token IDs changed, e.g. {moved[:5]}")
    new_ids = sorted(idx for tok, idx in extended_vocab.items() if tok not in base_vocab)
    start = len(base_tokenizer)
    if new_ids != list(range(start, start + expected_new)):
        raise AssertionError("new token IDs are not appended contiguously after the base vocab")


def extend_for_recipe(
    recipe: TrainingRecipe,
    base_tokenizer: Any,
    langs: list[str],
    new_tokens: int | None = None,
) -> tuple[Any, list[Candidate]]:
    """Train candidates per language, select, merge, and save the extended tokenizer."""
    selected: list[Candidate] = []
    for lang in langs:
        config = recipe.language(lang)
        top_n = new_tokens if new_tokens is not None else config.new_tokens
        corpus = recipe.paths.corpus(lang)
        if not corpus.exists():
            raise FileNotFoundError(f"no corpus for {lang!r} at {corpus}; run the data stage")
        model = train_candidates(
            corpus,
            recipe.paths.candidates(lang) / "spm",
            candidate_vocab_size(top_n),
        )
        chosen = select_new_tokens(
            count_pieces(model, corpus), base_tokenizer, lang, config.script, top_n
        )
        log.info(
            "tokens_selected",
            lang=lang,
            requested=top_n,
            selected=len(chosen),
            est_tokens_saved=sum(c.gain for c in chosen),
        )
        selected.extend(chosen)
    extended = extend_tokenizer(base_tokenizer, selected)
    out_dir = recipe.paths.extended_tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    extended.save_pretrained(str(out_dir))
    (out_dir / "new_tokens.json").write_text(
        json.dumps([{**asdict(c), "gain": c.gain} for c in selected], ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    log.info("tokenizer_extended", path=str(out_dir), base=len(base_tokenizer), new=len(selected))
    return extended, selected
