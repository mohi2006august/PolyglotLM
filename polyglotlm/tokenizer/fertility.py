"""Token fertility: average tokens per whitespace word, the per-language "token tax".

Fertility is measured on held-out native text, never on the text used to pick new tokens.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polyglotlm.config import TrainingRecipe
from polyglotlm.logs import get_logger
from polyglotlm.text import nfc, split_words
from polyglotlm.tokenizer import count_tokens

log = get_logger(__name__)


@dataclass(frozen=True)
class Fertility:
    lang: str
    docs: int
    words: int
    tokens: int

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words if self.words else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tokens_per_word": round(self.tokens_per_word, 4)}


def measure_fertility(tokenizer: Any, lang: str, docs: Iterable[str]) -> Fertility:
    n_docs = words = tokens = 0
    for doc in docs:
        text = nfc(doc).strip()
        if not text:
            continue
        n_docs += 1
        words += len(split_words(text))
        tokens += count_tokens(tokenizer, text)
    return Fertility(lang=lang, docs=n_docs, words=words, tokens=tokens)


def _read_docs(path: Path, max_docs: int | None) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    return lines if max_docs is None else lines[:max_docs]


def fertility_report(
    tokenizer: Any, recipe: TrainingRecipe, langs: list[str], max_docs: int | None = None
) -> dict[str, Fertility]:
    """Fertility per language on its held-out file."""
    report = {}
    for lang in langs:
        path = recipe.heldout_path(lang)
        if not path.exists():
            raise FileNotFoundError(f"no held-out text for {lang!r} at {path}; run the data stage")
        report[lang] = measure_fertility(tokenizer, lang, _read_docs(path, max_docs))
        log.info("fertility", **report[lang].as_dict())
    return report


def save_report(report: dict[str, Fertility], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {lang: row.as_dict() for lang, row in report.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_report(path: Path) -> dict[str, Fertility]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        lang: Fertility(lang=lang, docs=row["docs"], words=row["words"], tokens=row["tokens"])
        for lang, row in payload.items()
    }


def saving(base: Fertility, extended: Fertility) -> float:
    """Relative change in tokens per word; -0.6 means 60% fewer tokens."""
    if not base.tokens_per_word:
        return 0.0
    return extended.tokens_per_word / base.tokens_per_word - 1.0


def render_table(base: dict[str, Fertility], extended: dict[str, Fertility] | None = None) -> str:
    """Plain-text token tax table: tokens per word before, after and the saving."""
    header = f"{'Token tax (tokens per word)':<30}{'base':>8}"
    if extended:
        header += f"{'extended':>11}{'saving':>9}"
    lines = [header]
    for lang, row in base.items():
        line = f"{lang:<30}{row.tokens_per_word:>8.2f}"
        if extended and lang in extended:
            after = extended[lang]
            line += f"{after.tokens_per_word:>11.2f}{saving(row, after):>+9.0%}"
        lines.append(line)
    return "\n".join(lines)
