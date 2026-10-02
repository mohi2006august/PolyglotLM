"""Normalisation, PII redaction and quality/toxicity filters for raw documents."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polyglotlm.logs import get_logger
from polyglotlm.text import JOINERS, nfc, script_ratio, split_words

log = get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_JOINER_RUN = re.compile(r"[‌‍]{2,}")
_EDGE_JOINER = re.compile(r"(?:(?<=\s)|^)[‌‍]+|[‌‍]+(?=\s|$)")

# Arabic code points that Urdu text should spell with its own letters, plus tatweel removal.
_URDU_MAP = str.maketrans(
    {
        "ي": "ی",  # ARABIC YEH -> FARSI YEH
        "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
        "ك": "ک",  # ARABIC KAF -> KEHEH
        "ـ": None,  # TATWEEL (kashida) is purely typographic
    }
)


def _normalise_urdu(text: str) -> str:
    return text.translate(_URDU_MAP)


# Language-specific script normalisation applied after NFC.
SCRIPT_NORMALISERS: dict[str, Callable[[str], str]] = {"ur": _normalise_urdu}


def _drop_controls(text: str) -> str:
    """Remove control/format characters, keeping ZWJ/ZWNJ (meaningful in Indic and Urdu)."""
    return "".join(
        " " if ch in "\t\n\r" else ch
        for ch in text
        if ch in JOINERS or ch in "\t\n\r" or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def normalise(text: str, lang: str) -> str:
    """NFC, control stripping, joiner cleanup, per-language script fixes, whitespace collapse."""
    text = _drop_controls(nfc(text))
    text = _JOINER_RUN.sub(lambda m: m.group(0)[0], text)
    if normaliser := SCRIPT_NORMALISERS.get(lang):
        text = normaliser(text)
    text = _WHITESPACE.sub(" ", text).strip()
    return nfc(_EDGE_JOINER.sub("", text))


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 10-13 digits with optional separators: phone numbers and national ID numbers.
_PHONE = re.compile(r"(?<![\w.])\+?\d(?:[\s-]?\d){9,12}(?![\w.])")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

EMAIL_TAG = "<email>"
PHONE_TAG = "<phone>"


def redact_pii(text: str) -> tuple[str, int]:
    """Replace emails and phone/ID-like numbers with placeholders. Returns (text, n_redacted)."""
    text, n_email = _EMAIL.subn(EMAIL_TAG, text)
    text, n_phone = _PHONE.subn(PHONE_TAG, text)
    return text, n_email + n_phone


@dataclass(frozen=True)
class QualityFilter:
    """Heuristic document-quality rules. `reject_reason` returns None for documents to keep."""

    script: str
    min_chars: int = 20
    min_words: int = 3
    max_word_chars: int = 60
    min_script_ratio: float = 0.7
    max_digit_ratio: float = 0.3
    max_symbol_ratio: float = 0.3
    max_top_word_share: float = 0.5
    max_urls: int = 1

    def reject_reason(self, text: str) -> str | None:
        words = split_words(text)
        checks: tuple[tuple[str, Callable[[], bool]], ...] = (
            ("too_short", lambda: len(text) < self.min_chars or len(words) < self.min_words),
            ("urls", lambda: len(_URL.findall(text)) > self.max_urls),
            ("long_word", lambda: max(map(len, words), default=0) > self.max_word_chars),
            ("wrong_script", lambda: self._script_ratio(text) < self.min_script_ratio),
            ("digits", lambda: _share(text, str.isdigit) > self.max_digit_ratio),
            ("symbols", lambda: _share(text, _is_symbol) > self.max_symbol_ratio),
            ("repetitive", lambda: _top_word_share(words) > self.max_top_word_share),
        )
        return next((reason for reason, failed in checks if failed()), None)

    def _script_ratio(self, text: str) -> float:
        visible = text.replace(EMAIL_TAG, "").replace(PHONE_TAG, "")
        return script_ratio(_URL.sub("", visible), self.script)


def _share(text: str, predicate: Callable[[str], bool]) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    return sum(map(predicate, chars)) / len(chars) if chars else 0.0


def _is_symbol(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S")


def _top_word_share(words: list[str]) -> float:
    if len(words) < 6:
        return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)


class ToxicityFilter:
    """Lexicon-based blocklist. With no lexicon it keeps everything and says so in the logs."""

    def __init__(self, terms: frozenset[str]) -> None:
        self.terms = terms

    @classmethod
    def from_dir(cls, lexicon_dir: str | None, lang: str) -> ToxicityFilter:
        path = Path(lexicon_dir) / f"{lang}.txt" if lexicon_dir else None
        if path is None or not path.exists():
            log.warning("toxicity_lexicon_missing", lang=lang, path=str(path) if path else None)
            return cls(frozenset())
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls(frozenset(nfc(line).strip().casefold() for line in lines if line.strip()))

    def is_toxic(self, text: str) -> bool:
        if not self.terms:
            return False
        return any(
            word.strip(".,!?;:\"'()।").casefold() in self.terms for word in split_words(text)
        )
