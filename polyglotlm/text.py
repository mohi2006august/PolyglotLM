"""Unicode helpers shared by every stage: NFC normalisation, script detection, word splitting."""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from collections import Counter

# ISO 15924 script code -> inclusive code point ranges.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Latn": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
    "Arab": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "Deva": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Beng": ((0x0980, 0x09FF),),
    "Guru": ((0x0A00, 0x0A7F),),
    "Gujr": ((0x0A80, 0x0AFF),),
    "Orya": ((0x0B00, 0x0B7F),),
    "Taml": ((0x0B80, 0x0BFF),),
    "Telu": ((0x0C00, 0x0C7F),),
    "Knda": ((0x0C80, 0x0CFF),),
    "Mlym": ((0x0D00, 0x0D7F),),
}

# Joiners carry meaning inside Indic conjuncts and Urdu words; they belong to no script.
ZWNJ = "‌"
ZWJ = "‍"
JOINERS = frozenset({ZWNJ, ZWJ})

_RANGE_TABLE: list[tuple[int, int, str]] = sorted(
    (start, end, script) for script, ranges in SCRIPT_RANGES.items() for start, end in ranges
)
_RANGE_STARTS: list[int] = [start for start, _, _ in _RANGE_TABLE]


def nfc(text: str) -> str:
    """Canonical composition. All text is NFC-normalised at ingestion."""
    return unicodedata.normalize("NFC", text)


def char_script(ch: str) -> str | None:
    """Script code of a single character, or None if it is outside the known ranges."""
    cp = ord(ch)
    idx = bisect_right(_RANGE_STARTS, cp) - 1
    if idx < 0:
        return None
    start, end, script = _RANGE_TABLE[idx]
    return script if start <= cp <= end else None


def is_letter_or_mark(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "M")


def script_counts(text: str) -> Counter[str]:
    """Count letters and combining marks per script. Digits, punctuation and spaces are ignored."""
    counts: Counter[str] = Counter()
    for ch in text:
        if not is_letter_or_mark(ch):
            continue
        counts[char_script(ch) or "Zzzz"] += 1
    return counts


def script_ratio(text: str, script: str) -> float:
    """Share of letters/marks in `text` that belong to `script` (0.0 for text with no letters)."""
    counts = script_counts(text)
    total = sum(counts.values())
    return counts[script] / total if total else 0.0


def dominant_script(text: str) -> str | None:
    counts = script_counts(text)
    return counts.most_common(1)[0][0] if counts else None


def in_script(text: str, script: str) -> bool:
    """True if every character is a letter/mark of `script` (or a joiner), with >= 1 letter."""
    seen = False
    for ch in text:
        if ch in JOINERS:
            continue
        if char_script(ch) != script or not is_letter_or_mark(ch):
            return False
        seen = True
    return seen


def split_words(text: str) -> list[str]:
    """Whitespace word segmentation: the denominator of token fertility."""
    return text.split()
