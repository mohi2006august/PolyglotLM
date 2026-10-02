"""Automatic QA for teacher outputs: script check, length sanity and back-translation agreement."""

from __future__ import annotations

import re

from polyglotlm.distill.chrf import chrf
from polyglotlm.distill.teacher import Pair
from polyglotlm.text import nfc, script_ratio

_CODE_OR_URL = re.compile(r"```.*?```|`[^`]*`|https?://\S+", re.DOTALL)

# Translations legitimately change length across scripts; outside this band something is wrong.
MIN_LENGTH_RATIO = 0.25
MAX_LENGTH_RATIO = 4.0


def normalise_pair(pair: Pair) -> Pair:
    return Pair(instruction=nfc(pair.instruction).strip(), response=nfc(pair.response).strip())


def pair_text(pair: Pair) -> str:
    return f"{pair.instruction}\n{pair.response}"


def script_problem(pair: Pair, script: str, min_ratio: float) -> str | None:
    """Reject empty fields and text that is mostly outside the target script (code excluded)."""
    if not pair.instruction or not pair.response:
        return "empty_field"
    prose = _CODE_OR_URL.sub(" ", pair_text(pair))
    return "wrong_script" if script_ratio(prose, script) < min_ratio else None


def length_problem(candidate: Pair, source: Pair) -> str | None:
    ratio = len(pair_text(candidate)) / max(1, len(pair_text(source)))
    return None if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO else "length_ratio"


def back_translation_score(back_translation: Pair, source: Pair) -> float:
    """chrF between the English back-translation and the original English source."""
    return chrf(pair_text(back_translation), pair_text(source))
