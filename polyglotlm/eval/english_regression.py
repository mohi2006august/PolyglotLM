"""Base vs adapted comparison: target-language gains and the English regression budget."""

from __future__ import annotations

from typing import Any

from polyglotlm.config import ENGLISH


def _delta(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def compare(base: dict[str, Any], adapted: dict[str, Any], max_regression: float) -> dict[str, Any]:
    """Per-language score and bits-per-char deltas, plus the English regression verdict.

    The verdict uses benchmark points when English tasks exist; otherwise it falls back to the
    relative change in English bits per character and says so.
    """
    languages = {}
    for lang, after in adapted["languages"].items():
        before = base["languages"].get(lang, {})
        languages[lang] = {
            "score_before": before.get("score"),
            "score_after": after.get("score"),
            "score_delta": _delta(after.get("score"), before.get("score")),
            "bpc_before": before.get("bits_per_char"),
            "bpc_after": after.get("bits_per_char"),
            "bpc_delta": _delta(after.get("bits_per_char"), before.get("bits_per_char")),
        }
    english = languages.get(ENGLISH, {})
    if english.get("score_delta") is not None:
        drop = -english["score_delta"]
        verdict = {"metric": "points", "drop": drop, "regressed": drop > max_regression}
    elif english.get("bpc_delta") is not None and english["bpc_before"]:
        change = english["bpc_delta"] / english["bpc_before"] * 100.0
        verdict = {"metric": "bpc_change_pct", "drop": change, "regressed": change > max_regression}
    else:
        verdict = {"metric": None, "drop": None, "regressed": False}
    return {
        "base": base["name"],
        "adapted": adapted["name"],
        "max_regression": max_regression,
        "english": verdict,
        "languages": languages,
    }
