"""Tokenizer-aware speed benchmarks.

Tokens per second hides the token tax: the same 8 tok/s is 1.7 words/s for a tokenizer that
needs 4.8 tokens per Hindi word and 4.2 words/s for one that needs 1.9. Benchmarks therefore
report words per second per language, using the fertility measured on held-out text.

`llama-bench -o json` output (from a desktop build or an Android device via adb) is parsed into
generation and prompt-processing speeds.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyglotlm.quant.gguf import LlamaCpp
from polyglotlm.tokenizer.fertility import Fertility

TARGET_TOKENS_PER_SECOND = 8.0  # PRD: 3B Q4 on a mid-range Android phone


@dataclass(frozen=True)
class Speed:
    prompt_tps: float | None  # prompt processing, tokens/s
    gen_tps: float | None  # generation, tokens/s


def parse_llama_bench(records: list[dict[str, Any]]) -> Speed:
    """Read `llama-bench -o json`: rows with n_gen > 0 are generation, n_prompt > 0 prompt."""
    prompt, gen = None, None
    for row in records:
        if "avg_ts" not in row:
            raise ValueError(f"not llama-bench JSON output (no avg_ts): {sorted(row)}")
        if row.get("n_gen", 0) > 0 and not row.get("n_prompt"):
            gen = float(row["avg_ts"])
        elif row.get("n_prompt", 0) > 0 and not row.get("n_gen"):
            prompt = float(row["avg_ts"])
    if gen is None and prompt is None:
        raise ValueError("llama-bench output had neither a prompt nor a generation test")
    return Speed(prompt_tps=prompt, gen_tps=gen)


def run_llama_bench(
    tools: LlamaCpp, gguf: Path, threads: int, n_prompt: int = 128, n_gen: int = 64
) -> Speed:
    cmd = [*tools.bench, "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_gen)]
    cmd += ["-t", str(threads), "-o", "json"]
    output = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return parse_llama_bench(json.loads(output))


def words_per_second(speed: Speed, fertility: dict[str, Fertility]) -> dict[str, dict[str, Any]]:
    """Per-language generation speed in words/s, and whether it meets the tok/s target."""
    if speed.gen_tps is None:
        raise ValueError("no generation speed in the benchmark")
    return {
        lang: {
            "tokens_per_word": round(row.tokens_per_word, 3),
            "gen_tokens_per_s": round(speed.gen_tps, 2),
            "gen_words_per_s": round(speed.gen_tps / row.tokens_per_word, 2),
            "meets_target": speed.gen_tps >= TARGET_TOKENS_PER_SECOND,
        }
        for lang, row in fertility.items()
        if row.tokens_per_word
    }
