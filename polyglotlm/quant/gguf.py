"""Convert the final merged model to GGUF and quantise it with llama.cpp.

llama.cpp is an external checkout (LLAMA_CPP_DIR): `convert_hf_to_gguf.py` writes an f16 GGUF,
then `llama-quantize` produces the k-quant (Q4_K_M by default). Conversion reads the tokenizer
from the model directory, so the merged tokenizer files must sit next to the weights; the
preflight refuses to start otherwise, because a GGUF built without the new tokens would
silently map them to garbage.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from polyglotlm.logs import get_logger

log = get_logger(__name__)

QUANT_TYPES = frozenset(
    {"Q2_K", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16"}
)
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


@dataclass(frozen=True)
class LlamaCpp:
    """Command prefixes for the llama.cpp tools PolyglotLM drives."""

    convert: list[str]
    quantize: list[str]
    bench: list[str]

    @classmethod
    def discover(cls, root: str | Path | None = None) -> LlamaCpp:
        """Find the tools in a llama.cpp checkout built with CMake (default: $LLAMA_CPP_DIR)."""
        root = Path(root or os.environ.get("LLAMA_CPP_DIR", ""))
        if not (root / "convert_hf_to_gguf.py").exists():
            raise FileNotFoundError(
                f"no llama.cpp checkout at {root!s}; set LLAMA_CPP_DIR or pass --llama-cpp"
            )
        suffix = ".exe" if sys.platform == "win32" else ""
        candidates = [root / "build" / "bin", root / "build" / "bin" / "Release", root]

        def tool(name: str) -> list[str]:
            for folder in candidates:
                if (path := folder / f"{name}{suffix}").exists():
                    return [str(path)]
            raise FileNotFoundError(f"{name} not built in {root}; run cmake --build build")

        return cls(
            convert=[sys.executable, str(root / "convert_hf_to_gguf.py")],
            quantize=tool("llama-quantize"),
            bench=tool("llama-bench"),
        )


def preflight(model_dir: Path) -> list[str]:
    """Problems that would make the GGUF wrong or unloadable. Empty means good to go."""
    problems = []
    if not (model_dir / "config.json").exists():
        problems.append("config.json missing")
    if not any((model_dir / name).exists() for name in WEIGHT_FILES):
        problems.append("no model weights (merge the LoRA adapter first)")
    tokenizer_file = model_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        problems.append("tokenizer.json missing: save the merged tokenizer with the model")
    if problems:
        return problems
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    n_tokens = len(tokenizer["model"]["vocab"]) + sum(
        1
        for t in tokenizer.get("added_tokens", [])
        if t["content"] not in tokenizer["model"]["vocab"]
    )
    if config.get("vocab_size") != n_tokens:
        problems.append(
            f"config vocab_size {config.get('vocab_size')} != {n_tokens} tokenizer entries: "
            "the model was not resized for the extended tokenizer"
        )
    return problems


@dataclass(frozen=True)
class GGUFPlan:
    f16: Path
    quantized: Path
    convert_cmd: list[str]
    quantize_cmd: list[str]


def plan(model_dir: Path, out_dir: Path, quant: str, tools: LlamaCpp, name: str) -> GGUFPlan:
    if quant not in QUANT_TYPES:
        raise ValueError(f"unknown quant type {quant!r}; choose from {sorted(QUANT_TYPES)}")
    f16 = out_dir / f"{name}-F16.gguf"
    quantized = out_dir / f"{name}-{quant}.gguf"
    return GGUFPlan(
        f16=f16,
        quantized=quantized,
        convert_cmd=[*tools.convert, str(model_dir), "--outfile", str(f16), "--outtype", "f16"],
        quantize_cmd=[*tools.quantize, str(f16), str(quantized), quant],
    )


def quantise(model_dir: Path, out_dir: Path, quant: str, tools: LlamaCpp, name: str) -> Path:
    """Preflight, convert to f16 GGUF, quantise. Returns the quantised file."""
    if problems := preflight(model_dir):
        raise ValueError(f"{model_dir} is not ready for GGUF conversion: " + "; ".join(problems))
    steps = plan(model_dir, out_dir, quant, tools, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("gguf_convert", cmd=steps.convert_cmd)
    subprocess.run(steps.convert_cmd, check=True)
    if quant != "F16":
        log.info("gguf_quantize", cmd=steps.quantize_cmd)
        subprocess.run(steps.quantize_cmd, check=True)
    result = steps.quantized if quant != "F16" else steps.f16
    log.info("gguf_ready", path=str(result), mb=round(result.stat().st_size / 2**20, 1))
    return result
