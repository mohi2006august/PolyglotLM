from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from polyglotlm.config import TrainingRecipe
from polyglotlm.demo.corpus import DEMO_LANGS, DEMO_LICENCE, noisy_raw_corpus

TARGETS = {"hi": "Deva", "ta": "Taml", "ur": "Arab", "bn": "Beng"}


def write_demo_sources(root: Path, docs_per_lang: int = 200, seed: int = 0) -> Path:
    """Write noisy raw corpora plus a source manifest; return the manifest path."""
    sources = []
    for lang in DEMO_LANGS:
        raw = root / "incoming" / f"{lang}.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("\n".join(noisy_raw_corpus(lang, docs_per_lang, seed)) + "\n", "utf-8")
        sources.append(
            {"name": f"demo-{lang}", "lang": lang, "uri": str(raw), "licence": DEMO_LICENCE}
        )
    manifest = root / "sources.yaml"
    manifest.write_text(yaml.safe_dump({"sources": sources}), "utf-8")
    return manifest


def make_recipe(root: Path, **overrides) -> TrainingRecipe:
    base = {
        "base_model": str(root / "base"),
        "workdir": str(root / "work"),
        "sources": str(root / "sources.yaml"),
        "languages": [
            {"code": code, "script": script, "new_tokens": 120} for code, script in TARGETS.items()
        ],
        "heldout_fraction": 0.1,
        "lora_rank": 4,
        "embed_warmup_steps": 4,
        "cpt": {
            "max_steps": 12,
            "batch_size": 4,
            "block_size": 32,
            "lr": 1e-3,
            "embed_lr": 1e-3,
            "lr_warmup_steps": 2,
            "eval_every": 4,
            "eval_blocks": 4,
        },
        "sft": {
            "max_steps": 8,
            "batch_size": 4,
            "block_size": 48,
            "lr": 1e-3,
            "embed_lr": 1e-3,
            "lr_warmup_steps": 2,
            "eval_every": 4,
            "eval_blocks": 4,
        },
    }
    return TrainingRecipe.model_validate({**base, **overrides})


@pytest.fixture
def recipe(tmp_path: Path) -> TrainingRecipe:
    """A recipe over freshly written noisy raw corpora (nothing built yet)."""
    write_demo_sources(tmp_path)
    return make_recipe(tmp_path)
