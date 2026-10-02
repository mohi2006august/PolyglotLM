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


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> TrainingRecipe:
    """Corpora built and a tiny pre-trained base model saved. Treat as read-only."""
    from polyglotlm.data.pipeline import build_corpora
    from polyglotlm.demo.base import create_base
    from polyglotlm.demo.corpus import sample_documents

    root = tmp_path_factory.mktemp("built")
    write_demo_sources(root)
    recipe = make_recipe(root)
    build_corpora(recipe, recipe.codes)
    create_base(
        Path(recipe.base_model),
        english_docs=sample_documents("en", 300, seed=11),
        target_docs=[doc for lang in TARGETS for doc in sample_documents(lang, 8, seed=11)],
        pretrain_steps=60,
    )
    return recipe


@pytest.fixture(scope="session")
def extended(built) -> TrainingRecipe:
    """`built` plus the extended tokenizer. Treat as read-only."""
    from polyglotlm.tokenizer import load_tokenizer
    from polyglotlm.tokenizer.merge_vocab import extend_for_recipe

    extend_for_recipe(built, load_tokenizer(built.base_model), built.codes)
    return built


@pytest.fixture(scope="session")
def trained(extended) -> TrainingRecipe:
    """`extended` plus a CPT model. Treat as read-only."""
    from polyglotlm.train.cpt import run_cpt

    if not (extended.paths.model("cpt") / "config.json").exists():
        run_cpt(extended)
    return extended
