"""End-to-end CPU smoke run of the whole recipe on synthetic data (`poly demo run`).

`init` writes everything the tiny-demo recipe points at: noisy raw corpora and their manifest,
a tiny pre-trained base model, English SFT seeds, acceptability eval items and parallel TSVs.
`run` then executes every stage in order. It proves the plumbing, not the method: the corpora
are template sentences and the model has two layers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from polyglotlm.config import TrainingRecipe
from polyglotlm.demo.base import create_base
from polyglotlm.demo.corpus import (
    DEMO_LANGS,
    DEMO_LICENCE,
    demo_eval_items,
    demo_seed_pairs,
    noisy_raw_corpus,
    parallel_sentences,
    sample_documents,
)
from polyglotlm.logs import get_logger

log = get_logger(__name__)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def init_demo(recipe: TrainingRecipe, docs_per_lang: int = 600) -> None:
    """Write the synthetic inputs the tiny-demo recipe expects."""
    root = recipe.paths.workdir
    sources = []
    for lang in DEMO_LANGS:
        raw = root / "incoming" / f"{lang}.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            "\n".join(noisy_raw_corpus(lang, docs_per_lang, recipe.seed)) + "\n", "utf-8"
        )
        sources.append(
            {"name": f"demo-{lang}", "lang": lang, "uri": str(raw), "licence": DEMO_LICENCE}
        )
    if not recipe.sources:
        raise ValueError("the demo recipe needs a `sources` path")
    Path(recipe.sources).parent.mkdir(parents=True, exist_ok=True)
    Path(recipe.sources).write_text(yaml.safe_dump({"sources": sources}), "utf-8")

    if recipe.distill.seed_path:
        _jsonl(Path(recipe.distill.seed_path), demo_seed_pairs(80, recipe.seed))
    if recipe.eval_tasks_dir:
        for lang in DEMO_LANGS:
            _jsonl(Path(recipe.eval_tasks_dir) / f"{lang}.jsonl", demo_eval_items(lang, 40, 0))
    if recipe.embed.parallel_dir:
        rows = parallel_sentences()
        for lang in recipe.codes:
            path = Path(recipe.embed.parallel_dir) / f"{lang}.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(f"{r['en']}\t{r[lang]}\n" for r in rows), "utf-8")

    targets = [doc for lang in recipe.codes for doc in sample_documents(lang, 10, recipe.seed + 1)]
    create_base(
        Path(recipe.base_model),
        english_docs=sample_documents("en", 600, recipe.seed + 1),
        target_docs=targets,
        pretrain_steps=300,
        seed=recipe.seed,
    )
    log.info("demo_initialised", workdir=str(root))


def stages(recipe: TrainingRecipe) -> list[tuple[str, Callable[[], Any]]]:
    """Every pipeline stage in order, as (name, thunk)."""
    from polyglotlm.data.pipeline import build_corpora
    from polyglotlm.distill.teacher import DemoTeacher
    from polyglotlm.distill.teacher_generate import distill_sft
    from polyglotlm.embed.distill import run_embed_distill
    from polyglotlm.eval.harness import checkpoints, evaluate_checkpoint
    from polyglotlm.eval.report import write_report
    from polyglotlm.surgery import run_surgery
    from polyglotlm.tokenizer import load_tokenizer
    from polyglotlm.tokenizer.fertility import fertility_report, save_report
    from polyglotlm.tokenizer.merge_vocab import extend_for_recipe
    from polyglotlm.train.cpt import run_cpt
    from polyglotlm.train.sft import run_sft

    def fertility(name: str, path: str | Path) -> None:
        report = fertility_report(load_tokenizer(path), recipe, recipe.all_codes)
        save_report(report, recipe.paths.fertility(name))

    def evaluate() -> None:
        for name, path in checkpoints(recipe).items():
            evaluate_checkpoint(recipe, name, path, max_docs=100)

    return [
        ("data", lambda: build_corpora(recipe, recipe.codes)),
        ("fertility (base)", lambda: fertility("base", recipe.base_model)),
        (
            "tokenizer extend",
            lambda: extend_for_recipe(recipe, load_tokenizer(recipe.base_model), recipe.codes),
        ),
        ("fertility (extended)", lambda: fertility("extended", recipe.paths.extended_tokenizer)),
        ("surgery", lambda: run_surgery(recipe)),
        ("cpt", lambda: run_cpt(recipe)),
        ("distill", lambda: distill_sft(recipe, DemoTeacher())),
        ("sft", lambda: run_sft(recipe)),
        ("eval", evaluate),
        ("report", lambda: write_report(recipe)),
        ("embed", lambda: run_embed_distill(recipe)),
    ]


def run_demo(recipe: TrainingRecipe, on_stage: Callable[[str], None] | None = None) -> Path:
    """Initialise if needed, run every stage, return the report path."""
    if not (Path(recipe.base_model) / "config.json").exists():
        init_demo(recipe)
    for name, step in stages(recipe):
        if on_stage:
            on_stage(name)
        step()
    return recipe.paths.report / "index.html"
