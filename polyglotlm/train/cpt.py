"""FR-03 continued pre-training: LoRA on every linear layer plus trainable embeddings and LM head,
on packed native text mixed by temperature sampling with English replay.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from polyglotlm.config import TrainingRecipe
from polyglotlm.logs import get_logger
from polyglotlm.modeling import load_model, pick_device, save_model, set_seed
from polyglotlm.surgery import SURGERY, run_surgery
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.train.blocks import BlockSet, tokenize_corpus
from polyglotlm.train.guard import OverfitGuard
from polyglotlm.train.loop import (
    Batch,
    MetricsWriter,
    Trainer,
    TrainResult,
    apply_lora,
    evaluate_languages,
    trainable_groups,
)
from polyglotlm.train.mixing import MixtureSampler, mixture

log = get_logger(__name__)

CPT = "cpt"


def load_block_sets(
    recipe: TrainingRecipe, tokenizer: Any, split: str, block_size: int
) -> dict[str, BlockSet]:
    """Packed blocks for every language (targets + English) that has a `split` file."""
    sets = {}
    for lang in recipe.all_codes:
        path = recipe.paths.corpus(lang) if split == "train" else recipe.heldout_path(lang)
        if not path.exists():
            continue
        # One cache per tokenizer size: a memory-mapped file cannot be rewritten on Windows.
        cache = recipe.paths.workdir / "cache" / "tokens" / split / f"{lang}-v{len(tokenizer)}.bin"
        sets[lang] = BlockSet(tokenize_corpus(tokenizer, path, cache), block_size)
    missing = [lang for lang in recipe.codes if lang not in sets]
    if missing:
        raise FileNotFoundError(f"no {split} text for {missing}; run `poly data build`")
    return sets


def heldout_blocks(
    recipe: TrainingRecipe, tokenizer: Any, block_size: int, max_blocks: int
) -> dict[str, torch.Tensor]:
    """A fixed evaluation set per language, identical at every evaluation of a run."""
    sets = load_block_sets(recipe, tokenizer, "heldout", block_size)
    return {lang: blocks.head(max_blocks) for lang, blocks in sets.items()}


class MixedBatches:
    """Each batch row is a random block from a language drawn from the mixture."""

    def __init__(
        self, blocks: dict[str, BlockSet], probs: dict[str, float], batch_size: int, seed: int
    ) -> None:
        self.blocks = blocks
        self.batch_size = batch_size
        self.sampler = MixtureSampler(probs, seed)
        self._generator = torch.Generator().manual_seed(seed)

    def __call__(self, step: int) -> Batch:
        rows = []
        for lang in self.sampler.sample(self.batch_size):
            index = int(torch.randint(len(self.blocks[lang]), (1,), generator=self._generator))
            rows.append(self.blocks[lang].block(index))
        input_ids = torch.stack(rows)
        return {"input_ids": input_ids, "labels": input_ids}


def english_score_fn(
    recipe: TrainingRecipe, model: Any, tokenizer: Any
) -> Callable[[], float] | None:
    """English benchmark accuracy (points) for the in-loop regression alarm, if a task exists."""
    task = Path(recipe.eval_tasks_dir) / "en.jsonl" if recipe.eval_tasks_dir else None
    if task is None or not task.exists():
        log.warning("english_monitor_disabled", reason="no eval_tasks_dir/en.jsonl in recipe")
        return None
    from polyglotlm.eval.harness import english_monitor

    return english_monitor(task, model, tokenizer)


def run_cpt(recipe: TrainingRecipe) -> Path:
    """Surgery output -> continued pre-training -> merged model in `models/cpt`."""
    set_seed(recipe.seed)
    stage = recipe.cpt
    if not (recipe.paths.model(SURGERY) / "config.json").exists():
        run_surgery(recipe)
    tokenizer = load_tokenizer(recipe.paths.model(SURGERY))
    device = pick_device()
    model = load_model(recipe.paths.model(SURGERY)).to(device)

    train_sets = load_block_sets(recipe, tokenizer, "train", stage.block_size)
    probs = mixture(recipe, {lang: blocks.n_tokens for lang, blocks in train_sets.items()})
    log.info("cpt_mixture", **{lang: round(p, 4) for lang, p in probs.items()})
    heldout = heldout_blocks(recipe, tokenizer, stage.block_size, stage.eval_blocks)

    model = apply_lora(model, recipe.lora_rank, recipe.lora_alpha, recipe.lora_dropout)
    groups = trainable_groups(model, recipe.train_embeddings)
    writer = MetricsWriter(recipe.paths.metrics(CPT), run_name=f"{Path(recipe.workdir).name}-cpt")
    trainer = Trainer(
        model,
        groups,
        stage,
        evaluate=lambda: evaluate_languages(model, heldout, stage.batch_size, device),
        guard=OverfitGuard(recipe.codes, stage.patience, recipe.max_english_regression),
        writer=writer,
        embed_warmup_steps=recipe.embed_warmup_steps,
        device=device,
        english_score=english_score_fn(recipe, model, tokenizer),
    )
    try:
        result = trainer.fit(MixedBatches(train_sets, probs, stage.batch_size, recipe.seed))
    finally:
        writer.close()
    return save_stage(recipe, CPT, model, tokenizer, result, {"mixture": probs})


def save_stage(
    recipe: TrainingRecipe,
    stage_name: str,
    model: Any,
    tokenizer: Any,
    result: TrainResult,
    extra: dict[str, Any],
) -> Path:
    """Save the LoRA adapter (with embeddings), the merged model and a run summary."""
    out_dir = recipe.paths.model(stage_name)
    model.save_pretrained(str(out_dir / "adapter"), save_embedding_layers=recipe.train_embeddings)
    merged = model.merge_and_unload()
    save_model(merged, tokenizer, out_dir)
    summary = {"stage": stage_name, **asdict(result), **extra}
    (out_dir / f"{stage_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log.info(f"{stage_name}_done", path=str(out_dir), best_step=result.best_step)
    return out_dir
