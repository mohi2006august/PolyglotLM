"""Instruction tuning on the distilled SFT set (FR-05), with the same guard as CPT (FR-06).

Loss is computed on response tokens only. Batches mix languages by temperature sampling over
example counts plus the English replay share. Early stopping watches held-out *native text*
perplexity, never the translated SFT data itself.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from polyglotlm.config import TrainingRecipe
from polyglotlm.distill.teacher_generate import SFTExample, load_examples
from polyglotlm.logs import get_logger
from polyglotlm.modeling import load_model, pick_device, set_seed
from polyglotlm.prompts import format_prompt
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.train.cpt import CPT, english_score_fn, heldout_blocks, save_stage
from polyglotlm.train.guard import OverfitGuard
from polyglotlm.train.loop import (
    Batch,
    MetricsWriter,
    Trainer,
    apply_lora,
    evaluate_languages,
    trainable_groups,
)
from polyglotlm.train.mixing import MixtureSampler, mixture

log = get_logger(__name__)

SFT = "sft"
IGNORE_INDEX = -100


def encode_example(
    tokenizer: Any, example: SFTExample, max_length: int
) -> tuple[list[int], list[int]]:
    """Token IDs and labels; prompt positions are masked so only the response is learned."""
    prompt = tokenizer.encode(
        format_prompt(tokenizer, example.instruction), add_special_tokens=False
    )
    response = [
        *tokenizer.encode(example.response, add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    ids = (prompt + response)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt) + response)[:max_length]
    return ids, labels


def collate(rows: list[tuple[list[int], list[int]]], pad_id: int) -> Batch:
    """Right-pad to the longest row; padded positions are masked from attention and loss."""
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


class SFTBatches:
    """Language-mixed batches of encoded examples."""

    def __init__(
        self,
        encoded: dict[str, list[tuple[list[int], list[int]]]],
        probs: dict[str, float],
        batch_size: int,
        pad_id: int,
        seed: int,
    ) -> None:
        self.encoded = encoded
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.sampler = MixtureSampler(probs, seed)
        self._rng = random.Random(seed)

    def __call__(self, step: int) -> Batch:
        langs = self.sampler.sample(self.batch_size)
        return collate([self._rng.choice(self.encoded[lang]) for lang in langs], self.pad_id)


def encode_dataset(
    tokenizer: Any, examples: list[SFTExample], max_length: int
) -> dict[str, list[tuple[list[int], list[int]]]]:
    grouped: dict[str, list[tuple[list[int], list[int]]]] = defaultdict(list)
    for example in examples:
        ids, labels = encode_example(tokenizer, example, max_length)
        if any(label != IGNORE_INDEX for label in labels):  # the prompt alone filled the window
            grouped[example.lang].append((ids, labels))
    return dict(grouped)


def run_sft(recipe: TrainingRecipe) -> Path:
    """CPT model + distilled SFT data -> instruction-tuned merged model in `models/sft`."""
    set_seed(recipe.seed)
    stage = recipe.sft
    source = recipe.paths.model(CPT)
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"no CPT model at {source}; run `poly train cpt` first")
    if not recipe.paths.sft_data.exists():
        raise FileNotFoundError(f"no SFT data at {recipe.paths.sft_data}; run `poly distill sft`")
    tokenizer = load_tokenizer(source)
    device = pick_device()
    model = load_model(source).to(device)

    examples = [ex for ex in load_examples(recipe.paths.sft_data) if ex.review != "rejected"]
    encoded = encode_dataset(tokenizer, examples, stage.block_size)
    probs = mixture(recipe, {lang: len(rows) for lang, rows in encoded.items()}, use_explicit=False)
    log.info("sft_mixture", examples=len(examples), **{k: round(v, 4) for k, v in probs.items()})
    heldout = heldout_blocks(recipe, tokenizer, recipe.cpt.block_size, stage.eval_blocks)

    model = apply_lora(model, recipe.lora_rank, recipe.lora_alpha, recipe.lora_dropout)
    groups = trainable_groups(model, recipe.train_embeddings)
    writer = MetricsWriter(recipe.paths.metrics(SFT), run_name=f"{Path(recipe.workdir).name}-sft")
    trainer = Trainer(
        model,
        groups,
        stage,
        evaluate=lambda: evaluate_languages(model, heldout, stage.batch_size, device),
        guard=OverfitGuard(recipe.codes, stage.patience, recipe.max_english_regression),
        writer=writer,
        embed_warmup_steps=0,  # the new rows already settled during CPT
        device=device,
        english_score=english_score_fn(recipe, model, tokenizer),
    )
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    try:
        result = trainer.fit(SFTBatches(encoded, probs, stage.batch_size, pad_id, recipe.seed))
    finally:
        writer.close()
    return save_stage(recipe, SFT, model, tokenizer, result, {"mixture": probs})
