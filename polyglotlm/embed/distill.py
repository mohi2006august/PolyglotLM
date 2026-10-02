"""Multilingual embedding distillation (Reimers & Gurevych, 2020).

A frozen English teacher defines the embedding space. The student (the adapted LLM with LoRA
and a projection) learns to map both an English sentence and its translation onto the
teacher's vector for the English sentence:

    loss = MSE(student(en), teacher(en)) + MSE(student(xx), teacher(en))

Parallel pairs come for free from the distilled SFT set (every translated example keeps its
English source) plus optional `<parallel_dir>/<lang>.tsv` files. Quality is cross-lingual
retrieval on held-out pairs: for each target sentence, is its English translation the nearest
English sentence?
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from polyglotlm.config import TrainingRecipe
from polyglotlm.embed.encoder import LMEncoder, SentenceTransformerEncoder, TextEncoder
from polyglotlm.logs import get_logger
from polyglotlm.modeling import load_model, pick_device, set_seed
from polyglotlm.text import nfc
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.train.loop import apply_lora

log = get_logger(__name__)


@dataclass(frozen=True)
class ParallelPair:
    lang: str
    english: str
    target: str


def pairs_from_sft(path: Path) -> list[ParallelPair]:
    """Translated SFT examples, instruction and response each, with their English source."""
    pairs = []
    if not path.exists():
        return pairs
    with path.open(encoding="utf-8") as fh:
        for row in map(json.loads, fh):
            if row.get("origin") != "translated" or not row.get("source"):
                continue
            for field in ("instruction", "response"):
                pairs.append(ParallelPair(row["lang"], row["source"][field], row[field]))
    return pairs


def pairs_from_dir(parallel_dir: str | None, langs: list[str]) -> list[ParallelPair]:
    pairs = []
    for lang in langs if parallel_dir else []:
        path = Path(parallel_dir) / f"{lang}.tsv"  # type: ignore[arg-type]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                english, target = line.split("\t", 1)
                pairs.append(ParallelPair(lang, nfc(english).strip(), nfc(target).strip()))
    return pairs


def split_pairs(
    pairs: list[ParallelPair], fraction: float, seed: int
) -> tuple[list[ParallelPair], list[ParallelPair]]:
    """Held-out by English sentence, so no English sentence is on both sides of the split."""

    def held(english: str) -> bool:
        digest = hashlib.sha1(f"{seed}:{english}".encode(), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], "big") / 2**32 < fraction

    unique = list(dict.fromkeys(pairs))
    return [p for p in unique if not held(p.english)], [p for p in unique if held(p.english)]


def retrieval_scores(
    encoder: TextEncoder, pairs: list[ParallelPair]
) -> dict[str, dict[str, float]]:
    """Per language: accuracy@1 and MRR of finding the English translation of each target."""
    by_lang: dict[str, list[ParallelPair]] = defaultdict(list)
    for pair in pairs:
        by_lang[pair.lang].append(pair)
    scores = {}
    for lang, group in sorted(by_lang.items()):
        english = list(dict.fromkeys(p.english for p in group))
        index = {text: i for i, text in enumerate(english)}
        candidates = encoder.encode(english)
        queries = encoder.encode([p.target for p in group])
        sims = queries @ candidates.T
        gold = torch.tensor([index[p.english] for p in group])
        gold_sim = sims[torch.arange(len(group)), gold]
        ranks = (sims > gold_sim[:, None]).sum(dim=1) + 1
        scores[lang] = {
            "acc_at_1": float((ranks == 1).float().mean()) * 100.0,
            "mrr": float((1.0 / ranks.float()).mean()),
            "n": len(group),
            "candidates": len(english),
        }
    return scores


def build_teacher(recipe: TrainingRecipe) -> TextEncoder:
    name = recipe.embed.teacher
    if name == "base":
        model = load_model(recipe.base_model).to(pick_device()).eval()
        return LMEncoder(model, load_tokenizer(recipe.base_model), None, recipe.embed.max_length)
    return SentenceTransformerEncoder(name)


def _student(recipe: TrainingRecipe, source: Path, teacher_dim: int) -> LMEncoder:
    model = load_model(source).to(pick_device())
    model = apply_lora(model, recipe.lora_rank, recipe.lora_alpha, recipe.lora_dropout)
    hidden = model.config.hidden_size
    projection = torch.nn.Linear(hidden, teacher_dim).to(model.device)
    if hidden == teacher_dim:  # start from the student's own space
        with torch.no_grad():
            projection.weight.copy_(torch.eye(hidden))
            projection.bias.zero_()
    return LMEncoder(model, load_tokenizer(source), projection, recipe.embed.max_length)


def _source_model(recipe: TrainingRecipe) -> Path:
    for stage in ("sft", "cpt"):
        if (recipe.paths.model(stage) / "config.json").exists():
            return recipe.paths.model(stage)
    raise FileNotFoundError("no adapted model; run `poly train cpt` (and `poly train sft`) first")


def run_embed_distill(recipe: TrainingRecipe) -> Path:
    """Distil the English teacher into the adapted model; save the encoder and its scores."""
    set_seed(recipe.seed)
    cfg = recipe.embed
    pairs = pairs_from_sft(recipe.paths.sft_data) + pairs_from_dir(cfg.parallel_dir, recipe.codes)
    train, heldout = split_pairs(pairs, cfg.heldout_fraction, recipe.seed)
    if not train or not heldout:
        raise ValueError(f"need parallel pairs on both sides of the split (got {len(pairs)})")
    teacher = build_teacher(recipe)
    english = list(dict.fromkeys(p.english for p in train))
    targets = dict(zip(english, teacher.encode(english), strict=True))
    student = _student(recipe, _source_model(recipe), teacher.dim)
    before = retrieval_scores(student, heldout)

    params = [p for p in student.model.parameters() if p.requires_grad]
    params += list(student.projection.parameters())  # type: ignore[union-attr]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    rng = random.Random(recipe.seed)
    losses: list[float] = []
    student.model.train()
    for step in range(cfg.max_steps):
        batch = rng.sample(train, min(cfg.batch_size, len(train)))
        goal = torch.stack([targets[p.english] for p in batch]).to(student.model.device)
        loss = torch.nn.functional.mse_loss(
            student.forward([p.english for p in batch]), goal
        ) + torch.nn.functional.mse_loss(student.forward([p.target for p in batch]), goal)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % cfg.eval_every == 0:
            log.info("embed_step", step=step, loss=round(loss.item(), 5))
    student.model = student.model.merge_and_unload()
    after = retrieval_scores(student, heldout)
    out_dir = recipe.paths.embedder
    student.save(out_dir)
    summary: dict[str, Any] = {
        "teacher": cfg.teacher,
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "final_loss": losses[-1],
        "before": before,
        "after": after,
    }
    (out_dir / "embed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("embed_done", path=str(out_dir), **{k: v["acc_at_1"] for k, v in after.items()})
    return out_dir
