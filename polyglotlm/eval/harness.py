"""Evaluate checkpoints on every language, English included.

Per language: held-out bits per character (comparable across tokenizers), token perplexity
(comparable only between checkpoints that share a tokenizer) and benchmark scores from
`<eval_tasks_dir>/<lang>.jsonl`. Evaluation is deterministic: eval mode, no sampling, fixed
document and item order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from polyglotlm.config import TrainingRecipe
from polyglotlm.eval.tasks import MCItem, QAItem, load_items, task_path
from polyglotlm.eval.tasks.scoring import greedy_continuation, score_mc, score_qa
from polyglotlm.logs import get_logger
from polyglotlm.modeling import load_model, pick_device, set_seed
from polyglotlm.tokenizer import load_tokenizer

log = get_logger(__name__)

STAGES = ("cpt", "sft")
SAMPLE_PROMPT_WORDS = 6
SAMPLE_NEW_TOKENS = 24


def checkpoints(recipe: TrainingRecipe) -> dict[str, str]:
    """The base model plus every trained stage that exists, in pipeline order."""
    found = {"base": recipe.base_model}
    for stage in STAGES:
        path = recipe.paths.model(stage)
        if (path / "config.json").exists():
            found[stage] = str(path)
    return found


@torch.no_grad()
def doc_cross_entropy(
    model: Any, tokenizer: Any, docs: list[str], max_tokens: int
) -> tuple[float, int, int]:
    """(total nats, predicted tokens, characters covered) over documents, each after an EOS."""
    nats, tokens, chars = 0.0, 0, 0
    for doc in docs:
        ids = tokenizer.encode(doc, add_special_tokens=False)[: max_tokens - 1]
        if not ids:
            continue
        tensor = torch.tensor([[tokenizer.eos_token_id, *ids]], device=model.device)
        logprobs = torch.log_softmax(model(input_ids=tensor).logits[0, :-1].float(), dim=-1)
        nats -= logprobs.gather(1, torch.tensor(ids, device=model.device)[:, None]).sum().item()
        tokens += len(ids)
        chars += len(tokenizer.decode(ids, skip_special_tokens=True))
    return nats, tokens, chars


def _read_docs(path: Path, max_docs: int) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in islice(fh, max_docs) if line.strip()]


def task_scores(model: Any, tokenizer: Any, items: list[MCItem | QAItem]) -> dict[str, Any]:
    mc = [score_mc(model, tokenizer, item) for item in items if isinstance(item, MCItem)]
    qa = [score_qa(model, tokenizer, item) for item in items if isinstance(item, QAItem)]
    scores: dict[str, Any] = {"n_mc": len(mc), "n_qa": len(qa)}
    if mc:
        scores["mc_accuracy"] = 100.0 * sum(mc) / len(mc)
    if qa:
        scores["qa_f1"] = 100.0 * sum(f1 for f1, _ in qa) / len(qa)
        scores["qa_exact"] = 100.0 * sum(em for _, em in qa) / len(qa)
    return scores


def headline(scores: dict[str, Any]) -> float | None:
    """The single benchmark number reported per language: MC accuracy, else QA F1."""
    return scores.get("mc_accuracy", scores.get("qa_f1"))


def evaluate_language(
    model: Any,
    tokenizer: Any,
    recipe: TrainingRecipe,
    lang: str,
    max_docs: int,
    max_items: int | None,
) -> dict[str, Any]:
    max_tokens = getattr(model.config, "max_position_embeddings", 2048)
    docs = _read_docs(recipe.heldout_path(lang), max_docs)
    nats, tokens, chars = doc_cross_entropy(model, tokenizer, docs, max_tokens)
    result: dict[str, Any] = {
        "bits_per_char": nats / math.log(2) / chars if chars else None,
        "token_ppl": math.exp(nats / tokens) if tokens else None,
        "tokens_per_char": tokens / chars if chars else None,
    }
    if docs:
        prompt = " ".join(docs[0].split()[:SAMPLE_PROMPT_WORDS])
        result["sample"] = {
            "prompt": prompt,
            "output": greedy_continuation(model, tokenizer, prompt, SAMPLE_NEW_TOKENS),
        }
    if path := task_path(recipe.eval_tasks_dir, lang):
        result.update(task_scores(model, tokenizer, load_items(path)[:max_items]))
    result["score"] = headline(result)
    return result


def evaluate_checkpoint(
    recipe: TrainingRecipe,
    name: str,
    model_path: str,
    max_docs: int = 200,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Score one checkpoint on all languages and write `eval/<name>.json`."""
    set_seed(recipe.seed)
    model = load_model(model_path).to(pick_device()).eval()
    tokenizer = load_tokenizer(model_path)
    report = {
        "name": name,
        "model": model_path,
        "vocab": len(tokenizer),
        "languages": {
            lang: evaluate_language(model, tokenizer, recipe, lang, max_docs, max_items)
            for lang in recipe.all_codes
        },
    }
    out = recipe.paths.eval(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info(
        "checkpoint_evaluated", name=name, **{k: v["score"] for k, v in report["languages"].items()}
    )
    return report


def english_monitor(
    task: Path, model: Any, tokenizer: Any, max_items: int = 50
) -> Callable[[], float]:
    """English multiple-choice accuracy (points) for the in-training regression alarm."""
    items = [item for item in load_items(task) if isinstance(item, MCItem)][:max_items]
    if not items:
        raise ValueError(f"{task} has no multiple-choice items for the English monitor")

    def score() -> float:
        was_training = model.training
        model.eval()
        try:
            return 100.0 * sum(score_mc(model, tokenizer, item) for item in items) / len(items)
        finally:
            model.train(was_training)

    return score
