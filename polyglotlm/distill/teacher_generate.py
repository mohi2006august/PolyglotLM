"""Build the SFT set: English seeds, teacher translations and native generations, all QA'd.

Per target language the budget splits into translated seeds and instructions generated directly
in the language (the main defence against "translationese"). Every candidate passes a script
check; translations also need back-translation chrF above `min_chrf`. A sample of accepted
items is queued for native-speaker review (see `review.py`); rejected items are kept with their
reason so the filters can be audited.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from polyglotlm.config import ENGLISH, DistillConfig, TrainingRecipe
from polyglotlm.data.download import ALLOWED_LICENCES, LicenceError
from polyglotlm.distill.back_translate import (
    back_translation_score,
    length_problem,
    normalise_pair,
    script_problem,
)
from polyglotlm.distill.teacher import Pair, Teacher, TeacherError
from polyglotlm.logs import get_logger

log = get_logger(__name__)

Origin = Literal["seed", "translated", "native"]
Review = Literal["none", "pending", "approved", "rejected"]


class SFTExample(BaseModel):
    id: str
    lang: str
    origin: Origin
    instruction: str
    response: str
    source: Pair | None = None  # English original of a translation
    back_translation: Pair | None = None
    chrf: float | None = None
    review: Review = "none"
    reason: str | None = None  # why a rejected example was dropped


def example_id(lang: str, pair: Pair) -> str:
    digest = hashlib.sha1(f"{lang}\0{pair.instruction}".encode(), usedforsecurity=False)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Task:
    kind: Literal["translate", "native"]
    lang: str
    seed: Pair | None = None
    topic: str | None = None
    n: int = 1
    index: int = 0


@dataclass
class DistillReport:
    accepted: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)
    queued_for_review: int = 0
    teacher_errors: int = 0


def load_seeds(path: Path) -> list[Pair]:
    with path.open(encoding="utf-8") as fh:
        return [normalise_pair(Pair.model_validate_json(line)) for line in fh if line.strip()]


def plan_tasks(langs: list[str], seeds: list[Pair], cfg: DistillConfig, seed: int) -> list[Task]:
    """Split `cfg.n` examples evenly over languages, then into translated and native work."""
    rng = random.Random(seed)
    per_lang = max(1, cfg.n // len(langs))
    tasks: list[Task] = []
    for lang in langs:
        n_native = round(per_lang * cfg.native_fraction) if cfg.topics else 0
        n_translate = per_lang - n_native if seeds else 0
        order = rng.sample(range(len(seeds)), len(seeds)) if seeds else []
        tasks += [
            Task("translate", lang, seed=seeds[order[i % len(order)]], index=i)
            for i in range(n_translate)
        ]
        for i, start in enumerate(range(0, n_native, cfg.pairs_per_call)):
            topic = cfg.topics[(i + rng.randrange(len(cfg.topics))) % len(cfg.topics)]
            n = min(cfg.pairs_per_call, n_native - start)
            tasks.append(Task("native", lang, topic=topic, n=n, index=i))
    return tasks


def _check_translation(teacher: Teacher, task: Task, script: str, cfg: DistillConfig) -> SFTExample:
    assert task.seed is not None
    pair = normalise_pair(teacher.translate(task.seed, task.lang))
    example = SFTExample(
        id=example_id(task.lang, pair),
        lang=task.lang,
        origin="translated",
        source=task.seed,
        **pair.model_dump(),
    )
    if reason := script_problem(pair, script, cfg.min_script_ratio) or length_problem(
        pair, task.seed
    ):
        return example.model_copy(update={"reason": reason})
    back = teacher.back_translate(pair, task.lang)
    score = back_translation_score(back, task.seed)
    reason = "back_translation" if score < cfg.min_chrf else None
    return example.model_copy(update={"back_translation": back, "chrf": score, "reason": reason})


def _check_native(
    teacher: Teacher, task: Task, script: str, cfg: DistillConfig
) -> list[SFTExample]:
    assert task.topic is not None
    examples = []
    for raw in teacher.generate_native(task.lang, task.topic, task.n, task.index):
        pair = normalise_pair(raw)
        reason = script_problem(pair, script, cfg.min_script_ratio)
        back = None if reason else teacher.back_translate(pair, task.lang)
        examples.append(
            SFTExample(
                id=example_id(task.lang, pair),
                lang=task.lang,
                origin="native",
                back_translation=back,
                reason=reason,
                **pair.model_dump(),
            )
        )
    return examples


def run_task(teacher: Teacher, task: Task, script: str, cfg: DistillConfig) -> list[SFTExample]:
    """One teacher round trip. Teacher failures are logged and counted, never swallowed silently."""
    try:
        if task.kind == "translate":
            return [_check_translation(teacher, task, script, cfg)]
        return _check_native(teacher, task, script, cfg)
    except TeacherError as exc:
        log.warning("teacher_error", kind=task.kind, lang=task.lang, error=str(exc))
        return []


def _mark_for_review(examples: list[SFTExample], fraction: float, seed: int) -> list[SFTExample]:
    rng = random.Random(seed)
    return [
        ex.model_copy(update={"review": "pending"}) if rng.random() < fraction else ex
        for ex in examples
    ]


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row.model_dump_json(exclude_none=True) + "\n")


def _audit_seed_licence(cfg: DistillConfig) -> None:
    if cfg.seed_path and cfg.seed_licence not in ALLOWED_LICENCES:
        raise LicenceError(
            f"seed data licence {cfg.seed_licence!r} is not on the allowlist; "
            "set `distill.seed_licence` in the recipe"
        )


def distill_sft(
    recipe: TrainingRecipe, teacher: Teacher, n: int | None = None, langs: list[str] | None = None
) -> DistillReport:
    """Generate, QA and write the SFT set plus rejected items and review queues."""
    cfg = recipe.distill if n is None else recipe.distill.model_copy(update={"n": n})
    _audit_seed_licence(cfg)
    seeds = load_seeds(Path(cfg.seed_path)) if cfg.seed_path else []
    langs = langs or recipe.codes
    tasks = plan_tasks(langs, seeds, cfg, recipe.seed)
    log.info("distill_start", teacher=teacher.name, tasks=len(tasks), seeds=len(seeds))
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        results = list(
            pool.map(lambda t: run_task(teacher, t, recipe.script_of(t.lang), cfg), tasks)
        )
    report = DistillReport(teacher_errors=sum(1 for r in results if not r))
    accepted: dict[str, SFTExample] = {}
    rejected: list[SFTExample] = []
    for example in (ex for batch in results for ex in batch):
        if example.reason or example.id in accepted:
            rejected.append(
                example if example.reason else example.model_copy(update={"reason": "duplicate"})
            )
            report.rejected[example.reason or "duplicate"] += 1
        else:
            accepted[example.id] = example
            report.accepted[example.lang] += 1
    generated = _mark_for_review(list(accepted.values()), cfg.review_fraction, recipe.seed)
    english = list(
        {
            example_id(ENGLISH, s): SFTExample(
                id=example_id(ENGLISH, s), lang=ENGLISH, origin="seed", **s.model_dump()
            )
            for s in seeds
        }.values()
    )
    report.accepted[ENGLISH] = len(english)
    write_jsonl(recipe.paths.sft_data, [*english, *generated])
    write_jsonl(recipe.paths.sft_rejected, rejected)
    for lang in langs:
        queue = [ex for ex in generated if ex.lang == lang and ex.review == "pending"]
        write_jsonl(recipe.paths.review_queue(lang), queue)
        report.queued_for_review += len(queue)
    log.info(
        "distill_done",
        accepted=dict(report.accepted),
        rejected=dict(report.rejected),
        review=report.queued_for_review,
        teacher_errors=report.teacher_errors,
    )
    return report


def load_examples(path: Path) -> list[SFTExample]:
    with path.open(encoding="utf-8") as fh:
        return [SFTExample.model_validate(json.loads(line)) for line in fh if line.strip()]


def summary(report: DistillReport) -> dict[str, Any]:
    return {
        "accepted": dict(report.accepted),
        "rejected": dict(report.rejected),
        "queued_for_review": report.queued_for_review,
        "teacher_errors": report.teacher_errors,
    }
