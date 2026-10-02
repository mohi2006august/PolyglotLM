"""Native-speaker (cultural) review loop.

`poly distill sft` writes `sft/review/<lang>.jsonl` with a sample of accepted examples marked
`"review": "pending"`. Reviewers edit each line: set `review` to `approved` or `rejected`, and
may correct `instruction` / `response` in place. `poly distill review` folds those decisions
back into the SFT set: rejected examples move to `rejected.jsonl`, corrections replace the
teacher's text, and untouched items stay pending (still trained on, still counted).
"""

from __future__ import annotations

from collections import Counter

from polyglotlm.config import TrainingRecipe
from polyglotlm.distill.teacher_generate import SFTExample, load_examples, write_jsonl
from polyglotlm.logs import get_logger

log = get_logger(__name__)


def apply_reviews(recipe: TrainingRecipe) -> Counter[str]:
    """Merge reviewer decisions into the SFT set. Returns counts per decision."""
    reviewed: dict[str, SFTExample] = {}
    for path in sorted((recipe.paths.workdir / "sft" / "review").glob("*.jsonl")):
        for example in load_examples(path):
            reviewed[example.id] = example
    kept: list[SFTExample] = []
    newly_rejected: list[SFTExample] = []
    counts: Counter[str] = Counter()
    for example in load_examples(recipe.paths.sft_data):
        decision = reviewed.get(example.id, example)
        counts[decision.review] += 1
        if decision.review == "rejected":
            newly_rejected.append(decision.model_copy(update={"reason": "native_review"}))
        else:
            kept.append(decision)
    previous = (
        load_examples(recipe.paths.sft_rejected) if recipe.paths.sft_rejected.exists() else []
    )
    write_jsonl(recipe.paths.sft_data, kept)
    write_jsonl(recipe.paths.sft_rejected, [*previous, *newly_rejected])
    log.info("reviews_applied", **counts)
    return counts
