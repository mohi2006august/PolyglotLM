"""Benchmark items and their scorers.

A task file is `<eval_tasks_dir>/<lang>.jsonl`, one item per line, in either form:

* multiple choice: {"type": "mc", "id": ..., "prompt": "...", "choices": [...], "answer": 0}
  Scored by the log-likelihood of each choice continuing the prompt (accuracy, 0-100).
* generative QA: {"type": "qa", "id": ..., "prompt": "...", "answers": ["...", ...]}
  Scored by greedy decoding against the reference answers (token F1 and exact match, 0-100).

Export IndicQA / FLORES / MMLU-style sets into this shape to run them through the harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class MCItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mc"] = "mc"
    id: str
    prompt: str = ""
    choices: list[str] = Field(min_length=2)
    answer: int

    @model_validator(mode="after")
    def _answer_in_range(self) -> MCItem:
        if not 0 <= self.answer < len(self.choices):
            raise ValueError(f"item {self.id}: answer {self.answer} out of range")
        return self


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["qa"] = "qa"
    id: str
    prompt: str
    answers: list[str] = Field(min_length=1)


Item = Annotated[MCItem | QAItem, Field(discriminator="type")]
_ITEM = TypeAdapter(Item)


def load_items(path: Path) -> list[MCItem | QAItem]:
    with path.open(encoding="utf-8") as fh:
        return [_ITEM.validate_python(json.loads(line)) for line in fh if line.strip()]


def task_path(tasks_dir: str | None, lang: str) -> Path | None:
    if not tasks_dir:
        return None
    path = Path(tasks_dir) / f"{lang}.jsonl"
    return path if path.exists() else None
