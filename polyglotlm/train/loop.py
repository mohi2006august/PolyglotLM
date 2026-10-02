"""The optimisation loop shared by CPT and SFT.

Schedule: for the first `embed_warmup_steps` only the embeddings and LM head train (LoRA is
frozen), which lets freshly initialised rows settle before anything else moves. After that LoRA
joins with a linear learning-rate ramp. Every evaluation scores held-out perplexity on all
languages, English included, and asks the overfitting guard whether to checkpoint or stop.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from polyglotlm.config import StageConfig
from polyglotlm.logs import get_logger
from polyglotlm.modeling import embeddings_tied
from polyglotlm.train.guard import LossSpikeMonitor, OverfitGuard

log = get_logger(__name__)

Batch = dict[str, torch.Tensor]


@dataclass
class ParamGroups:
    lora: list[torch.nn.Parameter]
    embed: list[torch.nn.Parameter]

    def all(self) -> list[torch.nn.Parameter]:
        return [*self.lora, *self.embed]


@dataclass
class TrainResult:
    steps: int
    best_step: int | None
    stopped_early: bool
    heldout_ppl: dict[str, float]
    alerts: list[str] = field(default_factory=list)


class MetricsWriter:
    """Appends one JSON object per line; mirrors to Weights & Biases when it is configured."""

    def __init__(self, path: Path, run_name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        self._wandb = _init_wandb(run_name)

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(_flatten(record), step=record["step"])

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()


def _init_wandb(run_name: str) -> Any:
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        log.warning("wandb_not_installed", hint="pip install polyglotlm[tracking]")
        return None
    return wandb.init(project="polyglotlm", name=run_name)


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flat.update({f"{key}/{sub}": v for sub, v in value.items()})
        elif isinstance(value, (int, float)):
            flat[key] = value
    return flat


def apply_lora(model: Any, rank: int, alpha: int | None, dropout: float) -> Any:
    """Wrap the model with LoRA adapters on every linear layer except the LM head."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else 2 * rank,
        lora_dropout=dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def trainable_groups(model: Any, train_embeddings: bool) -> ParamGroups:
    """LoRA matrices plus, optionally, the full input embeddings and LM head."""
    lora = [p for name, p in model.named_parameters() if "lora_" in name]
    embed: list[torch.nn.Parameter] = []
    if train_embeddings:
        embed.append(model.get_input_embeddings().weight)
        if not embeddings_tied(model):
            head = model.get_output_embeddings()
            embed.extend(p for p in (head.weight, head.bias) if p is not None)
        for param in embed:
            param.data = param.data.float()  # never accumulate small updates in bf16
            param.requires_grad_(True)
    return ParamGroups(lora=lora, embed=embed)


def lora_lr_scale(step: int, embed_warmup_steps: int, lr_warmup_steps: int) -> float:
    """0 during the embedding warm-up, then a linear ramp to 1."""
    if step < embed_warmup_steps:
        return 0.0
    if lr_warmup_steps <= 0:
        return 1.0
    return min(1.0, (step - embed_warmup_steps + 1) / lr_warmup_steps)


def _forward_loss(model: Any, batch: Batch, device: torch.device) -> torch.Tensor:
    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(**inputs).loss


@torch.no_grad()
def heldout_perplexity(
    model: Any, blocks: torch.Tensor, batch_size: int, device: torch.device
) -> float:
    """Token-level perplexity over a fixed set of packed blocks."""
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(blocks), batch_size):
        chunk = blocks[start : start + batch_size]
        predicted = chunk.shape[0] * (chunk.shape[1] - 1)
        loss = _forward_loss(model, {"input_ids": chunk, "labels": chunk}, device)
        total_nll += loss.item() * predicted
        total_tokens += predicted
    model.train(was_training)
    return math.exp(total_nll / total_tokens)


def evaluate_languages(
    model: Any, heldout: dict[str, torch.Tensor], batch_size: int, device: torch.device
) -> dict[str, float]:
    return {
        lang: heldout_perplexity(model, blocks, batch_size, device)
        for lang, blocks in heldout.items()
    }


def _snapshot(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().to("cpu", copy=True) for p in params]


@torch.no_grad()
def _restore(params: list[torch.nn.Parameter], saved: list[torch.Tensor]) -> None:
    for param, value in zip(params, saved, strict=True):
        param.copy_(value.to(param.device))


class Trainer:
    """Runs one stage. `fit` trains until `max_steps` or early stop, then restores the best step."""

    def __init__(
        self,
        model: Any,
        groups: ParamGroups,
        stage: StageConfig,
        *,
        evaluate: Callable[[], dict[str, float]],
        guard: OverfitGuard,
        writer: MetricsWriter,
        embed_warmup_steps: int,
        device: torch.device,
        english_score: Callable[[], float] | None = None,
    ) -> None:
        self.model, self.groups, self.stage = model, groups, stage
        self.evaluate, self.guard, self.writer = evaluate, guard, writer
        self.english_score, self.device = english_score, device
        self.embed_warmup_steps = embed_warmup_steps if groups.embed else 0
        self.optimizer = torch.optim.AdamW(
            [
                {"params": groups.lora, "lr": stage.lr, "weight_decay": stage.weight_decay},
                {"params": groups.embed, "lr": stage.embed_lr, "weight_decay": 0.0},
            ]
        )
        self.spikes = LossSpikeMonitor()
        self.alerts: list[str] = []
        self._recent: list[float] = []
        self._best: list[torch.Tensor] | None = None
        self._best_ppl: dict[str, float] = {}

    def fit(self, next_batch: Callable[[int], Batch]) -> TrainResult:
        self.model.train()
        stopped, step = False, 0
        for step in range(self.stage.max_steps + 1):
            last = step == self.stage.max_steps
            if (step % self.stage.eval_every == 0 or last) and self._eval_step(step):
                stopped = True
                break
            if last:
                break
            self._train_step(step, next_batch(step))
        if self._best is not None:
            _restore(self.groups.all(), self._best)
        return TrainResult(step, self.guard.best_step, stopped, self._best_ppl, self.alerts)

    def _eval_step(self, step: int) -> bool:
        """Score every language, consult the guard, checkpoint on improvement. True means stop."""
        ppl = self.evaluate()
        train_loss = sum(self._recent) / len(self._recent) if self._recent else None
        score = self.english_score() if self.english_score else None
        decision = self.guard.update(
            step, ppl, math.inf if train_loss is None else train_loss, score
        )
        self.alerts.extend(f"step {step}: {alert}" for alert in decision.alerts)
        self.writer.log(
            {
                "step": step,
                "kind": "eval",
                "heldout_ppl": ppl,
                "train_loss": train_loss,
                "english_score": score,
                "improved": decision.improved,
                "alerts": list(decision.alerts),
            }
        )
        log.info("eval", step=step, improved=decision.improved, alerts=decision.alerts, **ppl)
        self._recent.clear()
        if decision.improved:
            self._best, self._best_ppl = _snapshot(self.groups.all()), dict(ppl)
        if decision.stop:
            log.info("early_stop", step=step, best_step=self.guard.best_step)
        return decision.stop

    def _train_step(self, step: int, batch: Batch) -> None:
        scale = lora_lr_scale(step, self.embed_warmup_steps, self.stage.lr_warmup_steps)
        for param in self.groups.lora:
            param.requires_grad_(scale > 0.0)
        self.optimizer.param_groups[0]["lr"] = self.stage.lr * scale
        loss = _forward_loss(self.model, batch, self.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.groups.all(), self.stage.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        value = loss.item()
        self._recent.append(value)
        spiked = self.spikes.update(value)
        if spiked:
            self.alerts.append(f"step {step}: loss_spike:{value:.2f}")
            log.warning(
                "loss_spike",
                step=step,
                loss=value,
                hint="lower the LR, lengthen embed_warmup_steps, re-check embedding init",
            )
        self.writer.log(
            {
                "step": step,
                "kind": "train",
                "loss": value,
                "phase": "embed_warmup" if scale == 0.0 else "full",
                "lora_lr": self.stage.lr * scale,
                "spike": spiked,
            }
        )
