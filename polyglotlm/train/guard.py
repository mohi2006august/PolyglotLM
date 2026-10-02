"""FR-06 overfitting guard.

* Early stopping is driven by held-out *native* text perplexity per target language, never by
  translated or synthetic data (translated validation sets hide overfitting).
* Per-language overfitting: held-out perplexity rising while the training loss keeps falling.
* English regression: an alert when the English score drops more than the recipe allows.
* Loss spikes: the usual symptom of badly initialised new embeddings after a vocab resize.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


@dataclass(frozen=True)
class GuardDecision:
    improved: bool  # best target-language held-out perplexity so far: checkpoint this step
    stop: bool  # patience exhausted
    alerts: tuple[str, ...]


@dataclass
class OverfitGuard:
    """Feed one `update` per evaluation; it says whether to checkpoint, stop, or raise alerts."""

    target_langs: list[str]
    patience: int
    max_english_regression: float = 3.0  # points
    min_delta: float = 1e-4  # relative improvement needed to reset patience
    best_step: int | None = None
    best_score: float = math.inf
    english_baseline: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    _train_losses: list[float] = field(default_factory=list)
    _bad_evals: int = 0

    def update(
        self,
        step: int,
        heldout_ppl: dict[str, float],
        train_loss: float,
        english_score: float | None = None,
    ) -> GuardDecision:
        missing = [lang for lang in self.target_langs if lang not in heldout_ppl]
        if missing:
            raise ValueError(f"held-out perplexity missing for target languages: {missing}")
        alerts = [
            *self._overfit_alerts(heldout_ppl, train_loss),
            *self._english_alerts(english_score),
        ]
        score = geometric_mean([heldout_ppl[lang] for lang in self.target_langs])
        improved = score < self.best_score * (1.0 - self.min_delta)
        if improved:
            self.best_score, self.best_step, self._bad_evals = score, step, 0
        else:
            self._bad_evals += 1
        self.history.append(dict(heldout_ppl))
        self._train_losses.append(train_loss)
        return GuardDecision(improved, self._bad_evals >= self.patience, tuple(alerts))

    def _overfit_alerts(self, heldout_ppl: dict[str, float], train_loss: float) -> list[str]:
        """Languages whose held-out perplexity rose while the training loss fell."""
        if not self.history or train_loss >= self._train_losses[-1]:
            return []
        previous = self.history[-1]
        return [
            f"overfit:{lang}"
            for lang in self.target_langs
            if heldout_ppl[lang] > previous[lang] * (1.0 + self.min_delta)
        ]

    def _english_alerts(self, english_score: float | None) -> list[str]:
        """Compare the English benchmark score (points, higher is better) with the first one."""
        if english_score is None:
            return []
        if self.english_baseline is None:
            self.english_baseline = english_score
            return []
        drop = self.english_baseline - english_score
        return [f"english_regression:{drop:.1f}pts"] if drop > self.max_english_regression else []


@dataclass
class LossSpikeMonitor:
    """Flags a training loss far above its running average (e.g. right after a vocab resize)."""

    factor: float = 2.0
    beta: float = 0.9
    warmup: int = 5
    _ema: float | None = None
    _seen: int = 0

    def update(self, loss: float) -> bool:
        spiked = (
            self._ema is not None and self._seen >= self.warmup and loss > self.factor * self._ema
        )
        self._ema = loss if self._ema is None else self.beta * self._ema + (1 - self.beta) * loss
        self._seen += 1
        return spiked
