"""Language-mixing curriculum: temperature sampling over target languages plus English replay.

Target language i gets weight `n_i ** T` (n_i = corpus size in tokens, T = 0.3 by default), which
upweights low-resource languages relative to proportional sampling. A fixed `english_replay`
share of every batch is English text to limit catastrophic forgetting.
"""

from __future__ import annotations

import random

from polyglotlm.config import ENGLISH, TrainingRecipe


def temperature_weights(sizes: dict[str, int], temperature: float) -> dict[str, float]:
    """Normalised `n_i ** temperature`. T=1 is proportional, T->0 is uniform."""
    if not sizes or any(n <= 0 for n in sizes.values()):
        raise ValueError(f"every language needs a positive corpus size, got {sizes}")
    raw = {lang: n**temperature for lang, n in sizes.items()}
    total = sum(raw.values())
    return {lang: weight / total for lang, weight in raw.items()}


def mixture(recipe: TrainingRecipe, sizes: dict[str, int]) -> dict[str, float]:
    """Sampling probability per language for one batch element. Sums to 1.

    `sizes` holds the token count of every language that has training data. Explicit
    `cpt_mix_ratio` values win over temperature sampling. English replay is dropped (and the
    targets renormalised) only when there is no English data to replay.
    """
    targets = {lang: n for lang, n in sizes.items() if lang != ENGLISH}
    replay = recipe.english_replay if ENGLISH in sizes else 0.0
    explicit = {
        lang: recipe.language(lang).cpt_mix_ratio
        for lang in targets
        if recipe.language(lang).cpt_mix_ratio is not None
    }
    if explicit:
        total = sum(explicit.values())
        weights = {lang: ratio / total for lang, ratio in explicit.items()}
    else:
        weights = temperature_weights(targets, recipe.mix_temperature)
    probs = {lang: weight * (1.0 - replay) for lang, weight in weights.items()}
    if replay:
        probs[ENGLISH] = replay
    return probs


class MixtureSampler:
    """Seeded sampler of a language per batch element."""

    def __init__(self, probs: dict[str, float], seed: int) -> None:
        if abs(sum(probs.values()) - 1.0) > 1e-6:
            raise ValueError(f"mixture probabilities must sum to 1, got {probs}")
        self.langs = list(probs)
        self.weights = [probs[lang] for lang in self.langs]
        self._rng = random.Random(seed)

    def sample(self, n: int) -> list[str]:
        return self._rng.choices(self.langs, weights=self.weights, k=n)
