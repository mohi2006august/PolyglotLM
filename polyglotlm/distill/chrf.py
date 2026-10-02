"""chrF (Popović, 2015): character n-gram F-score, robust for morphologically rich languages.

Scores are on a 0-100 scale like sacreBLEU. Whitespace is ignored; n-gram orders 1..6 are
averaged; beta=2 weights recall twice as much as precision.
"""

from __future__ import annotations

from collections import Counter


def _ngrams(text: str, n: int) -> Counter[str]:
    chars = "".join(text.split())
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def chrf(hypothesis: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    precisions, recalls = [], []
    for n in range(1, max_order + 1):
        hyp, ref = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not hyp or not ref:
            continue
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / sum(hyp.values()))
        recalls.append(overlap / sum(ref.values()))
    if not precisions:
        return 100.0 if hypothesis.strip() == reference.strip() else 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * precision * recall / (b2 * precision + recall)
