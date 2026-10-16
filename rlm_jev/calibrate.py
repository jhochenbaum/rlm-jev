"""Calibration helpers.

fit_weights: per-label multiplicative weights fit on a small labeled sample (same idea as DSPy's ReAnchor
Choice weights). Fit them on the same request format and data distribution you deploy on; they do not transfer.
mc_confidence: confidence for any answer computed from per-item probabilities, by recomputing the answer from
labels sampled from those probabilities. Lets an RLM abstain when its aggregate answer is fragile.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

GRID = (0.03, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)


def apply_weights(probs: list[dict[str, float]], weights: dict[str, float]) -> list[str]:
    """Pick each item's label after multiplying its probabilities by per-label weights (missing labels: 1.0)."""
    return [max(p, key=lambda k: p[k] * weights.get(k, 1.0)) for p in probs]


def fit_weights(probs: list[dict[str, float]], gold: list[str], grid=GRID, rounds: int = 3) -> dict[str, float]:
    """Coordinate ascent on accuracy; a change is kept only if it strictly improves the fit set."""
    labels = list(probs[0])
    acc = lambda w: sum(a == g for a, g in zip(apply_weights(probs, w), gold, strict=False)) / len(gold)
    w = dict.fromkeys(labels, 1.0)
    for _ in range(rounds):
        for k in labels:
            best = max(grid, key=lambda g: (acc({**w, k: g}), -abs(math.log(g))))
            if acc({**w, k: best}) > acc(w):
                w[k] = best
    return w


def mc_confidence(
    compute: Callable[[list[str]], Any],
    probs: list[dict[str, float]],
    n: int = 60,
    seed: int = 0,
    weights: dict[str, float] | None = None,
) -> tuple[Any, float]:
    """compute(labels) -> answer. Returns (answer from the most likely labels, share of answers recomputed from
    labels sampled from the probabilities that agree with it). With `weights`, both use the reweighted probabilities."""
    rng = random.Random(seed)
    w = weights or {}
    point = compute(apply_weights(probs, w))
    keys = [list(p) for p in probs]
    agree = 0
    for _ in range(n):
        labels = [
            rng.choices(k, weights=[p[x] * w.get(x, 1.0) for x in k])[0] for k, p in zip(keys, probs, strict=False)
        ]
        agree += compute(labels) == point
    return point, agree / n


def ece(confidence, correct, bins: int = 10) -> float:
    """Expected calibration error: the gap between confidence and accuracy, averaged over equal-width confidence bins
    and weighted by how many predictions fall in each."""
    pairs = [(min(max(float(c), 0.0), 1.0), ok) for c, ok in zip(confidence, correct, strict=True)]
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            total += (
                len(sel) / len(pairs) * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
            )
    return total
