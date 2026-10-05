"""Why 'the worst frozen model gains the most' is not a claim we can make.

`delta = fine_tuned - frozen` carries the frozen score on both axes: directly on
X and subtracted on Y. Noise in the frozen measurement alone produces a negative
correlation between frozen and delta with no real effect behind it - the same
family as regression to the mean.

This script quantifies how often that happens at our own noise level, so the
README's warning cites a measured number instead of a hand-wave.

    python scripts/coupling_check.py
"""

from __future__ import annotations

import argparse
import random


def spearman(xs: list[float], ys: list[float]) -> float:
    rank_x = {v: i for i, v in enumerate(sorted(xs))}
    rank_y = {v: i for i, v in enumerate(sorted(ys))}
    a = [rank_x[v] for v in xs]
    b = [rank_y[v] for v in ys]
    n = len(xs)
    d2 = sum((p - q) ** 2 for p, q in zip(a, b))
    return 1 - 6 * d2 / (n * (n * n - 1))


def spurious_rate(models: int, noise: float, threshold: float,
                  trials: int, seed: int) -> float:
    """Share of trials where coupling alone beats `threshold`.

    Every model is given the SAME true frozen and fine-tuned score, so any
    correlation that shows up is the artefact and nothing else.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        frozen = [0.55 + rng.gauss(0, noise) for _ in range(models)]
        tuned = [0.72 + rng.gauss(0, noise) for _ in range(models)]
        delta = [t - f for f, t in zip(frozen, tuned)]
        if spearman(frozen, delta) < threshold:
            hits += 1
    return hits / trials


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=int, default=8, help="size of the bank")
    p.add_argument("--noise", type=float, default=0.02,
                   help="sd of the measurement noise; 0.02 matches our bootstrap intervals")
    p.add_argument("--threshold", type=float, default=-0.6)
    p.add_argument("--trials", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    rate = spurious_rate(a.models, a.noise, a.threshold, a.trials, a.seed)
    print(f"{a.models} models, noise sd {a.noise}, zero real effect")
    print(f"P(spearman(frozen, delta) < {a.threshold}) = {rate:.1%}")
    print("\nThe valid comparison is fine-tuned against frozen, where the frozen "
          "score appears once.")


if __name__ == "__main__":
    main()
