#!/usr/bin/env python3
"""Day 006: information theory lab. Python standard library only."""

from __future__ import annotations

import math
from typing import Sequence


def validate_distribution(probabilities: Sequence[float], *, tol: float = 1e-9) -> None:
    if not probabilities:
        raise ValueError("distribution must not be empty")
    if any(p < 0.0 or not math.isfinite(p) for p in probabilities):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=tol):
        raise ValueError("probabilities must sum to one")


def entropy(probabilities: Sequence[float], *, base: float = 2.0) -> float:
    """Shannon entropy; base=2 returns bits, base=e returns nats."""
    validate_distribution(probabilities)
    if base <= 0.0 or base == 1.0:
        raise ValueError("logarithm base must be positive and not equal to one")
    return -sum(p * math.log(p, base) for p in probabilities if p > 0.0)


def cross_entropy(target: Sequence[float], predicted: Sequence[float], *,
                  base: float = 2.0) -> float:
    validate_distribution(target)
    validate_distribution(predicted)
    if len(target) != len(predicted):
        raise ValueError("target and predicted distributions must have equal length")
    if any(p > 0.0 and q == 0.0 for p, q in zip(target, predicted)):
        return math.inf
    return -sum(p * math.log(q, base)
                for p, q in zip(target, predicted) if p > 0.0)


def kl_divergence(target: Sequence[float], predicted: Sequence[float], *,
                  base: float = 2.0) -> float:
    validate_distribution(target)
    validate_distribution(predicted)
    if len(target) != len(predicted):
        raise ValueError("target and predicted distributions must have equal length")
    if any(p > 0.0 and q == 0.0 for p, q in zip(target, predicted)):
        return math.inf
    return sum(p * math.log(p / q, base)
               for p, q in zip(target, predicted) if p > 0.0)


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax using max subtraction."""
    if not logits or any(not math.isfinite(z) for z in logits):
        raise ValueError("logits must be a non-empty sequence of finite values")
    maximum = max(logits)
    exponentials = [math.exp(z - maximum) for z in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def cross_entropy_from_logits(logits: Sequence[float], target_index: int) -> float:
    """Stable one-hot cross-entropy in nats, without materializing softmax."""
    if not logits or any(not math.isfinite(z) for z in logits):
        raise ValueError("logits must be a non-empty sequence of finite values")
    if not 0 <= target_index < len(logits):
        raise ValueError("target index is outside the logits range")
    maximum = max(logits)
    log_sum_exp = maximum + math.log(sum(math.exp(z - maximum) for z in logits))
    return log_sum_exp - logits[target_index]


def softmax_cross_entropy_gradient(probabilities: Sequence[float],
                                   target_index: int) -> list[float]:
    """Gradient of one-hot softmax cross-entropy with respect to logits."""
    if not 0 <= target_index < len(probabilities):
        raise ValueError("target index is outside the probability range")
    gradient = list(probabilities)
    gradient[target_index] -= 1.0
    return gradient


def information_gain(parent: Sequence[float],
                     children: Sequence[tuple[float, Sequence[float]]]) -> float:
    """Weighted entropy reduction. Child weights must sum to one."""
    validate_distribution(parent)
    weights = [weight for weight, _ in children]
    if not children or any(w < 0.0 for w in weights):
        raise ValueError("children and non-negative weights are required")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("child weights must sum to one")
    return entropy(parent) - sum(weight * entropy(dist)
                                  for weight, dist in children)


def main() -> None:
    fair = [0.5, 0.5]
    biased = [0.9, 0.1]
    target = [0.8, 0.2]
    predicted = [0.6, 0.4]

    h_fair = entropy(fair)
    h_biased = entropy(biased)
    h_target = entropy(target)
    ce = cross_entropy(target, predicted)
    kl = kl_divergence(target, predicted)

    probabilities = softmax([2.0, 1.0, 0.0])
    logits_loss = cross_entropy_from_logits([2.0, 1.0, 0.0], 0)
    gradient = softmax_cross_entropy_gradient(probabilities, 0)
    gain = information_gain(
        [0.5, 0.5],
        [(0.5, [0.9, 0.1]), (0.5, [0.1, 0.9])],
    )

    assert math.isclose(h_fair, 1.0, abs_tol=1e-12)
    assert math.isclose(h_biased, 0.4689955935892812, abs_tol=1e-12)
    assert math.isclose(ce, 0.8539580943104375, abs_tol=1e-12)
    assert math.isclose(kl, 0.1320299994230752, abs_tol=1e-12)
    assert math.isclose(ce, h_target + kl, abs_tol=1e-12)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    assert math.isclose(probabilities[0], 0.6652409557748218, abs_tol=1e-12)
    assert math.isclose(logits_loss, 0.4076059644443806, abs_tol=1e-12)
    assert math.isclose(sum(gradient), 0.0, abs_tol=1e-12)
    assert math.isclose(gain, 0.5310044064107188, abs_tol=1e-12)

    # A zero predicted probability for a positive target mass has infinite cost.
    assert math.isinf(cross_entropy([1.0,  0.0], [0.0, 1.0]))

    print("Day 006 — Information Theory Lab")
    print(f"Entropy [0.5, 0.5]: {h_fair:.6f} bits")
    print(f"Entropy [0.9, 0.1]: {h_biased:.6f} bits")
    print(f"Cross-entropy P/Q: {ce:.6f} bits")
    print(f"KL divergence P||Q: {kl:.6f} bits")
    print(f"Softmax [2, 1, 0]: {[round(x, 6) for x in probabilities]}")
    print(f"Cross-entropy from logits: {logits_loss:.6f} nats")
    print(f"Softmax-cross-entropy gradient: {[round(x, 6) for x in gradient]}")
    print(f"Decision-tree information gain: {gain:.6f} bits")
    print("All assertions passed.")


if __name__ == "__main__":
    main()
