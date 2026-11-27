#!/usr/bin/env python3
"""Day 005 lab: probability, Bayes' theorem and Gaussian maximum likelihood.

Uses only the Python standard library. Run with: python lab.py
"""

from math import exp, log, pi, sqrt
import random


def validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def expectation(probabilities: dict[float, float]) -> float:
    if not probabilities:
        raise ValueError("distribution cannot be empty")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to 1")
    for p in probabilities.values():
        validate_probability(p, "probability")
    return sum(x * p for x, p in probabilities.items())


def population_variance(probabilities: dict[float, float]) -> float:
    mean = expectation(probabilities)
    return sum((x - mean) ** 2 * p for x, p in probabilities.items())


def bayes_fault_after_alarm(
    prevalence: float, sensitivity: float, specificity: float
) -> float:
    """P(fault | alarm), using prevalence, TPR and TNR."""
    for value, name in (
        (prevalence, "prevalence"),
        (sensitivity, "sensitivity"),
        (specificity, "specificity"),
    ):
        validate_probability(value, name)

    false_positive_rate = 1.0 - specificity
    alarm_probability = (
        sensitivity * prevalence
        + false_positive_rate * (1.0 - prevalence)
    )
    if alarm_probability == 0.0:
        raise ValueError("an alarm has zero probability under these assumptions")
    return sensitivity * prevalence / alarm_probability


def simulate_fault_after_alarm(
    prevalence: float,
    sensitivity: float,
    specificity: float,
    trials: int,
    seed: int = 2026,
) -> tuple[int, int]:
    """Return (fault_and_alarm_count, alarm_count) in a reproducible simulation."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    for value, name in (
        (prevalence, "prevalence"),
        (sensitivity, "sensitivity"),
        (specificity, "specificity"),
    ):
        validate_probability(value, name)

    rng = random.Random(seed)
    true_positive = 0
    alarms = 0
    for _ in range(trials):
        fault = rng.random() < prevalence
        alarm = (
            rng.random() < sensitivity
            if fault
            else rng.random() < (1.0 - specificity)
        )
        if alarm:
            alarms += 1
            if fault:
                true_positive += 1
    return true_positive, alarms


def gaussian_mle(samples: list[float]) -> tuple[float, float]:
    """Return MLE mean and population-form variance (denominator n)."""
    if not samples:
        raise ValueError("samples cannot be empty")
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)
    if variance <= 0.0:
        raise ValueError("Gaussian MLE requires non-zero variance here")
    return mean, variance


def gaussian_log_likelihood(
    samples: list[float], mean: float, variance: float
) -> float:
    if not samples:
        raise ValueError("samples cannot be empty")
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    n = len(samples)
    squared_error = sum((x - mean) ** 2 for x in samples)
    return -0.5 * n * log(2.0 * pi * variance) - squared_error / (2.0 * variance)


def main() -> None:
    retries = {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}
    mean_retries = expectation(retries)
    variance_retries = population_variance(retries)
    print("Discrete distribution")
    print(f"  E[X] = {mean_retries:.4f}")
    print(f"  Var(X) = {variance_retries:.4f}")
    print(f"  Std dev = {sqrt(variance_retries):.4f}")

    prevalence, sensitivity, specificity = 0.01, 0.95, 0.90
    posterior = bayes_fault_after_alarm(prevalence, sensitivity, specificity)
    print("\nBayes diagnostic example")
    print(f"  P(fault | alarm) = {posterior:.6f} ({posterior:.2%})")

    tp, alarm_count = simulate_fault_after_alarm(
        prevalence, sensitivity, specificity, trials=200_000, seed=2026
    )
    simulated = tp / alarm_count
    print("\nSeeded Monte Carlo check")
    print(f"  true-positive alarms = {tp}")
    print(f"  all alarms = {alarm_count}")
    print(f"  simulated P(fault | alarm) = {simulated:.6f}")
    print(f"  analytical P(fault | alarm) = {posterior:.6f}")

    samples = [9.0, 10.0, 11.0, 12.0, 13.0]
    mle_mean, mle_variance = gaussian_mle(samples)
    log_likelihood = gaussian_log_likelihood(samples, mle_mean, mle_variance)
    print("\nGaussian maximum likelihood")
    print(f"  MLE mean = {mle_mean:.4f}")
    print(f"  MLE variance = {mle_variance:.4f}")
    print(f"  MLE std dev = {sqrt(mle_variance):.4f}")
    print(f"  log likelihood = {log_likelihood:.6f}")

    # Deterministic checks for the expected lesson results.
    assert abs(mean_retries - 1.1) < 1e-12
    assert abs(variance_retries - 0.49) < 1e-12
    assert abs(posterior - (0.0095 / 0.1085)) < 1e-12
    assert abs(mle_mean - 11.0) < 1e-12
    assert abs(mle_variance - 2.0) < 1e-12
    assert abs(simulated - posterior) < 0.02
    assert log_likelihood < 0.0

    # Guard against invalid inputs.
    try:
        bayes_fault_after_alarm(-0.1, sensitivity, specificity)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid prevalence should be rejected")

    print("\nPASS: all deterministic assertions and input checks passed.")


if __name__ == "__main__":
    main()
