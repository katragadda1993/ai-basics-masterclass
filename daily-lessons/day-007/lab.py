#!/usr/bin/env python3
"""Day 007: compare common optimizers on a deterministic quadratic.

Python standard library only.
Objective: f(x, y) = 0.5 * (x*x + 25*y*y)
Minimum: (0, 0); gradient: (x, 25*y).
"""
from __future__ import annotations
import math
from typing import Callable

Vector = list[float]

def objective(point: Vector) -> float:
    x, y = point
    return 0.5 * (x*x + 25.0*y*y)

def gradient(point: Vector) -> Vector:
    x, y = point
    return [x, 25.0*y]

def cosine_learning_rate(base_lr: float, step: int, total_steps: int,
                         min_lr: float = 0.0) -> float:
    if base_lr < 0 or min_lr < 0 or min_lr > base_lr:
        raise ValueError("require 0 <= min_lr <= base_lr")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(step / total_steps, 0.0), 1.0)
    return min_lr + 0.5*(base_lr-min_lr)*(1.0+math.cos(math.pi*progress))

def clip_by_global_norm(values: Vector, max_norm: float) -> tuple[Vector, float]:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("gradient values must be finite")
    norm = math.sqrt(sum(v*v for v in values))
    if norm == 0 or norm <= max_norm:
        return list(values), norm
    scale = max_norm/norm
    return [v*scale for v in values], norm

def finite_difference(fn: Callable[[Vector], float], point: Vector,
                      index: int, h: float = 1e-6) -> float:
    if h <= 0 or not 0 <= index < len(point):
        raise ValueError("invalid finite-difference arguments")
    plus, minus = list(point), list(point)
    plus[index] += h
    minus[index] -= h
    return (fn(plus)-fn(minus))/(2*h)

def optimize(name: str, learning_rate: float, steps: int = 300,
             beta1: float = 0.9, beta2: float = 0.999,
             epsilon: float = 1e-8, weight_decay: float = 0.01
             ) -> tuple[float, float, Vector]:
    if learning_rate <= 0 or steps <= 0:
        raise ValueError("learning_rate and steps must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("betas must be in [0, 1)")
    if epsilon <= 0 or weight_decay < 0:
        raise ValueError("epsilon must be positive and weight_decay non-negative")

    point = [10.0, 10.0]
    initial_loss = objective(point)
    first = [0.0, 0.0]
    second = [0.0, 0.0]

    for step in range(1, steps+1):
        grad = gradient(point)
        if name == "sgd":
            point = [p-learning_rate*g for p, g in zip(point, grad)]
        elif name == "momentum":
            first = [beta1*m+g for m, g in zip(first, grad)]
            point = [p-learning_rate*m for p, m in zip(point, first)]
        elif name == "rmsprop":
            second = [beta2*v+(1-beta2)*g*g for v, g in zip(second, grad)]
            point = [p-learning_rate*g/(math.sqrt(v)+epsilon)
                     for p, g, v in zip(point, grad, second)]
        elif name in ("adam", "adamw"):
            first = [beta1*m+(1-beta1)*g for m, g in zip(first, grad)]
            second = [beta2*v+(1-beta2)*g*g for v, g in zip(second, grad)]
            first_hat = [m/(1-beta1**step) for m in first]
            second_hat = [v/(1-beta2**step) for v in second]
            point = [p-learning_rate*m/(math.sqrt(v)+epsilon)
                     for p, m, v in zip(point, first_hat, second_hat)]
            if name == "adamw":
                point = [p*(1-learning_rate*weight_decay) for p in point]
        else:
            raise ValueError(f"unknown optimizer: {name}")
        if any(not math.isfinite(v) for v in point):
            raise FloatingPointError(f"{name} produced non-finite parameters")

    return initial_loss, objective(point), point

def main() -> None:
    start = [10.0, 10.0]
    analytic = gradient(start)
    numeric = [finite_difference(objective, start, i) for i in range(2)]
    assert all(math.isclose(a, n, rel_tol=1e-6, abs_tol=1e-6)
               for a, n in zip(analytic, numeric))

    settings = {
        "SGD": ("sgd", 0.04),
        "Momentum": ("momentum", 0.005),
        "RMSProp": ("rmsprop", 0.1),
        "Adam": ("adam", 0.1),
        "AdamW": ("adamw", 0.1),
    }
    print("Day 007 — Optimization Algorithms Lab")
    print(f"Initial objective: {objective(start):.6f}")
    print(f"Analytical gradient: {analytic}")
    print(f"Finite-difference gradient: {[round(v, 6) for v in numeric]}")
    print("\nOptimizer comparison (illustrative hyperparameters):")
    for label, (name, lr) in settings.items():
        initial, final, point = optimize(name, lr)
        assert math.isfinite(final) and final < initial
        print(f"{label:10s} lr={lr:<6g} initial={initial:.6g} "
              f"final={final:.6g} parameters={[round(v, 8) for v in point]}")

    clipped, original_norm = clip_by_global_norm([3.0, 4.0], 2.0)
    assert math.isclose(original_norm, 5.0, abs_tol=1e-12)
    assert math.isclose(math.sqrt(sum(v*v for v in clipped)), 2.0, abs_tol=1e-12)

    schedule = [cosine_learning_rate(0.1, s, 100) for s in (0, 25, 50, 75, 100)]
    assert math.isclose(schedule[0], 0.1, abs_tol=1e-12)
    assert math.isclose(schedule[-1], 0.0, abs_tol=1e-12)
    assert all(a >= b for a, b in zip(schedule, schedule[1:]))

    print(f"Global norm clipping [3, 4] to 2: {clipped}")
    print(f"Cosine LR at 0/25/50/75/100: {[round(v, 6) for v in schedule]}")
    print("All assertions passed.")

if __name__ == "__main__":
    main()
