#!/usr/bin/env python3
"""Day 004 lab: derivatives, gradients, chain rule and gradient descent.

Uses only the Python standard library.
"""

from math import isclose


def quadratic(x: float) -> float:
    return x * x


def quadratic_derivative(x: float) -> float:
    return 2.0 * x


def central_difference(function, x: float, epsilon: float = 1e-5) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (function(x + epsilon) - function(x - epsilon)) / (2.0 * epsilon)


def mse_and_gradients(weights: float, bias: float, xs, ys):
    """Return mean squared error and analytical gradients for y_hat=w*x+b."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs and ys must have the same non-zero length")

    errors = [weights * x + bias - y for x, y in zip(xs, ys)]
    count = len(xs)
    loss = sum(error * error for error in errors) / count
    dw = (2.0 / count) * sum(error * x for error, x in zip(errors, xs))
    db = (2.0 / count) * sum(errors)
    return loss, dw, db


def numerical_parameter_gradient(parameter, objective, epsilon=1e-5):
    """Central-difference gradient for a scalar parameter."""
    return (objective(parameter + epsilon) - objective(parameter - epsilon)) / (
        2.0 * epsilon
    )


def train_linear_regression(xs, ys, learning_rate=0.1, steps=100):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    weights = 0.0
    bias = 0.0
    history = []

    for _ in range(steps):
        loss, dw, db = mse_and_gradients(weights, bias, xs, ys)
        history.append(loss)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    final_loss, _, _ = mse_and_gradients(weights, bias, xs, ys)
    return weights, bias, history, final_loss


def main():
    x = 3.0
    exact = quadratic_derivative(x)
    approximate = central_difference(quadratic, x)
    assert isclose(exact, 6.0, rel_tol=1e-9)
    assert isclose(approximate, exact, rel_tol=1e-7, abs_tol=1e-7)

    # Chain rule: y=(3x+2)^2, dy/dx = 2(3x+2)*3.
    chain_x = 1.0
    u = 3.0 * chain_x + 2.0
    chain_derivative = (2.0 * u) * 3.0
    assert isclose(chain_derivative, 30.0)

    xs = [1.0, 2.0]
    ys = [2.0, 4.0]
    initial_loss, dw, db = mse_and_gradients(0.0, 0.0, xs, ys)
    assert isclose(initial_loss, 10.0)
    assert isclose(dw, -10.0)
    assert isclose(db, -6.0)

    # Check both analytical parameter gradients numerically.
    objective_w = lambda w: mse_and_gradients(w, 0.0, xs, ys)[0]
    objective_b = lambda b: mse_and_gradients(0.0, b, xs, ys)[0]
    numeric_dw = numerical_parameter_gradient(0.0, objective_w)
    numeric_db = numerical_parameter_gradient(0.0, objective_b)
    assert isclose(numeric_dw, dw, rel_tol=1e-7, abs_tol=1e-7)
    assert isclose(numeric_db, db, rel_tol=1e-7, abs_tol=1e-7)

    one_step_w = 0.0 - 0.1 * dw
    one_step_b = 0.0 - 0.1 * db
    one_step_loss, _, _ = mse_and_gradients(one_step_w, one_step_b, xs, ys)
    assert isclose(one_step_w, 1.0)
    assert isclose(one_step_b, 0.6)
    assert isclose(one_step_loss, 1.06)

    weights, bias, history, final_loss = train_linear_regression(
        xs, ys, learning_rate=0.1, steps=100
    )
    assert history[-1] < history[0]
    assert final_loss < history[0]
    assert isclose(weights, 2.0, abs_tol=1e-3)
    assert isclose(bias, 0.0, abs_tol=1e-3)

    print(f"quadratic derivative at x=3: {exact:.1f}")
    print(f"finite difference check: PASS (approx={approximate:.6f})")
    print(f"chain rule derivative at x=1: {chain_derivative:.1f}")
    print(f"initial loss: {initial_loss:.1f}")
    print(f"gradient dw: {dw:.1f}")
    print(f"gradient db: {db:.1f}")
    print(f"parameters after one step: w={one_step_w:.1f}, b={one_step_b:.1f}")
    print(f"loss after one step: {one_step_loss:.2f}")
    print(f"trained parameters after 100 steps: w={weights:.6f}, b={bias:.6f}")
    print(f"final loss: {final_loss:.8f}")
    print("training loss decreased: PASS")
    print("all checks passed")


if __name__ == "__main__":
    main()
