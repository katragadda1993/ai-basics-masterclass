"""Session 015 lab: one-step telemetry forecasting and CUSUM change detection.

Run:
    python lab.py

The dataset is synthetic. It demonstrates temporal validation, lag features,
forecast baselines, and residual-based change detection.
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 15015
SAMPLES_PER_DAY = 96  # 15-minute telemetry windows
DAYS = 14
LAGS = (1, 2, 3, 4, SAMPLES_PER_DAY)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def generate_series():
    """Return a 14-day series with a held-out positive level shift."""
    rng = np.random.default_rng(SEED)
    n = DAYS * SAMPLES_PER_DAY
    t = np.arange(n)

    daily = 1.7 * np.sin(2 * np.pi * t / SAMPLES_PER_DAY)
    daily += 0.45 * np.cos(2 * np.pi * t / SAMPLES_PER_DAY)
    weekly = 0.35 * np.sin(2 * np.pi * t / (SAMPLES_PER_DAY * 7))
    trend = 0.00035 * t

    innovations = rng.normal(0.0, 0.32, n)
    correlated_noise = np.zeros(n)
    for i in range(1, n):
        correlated_noise[i] = 0.62 * correlated_noise[i - 1] + innovations[i]

    y = 12.0 + daily + weekly + trend + correlated_noise

    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)
    change_index = validation_end + 45
    y[change_index:] += 1.55
    return t, y, train_end, validation_end, change_index


def make_features(y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Features at t use only observations strictly before t."""
    rows = []
    for i in indices:
        row = [y[i - lag] for lag in LAGS]
        row.extend((np.mean(y[i - 4:i]), np.mean(y[i - SAMPLES_PER_DAY:i])))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return float(mae), float(rmse)


def positive_cusum(
    residuals: np.ndarray,
    reference_mean: float,
    reference_std: float,
    allowance_k: float = 0.5,
    threshold_h: float = 8.0,
) -> tuple[int | None, np.ndarray]:
    """One-sided standardized CUSUM for an upward residual shift."""
    if reference_std <= 0:
        raise ValueError("reference_std must be positive")
    z = (residuals - reference_mean) / reference_std
    running = 0.0
    trace = []
    first_alarm = None
    for i, value in enumerate(z):
        running = max(0.0, running + value - allowance_k)
        trace.append(running)
        if first_alarm is None and running > threshold_h:
            first_alarm = i
    return first_alarm, np.asarray(trace)


def main() -> None:
    t, y, train_end, validation_end, change_index = generate_series()
    n = len(y)
    first_index = max(LAGS)

    # Chronological target indices. No random shuffle is used.
    indices = np.arange(first_index, n)
    train_idx = indices[indices < train_end]
    val_idx = indices[(indices >= train_end) & (indices < validation_end)]
    test_idx = indices[indices >= validation_end]

    X = make_features(y, indices)
    target = y[indices]

    train_mask = indices < train_end
    val_mask = (indices >= train_end) & (indices < validation_end)
    test_mask = indices >= validation_end

    X_train, y_train = X[train_mask], target[train_mask]
    X_val, y_val = X[val_mask], target[val_mask]
    X_test, y_test = X[test_mask], target[test_mask]

    assert train_idx.max() < val_idx.min() < test_idx.min()
    assert change_index > validation_end
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(y_train))

    # Tune regularization using validation only. StandardScaler is fitted
    # inside each pipeline on training data only.
    candidates = []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        _, val_rmse = regression_metrics(y_val, val_pred)
        candidates.append((val_rmse, alpha, model))

    best_val_rmse, best_alpha, best_model = min(candidates, key=lambda item: item[0])
    ridge_test = best_model.predict(X_test)

    # Two simple operational baselines.
    persistence_test = np.asarray([y[i - 1] for i in test_idx])
    seasonal_test = np.asarray([y[i - SAMPLES_PER_DAY] for i in test_idx])

    forecasts = {
        "Persistence (lag 1)": persistence_test,
        "Seasonal naive (lag 96)": seasonal_test,
        f"Ridge lag model (alpha={best_alpha:g})": ridge_test,
    }

    print("Session 015 — Time-series forecasting and change-point lab")
    print(f"NumPy {np.__version__} | scikit-learn {sklearn.__version__}")
    print(f"Samples: {n} ({DAYS} days at 15-minute intervals)")
    print(
        f"Chronological split: train={len(X_train)}, "
        f"validation={len(X_val)}, test={len(X_test)}"
    )
    print(f"Injected held-out change index: {change_index}")
    print(f"Validation-selected Ridge alpha: {best_alpha:g}")
    print(f"Best validation RMSE: {best_val_rmse:.6f} ms")
    print("\nHeld-out one-step forecast metrics (ms):")
    for name, prediction in forecasts.items():
        mae, rmse = regression_metrics(y_test, prediction)
        print(f"  {name:34s} MAE={mae:.6f} RMSE={rmse:.6f}")

    # Compare the stable pre-change portion with the shifted regime.
    pre_change = test_idx < change_index
    post_change = ~pre_change
    print("\nRidge regime-specific test metrics (ms):")
    for label, mask in (("before change", pre_change), ("after change", post_change)):
        mae, rmse = regression_metrics(y_test[mask], ridge_test[mask])
        print(f"  {label:16s} MAE={mae:.6f} RMSE={rmse:.6f}")

    # Estimate the residual reference distribution on validation only.
    # Seasonal-naive residuals retain a sustained offset after a level shift,
    # making them suitable for this simple change-point demonstration.
    val_seasonal = np.asarray([y[i - SAMPLES_PER_DAY] for i in val_idx])
    val_residuals = y_val - val_seasonal
    test_residuals = y_test - seasonal_test
    reference_mean = float(np.mean(val_residuals))
    reference_std = float(np.std(val_residuals, ddof=1))

    alarm_offset, cusum_trace = positive_cusum(
        test_residuals,
        reference_mean,
        reference_std,
        allowance_k=0.5,
        threshold_h=8.0,
    )
    print("\nCUSUM change-point experiment:")
    print(f"  Validation residual mean: {reference_mean:.6f} ms")
    print(f"  Validation residual std:  {reference_std:.6f} ms")
    print("  Allowance k=0.5 sigma | threshold h=8.0 sigma")
    if alarm_offset is None:
        print("  No alarm in the held-out period")
    else:
        alarm_index = int(test_idx[alarm_offset])
        delay_samples = alarm_index - change_index
        delay_minutes = delay_samples * 15
        print(f"  First alarm index: {alarm_index}")
        print(f"  Injected change index: {change_index}")
        print(f"  Detection delay: {delay_samples} samples ({delay_minutes} minutes)")

    # Autocorrelation at lag 1, calculated on the training interval only.
    train_series = y[:train_end]
    acf_lag1 = float(np.corrcoef(train_series[1:], train_series[:-1])[0, 1])
    print(f"\nTraining-series lag-1 autocorrelation: {acf_lag1:.6f}")

    # Acceptance checks: temporal ordering, finite values, a valid model,
    # and detection after (not before) the deliberately injected change.
    assert len(train_idx) > len(val_idx) > 0
    assert len(test_idx) > 0
    assert np.isfinite(best_val_rmse)
    assert np.all(np.isfinite(ridge_test))
    assert alarm_offset is not None
    alarm_index = int(test_idx[alarm_offset])
    assert alarm_index >= change_index
    assert alarm_index - change_index <= 12  # <= 3 hours in this seeded demo
    assert len(cusum_trace) == len(test_idx)

    print("\nPASS: all temporal-split, finite-value and change-detection assertions passed.")


if __name__ == "__main__":
    main()
