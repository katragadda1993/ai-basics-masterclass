# Session 015 — Time-Series Machine Learning: Forecasting, Windowing, Temporal Validation and Change-Point Detection

**Phase:** 3 — Machine Learning  
**Duration:** 60 minutes  
**Prerequisites:** Sessions 005–014: probability, optimization, supervised learning, clustering, PCA and anomaly detection.

## Why this lesson follows anomaly detection

Session 014 treated telemetry observations as individual feature vectors. Real silicon and system behavior is temporal: queue pressure rises, clocks change, DMA retries accumulate, temperature drifts, and errors may appear only after a workload transition.

This session adds the time axis. We will forecast a signal, evaluate it without looking into the future, and monitor forecast residuals for a sustained process change.

The teaching example uses a synthetic DMA-latency series. It is not measured hardware data and must not be used as a production threshold.

## Learning objectives

By the end of this session, you will be able to:

1. Explain trend, seasonality, noise, stationarity and autocorrelation.
2. Build lag, rolling-window and calendar features without leaking future values.
3. Distinguish one-step, multi-step, recursive and direct forecasting.
4. Use chronological and rolling-origin validation instead of shuffled splits.
5. Compare persistence, seasonal-naive and regression-based forecasts.
6. Explain moving averages, exponential smoothing and ARIMA at a practical level.
7. Use forecast residuals with a CUSUM change detector.
8. Design an event-driven, observable time-series monitoring workflow with human review.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | From point anomalies to temporal behavior |
| 5–15 min | Components, stationarity and autocorrelation |
| 15–25 min | Windowing, lag features and forecasting horizons |
| 25–35 min | Temporal validation, baselines and ARIMA |
| 35–43 min | Residual monitoring and CUSUM change detection |
| 43–55 min | Python forecasting and change-point lab |
| 55–60 min | Production agent architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Time series | Ordered observations indexed by time |
| Cadence / sampling interval | Expected time between consecutive observations |
| Trend | Long-term direction of a signal |
| Seasonality | Repeating pattern with a known period |
| Stationarity | Stable statistical behavior over time, under a stated definition |
| Autocorrelation | Correlation between a signal and a lagged version of itself |
| Lag feature | A past observation used as a model input |
| Rolling window | A fixed or expanding recent interval used to compute statistics |
| Forecast origin | The latest timestamp whose information is allowed in a forecast |
| Forecast horizon | How far into the future the target lies |
| Exogenous feature | An external variable used to explain or predict the target |
| Residual | Actual observation minus forecast |
| Change point | A time at which the signal-generating process changes |
| CUSUM | Cumulative-sum detector for sustained shifts |
| Watermark | Event-time progress marker used to decide when a stream window is sufficiently complete |
| Rolling-origin backtest | Repeated evaluation at successive forecast origins |

## 1. Why time changes the ML problem

A normal tabular dataset often treats each row as independent. Time-series observations are ordered and frequently dependent on previous observations.

Examples from SoC and audio systems:

- DMA latency depends on recent queue depth and bus contention.
- PCIe replay activity may rise after a link or workload transition.
- SoundWire error counters can accumulate over a stream interval.
- Codec latency may change after a sample-rate or clock configuration change.
- Temperature and power consumption drift more slowly than individual transactions.

A time-series sample therefore needs at least a value and a timestamp. In production, it also needs device identity, units, sampling interval, operating mode, firmware version, workload context and data-quality metadata.

### Four common components

A useful conceptual decomposition is:

\[
y_t = T_t + S_t + C_t + \epsilon_t
\]

where:

- **Trend** \(T_t\): long-term movement.
- **Seasonality** \(S_t\): repeatable pattern with a known period.
- **Cycle** \(C_t\): longer or less regular fluctuations.
- **Noise** \(\epsilon_t\): unexplained variation.

This additive form is a model, not a universal law. Some signals are multiplicative or have changing variance.

For example, a DMA latency signal might have a slow thermal trend, a periodic workload cycle, and short-term queueing noise.

## 2. Stationarity and autocorrelation

### Stationarity

A weakly stationary process has a stable mean and variance, and its autocovariance depends on the lag rather than the absolute time.

Many practical telemetry signals are not stationary because of workload changes, firmware or driver updates, temperature drift, new operating modes, sensor calibration changes, aging or hardware degradation.

Differencing can remove some trend:

\[
\Delta y_t = y_t-y_{t-1}
\]

Seasonal differencing removes a repeating period:

\[
\Delta_s y_t = y_t-y_{t-s}
\]

Differencing can help an ARIMA model, but it can also remove useful information or amplify noise. Always inspect the transformed series and validate forecasts in the original engineering units.

### Autocorrelation

Autocorrelation measures how a signal relates to a lagged copy of itself:

\[
\rho_k = Corr(y_t,y_{t-k})
\]

- Lag 1 asks whether the current sample resembles the previous sample.
- Lag 4 at 15-minute cadence represents one hour.
- Lag 96 at 15-minute cadence represents one day.

A high lag-1 autocorrelation means nearby samples are strongly related. It does not mean the signal is predictable indefinitely.

The autocorrelation function (ACF) is useful for exploring persistence and periodicity. The partial autocorrelation function (PACF) can help identify direct autoregressive relationships after accounting for shorter lags.

## 3. Windowing and feature construction

A sliding window converts a sequence into supervised examples.

For a one-step forecast:

\[
X_t=[y_{t-1},y_{t-2},\ldots,y_{t-L}],\qquad target=y_t
\]

For a horizon of \(h\):

\[
target=y_{t+h}
\]

Common feature families:

| Feature | Example | Purpose |
|---|---|---|
| Lag | \(y_{t-1}, y_{t-4}, y_{t-96}\) | Recent and seasonal history |
| Rolling mean | mean of previous 4 samples | Local level |
| Rolling standard deviation | std of previous 96 samples | Local variability |
| Rolling min/max | previous-window extremes | Range and spikes |
| Calendar | hour, day, scheduled workload | Known periodic effects |
| Exogenous | queue depth, clock state, workload ID | External drivers |

**Leakage rule:** a feature used to predict \(y_t\) must be available at the prediction time. A rolling statistic must be shifted so it excludes the target and all future values.

Overlapping windows are not independent. If a random split puts near-duplicate windows in train and test, evaluation can look much better than real deployment. Preserve chronological order and consider a gap when the forecast horizon or target window overlaps the split boundary.

### Forecasting horizons

- **One-step ahead:** predict the next sample using the latest observed history.
- **Multi-step direct:** train a separate output or model for each horizon.
- **Multi-step recursive:** feed each prediction back as an input for the next step.
- **Rolling-origin:** move the forecast origin forward and evaluate repeatedly.

One-step rolling evaluation may use each newly observed value after it arrives. That is valid for an online one-step service, but it is not the same as forecasting an entire future day without observing intermediate values. State which protocol you evaluate.

## 4. Baselines and common forecasting models

A complex model must beat a simple baseline on the same chronological evaluation.

### Persistence / naive forecast

\[
\hat y_{t+1}=y_t
\]

This is difficult to beat for highly autocorrelated signals. It is a required baseline, not an embarrassing shortcut.

### Seasonal naive forecast

\[
\hat y_t=y_{t-s}
\]

At 15-minute cadence with daily repetition, \(s=96\). This is useful when a daily pattern is stable.

### Moving average

\[
\hat y_{t+1}=\frac{1}{m}\sum_{i=0}^{m-1}y_{t-i}
\]

A moving average smooths short-term noise but introduces lag when the signal changes quickly.

### Exponential smoothing

\[
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}
\]

Recent observations receive more weight. A larger \(\alpha\) adapts faster but may react more to noise. Holt's method adds trend; Holt–Winters adds seasonality.

### ARIMA

ARIMA models temporal dependence through autoregressive, differencing and moving-average-error terms:

\[
ARIMA(p,d,q)
\]

- \(p\): autoregressive order.
- \(d\): number of differences.
- \(q\): moving-average error order.

The MA term in ARIMA is based on past forecast errors; it is not the same as a rolling arithmetic mean.

Seasonal ARIMA adds seasonal orders and a period, often written SARIMA \((p,d,q)(P,D,Q)_s\). ARIMA is useful for structured univariate signals, but model diagnostics, stationarity assumptions, missing intervals and changing regimes still matter.

A lag-feature regression model is another practical option. It can combine temporal history with exogenous telemetry such as queue depth, clock state and workload class.

## 5. Temporal validation: never shuffle time

A random train/test split can train on the future and evaluate on the past. For forecasting, that is usually invalid.

A simple chronological design is:

1. Earliest interval: training.
2. Next interval: validation and model/parameter selection.
3. Latest interval: final test, used once after selection.

For stronger evaluation, use rolling-origin backtesting:

~~~text
Fold 1: [ train ] [ validation ]
Fold 2: [    expanded train    ] [ validation ]
Fold 3: [          expanded train          ] [ validation ]
~~~

Scikit-learn's TimeSeriesSplit uses expanding training sets and later test folds. Its documentation notes that equally spaced samples are needed for comparable test durations. Its gap parameter can exclude observations immediately before the test fold.

### Metrics

\[
MAE=\frac{1}{N}\sum_t|y_t-\hat y_t|
\]

\[
RMSE=\sqrt{\frac{1}{N}\sum_t(y_t-\hat y_t)^2}
\]

- MAE is in the original units and is less dominated by large errors.
- RMSE penalizes large misses more strongly.
- MAPE can be unstable when actual values are zero or near zero.
- MASE compares a forecast against a naive scale and is useful across series.
- Prediction intervals or quantile forecasts communicate uncertainty, not just a point estimate.

Always report metrics by device, workload, operating mode and forecast horizon when those slices matter. A good aggregate score can hide a poor result for one device family.

## 6. Change-point detection from forecast residuals

Forecasting estimates expected behavior. Change-point detection asks whether the generating process has changed.

Define the residual:

\[
e_t=y_t-\hat y_t
\]

A persistent shift in residual mean, variance or distribution can indicate a new workload regime, firmware change, sensor recalibration, thermal or power shift, degradation or fault condition.

A single large residual may be noise. CUSUM accumulates small deviations to detect a sustained shift.

For standardized residual \(z_t=(e_t-\mu_0)/\sigma_0\), a one-sided positive CUSUM is:

\[
C_t^+=\max(0,C_{t-1}^+ + z_t-k)
\]

Raise a signal when \(C_t^+>h\).

- \(k\): allowance or reference value; it controls the shift size the detector is tuned to accumulate.
- \(h\): decision threshold; a larger threshold generally reduces false alarms but increases detection delay.
- \(\mu_0,\sigma_0\): baseline residual mean and standard deviation estimated from a clean reference interval.

A two-sided detector tracks positive and negative shifts. EWMA is another useful smoother for gradual changes. Page–Hinkley is a related online change detector.

**Important:** forecast residuals may be autocorrelated and non-Gaussian. The textbook CUSUM settings are starting points, not universal production thresholds. Calibrate false-alarm rate and detection delay on representative historical data, and consider residual whitening or robust scale estimates.

## 7. Worked SoC example

Assume a 15-minute DMA-latency monitor has a stable daily pattern. A firmware build is deployed at 14:00. Latency rises by a sustained amount while the workload and clock configuration remain unchanged.

A reasonable workflow is:

1. Validate timestamp cadence, device identity, units and firmware metadata.
2. Generate a one-step forecast from the last observed window.
3. Record forecast, actual value and residual.
4. Feed residuals to a CUSUM/EWMA monitor calibrated on a clean baseline.
5. If a policy threshold is crossed, create an investigation case.
6. Attach the firmware version, workload, queue depth, bus counters and nearby traces.
7. Ask a read-only assistant to summarize evidence and retrieve approved runbooks.
8. Have an engineer confirm whether this is an expected release effect, instrumentation issue or a real regression.

The time-series detector identifies a change; it does not establish root cause.

## 8. Hands-on lab: synthetic DMA latency forecasting

The runnable implementation is in [lab.py](./lab.py). It uses NumPy and scikit-learn and requires no external data.

### Dataset

- 14 days of synthetic samples.
- 15-minute sampling cadence: 96 samples per day.
- Daily and weekly components, slow trend and autocorrelated noise.
- A deliberate positive level shift introduced only in the held-out test interval.
- One target: synthetic DMA latency in milliseconds.

The code constructs lag-1, lag-2, lag-3, lag-4 and lag-96 features, plus short and daily rolling means. Every feature uses only values strictly before its target timestamp.

### Evaluation protocol

- 70% earliest interval: training.
- Next 15%: validation.
- Final 15%: test.
- Ridge regularization is selected on validation only.
- StandardScaler is inside a pipeline and is fitted only on training data.
- Test forecasts use one-step rolling evaluation: the latest actual observation is available before the next forecast.

### Reproduced lab results

Test set contains 202 one-step predictions.

| Forecast | MAE (ms) | RMSE (ms) |
|---|---:|---:|
| Persistence (lag 1) | 0.3039 | 0.3736 |
| Seasonal naive (lag 96) | 1.1053 | 1.3230 |
| Ridge lag model, alpha 0.01 | 0.3390 | 0.4169 |

The Ridge model was slightly better than persistence on the stable pre-change test segment:

| Ridge model segment | MAE (ms) | RMSE (ms) |
|---|---:|---:|
| Before injected change | 0.2960 | 0.3723 |
| After injected change | 0.3513 | 0.4289 |

Persistence adapted quickly after the level shift because each new actual sample became the next forecast. This illustrates why simple baselines can be strong and why regime-aware evaluation matters.

### CUSUM experiment

The lab uses seasonal-naive residuals and estimates the reference residual mean and standard deviation on validation data only.

| Setting / result | Value |
|---|---:|
| Allowance \(k\) | 0.5 standard deviations |
| Decision threshold \(h\) | 8.0 |
| Injected change index | 1187 |
| First CUSUM alarm index | 1191 |
| Detection delay | 4 samples = 60 minutes |

This seeded demonstration intentionally injects a detectable shift. The 60-minute delay is not a production performance guarantee.

### Run it

~~~bash
cd daily-lessons/day-015
python -m pip install -r requirements.txt
python lab.py
~~~

### Exercises

1. Change the daily period from 96 to another sampling cadence and update the seasonal lag.
2. Remove lag 96 and compare validation and test errors.
3. Increase the injected shift from 1.55 ms to 0.6 ms. Observe the CUSUM detection delay.
4. Increase \(h\) from 8 to 12. Measure the delay and explain the false-alarm trade-off.
5. Add a missing 15-minute interval. Decide whether to resample, mark missingness or reject the window; do not silently pretend the cadence is unchanged.
6. Replace the single fixed test interval with rolling-origin backtesting and report the distribution of errors across folds.
7. Add queue depth as an exogenous feature. Ensure it is known at the forecast timestamp and is not computed using future data.

## 9. Production agent architecture: Time-Series Forecasting and Change-Point Monitoring Gate (A21)

The LLM can help translate an engineer's question into a bounded analysis request and explain evidence. Deterministic services own timestamp validation, window construction, model execution, residual scoring, alert policy and state transitions.

~~~mermaid
flowchart TD
    A[Telemetry event] --> B[Identity, timestamp and schema gate]
    B --> C[Ordering, deduplication and watermark]
    C --> D[Versioned window and feature builder]
    D --> E[Deterministic forecast service]
    E --> F[Forecast plus prediction interval]
    F --> G[Residual calculation]
    G --> H[CUSUM / EWMA / change-point service]
    H --> I[Alert policy and idempotent case key]
    I --> J{Policy threshold crossed?}
    J -- No --> K[Persist state and continue monitoring]
    J -- Yes --> L[Create investigation event]
    L --> M[Read-only explanation agent]
    M --> N[Human engineer review]
    N --> O[Approved label and feedback]
    O --> P[Versioned baseline and model governance]
~~~

### Component boundaries

| Component | Owns | Must not do |
|---|---|---|
| Intake/router agent | Convert request into a typed analysis contract; select approved forecast or detector workflow | Invent telemetry or grant permissions |
| Event validator | Device scope, schema, units, timestamps, freshness and data quality | Silently reorder or fabricate missing samples |
| Stream/window service | Deduplication, event-time ordering, watermarks and lag/rolling features | Read beyond the forecast origin |
| Forecast service | Versioned model inference and prediction intervals | Fit a new model on the online request |
| Residual monitor | Residual calculation and CUSUM/EWMA state | Diagnose root cause |
| Policy gate | Thresholds, rate limits, severity and case creation | Trust free-form model confidence |
| Verifier | Check time ordering, model/feature version compatibility and finite outputs | Treat the same model output as independent proof |
| Explanation agent | Summarize evidence and retrieve approved runbooks | Execute shell, register writes, resets or firmware changes |
| Human reviewer | Confirm, reject, classify or escalate | Lose the audit trail or provenance |

### Pattern selection and trade-offs

- **Deterministic workflow:** default for validate → window → forecast → residual → policy. It is easiest to test and reason about.
- **Single agent with typed tools:** appropriate for a read-only engineering assistant that calls get_forecast, get_residual_history, get_device_context and fetch_approved_runbook tools.
- **Router-specialist:** route between univariate forecast, multivariate forecast, CUSUM and anomaly investigation. Use deterministic eligibility rules; model-based routing may suggest but cannot authorize.
- **Planner–executor–verifier:** useful for offline investigations that compare bounded horizons, lag sets or forecast models. The verifier independently checks the split, metrics and artifact lineage.
- **Supervisor–worker:** parallelize independent device cohorts or model candidates only when they share explicit time boundaries and resource budgets. Avoid duplicate scoring of the same event.
- **Event-driven:** process telemetry with event IDs, per-device sequence numbers, watermarks, checkpoints and idempotent case creation. Handle late events explicitly; send poison messages to a dead-letter queue.
- **Human-in-the-loop:** require review before changing production thresholds, declaring a silicon fault, opening a high-severity incident or initiating remediation.

### State and memory

Separate three kinds of state:

1. **Stream state:** last processed event ID, watermark, lag buffer, rolling statistics and CUSUM/EWMA accumulator. This must be durable and scoped by device and signal.
2. **Model state:** model version, training interval, feature schema, scaler, forecast horizon, residual baseline and threshold policy.
3. **Investigation memory:** confirmed incidents, reviewer notes and approved runbook links. Store provenance and tenant/device scope; do not treat an LLM summary as ground truth.

A restart must restore stream state from a checkpoint without double-counting events. Model and feature versions must be compatible before scoring resumes.

### Typed tool contract example

~~~json
{
  "name": "forecast_signal",
  "input": {
    "device_id": "string",
    "signal_id": "string",
    "window_end_utc": "RFC3339 timestamp",
    "horizon_steps": "integer, 1..96",
    "feature_schema_version": "string"
  },
  "output": {
    "forecast_id": "string",
    "model_version": "string",
    "forecast_value": "number",
    "lower_bound": "number",
    "upper_bound": "number",
    "units": "string",
    "data_watermark_utc": "RFC3339 timestamp"
  }
}
~~~

Validate bounds, identity and authorization outside the model. Never accept arbitrary SQL, shell commands or user-provided file paths as tool arguments.

### Security, reliability and observability

- Treat log text, firmware strings and telemetry metadata as untrusted data; they are evidence, not instructions.
- Enforce device/tenant authorization in the tool service, not in the prompt.
- Keep credentials out of prompts and traces; redact identifiers where policy requires.
- Use idempotency keys such as device + signal + window-end + policy-version.
- Bound retries, forecast horizon, window size, concurrency and model execution time.
- Distinguish late data, duplicate data, missing samples and genuine zeros.
- Record trace ID, event ID, device scope, model/feature/policy versions, forecast origin, horizon, score, threshold, decision and reviewer action.
- Monitor data freshness, event lag, missingness, forecast MAE/RMSE by cohort, residual mean/variance, alerts per device-day, false-positive rate, detection delay, CPU/memory, p50/p95 latency and cost per monitored device.
- Use drift and performance gates before retraining; do not automatically promote a new baseline after a suspected fault.

### Evaluation contract

A production acceptance suite should include:

1. No future values in feature construction.
2. Chronological split and rolling-origin backtest.
3. Forecast metrics by horizon and operating mode.
4. Residual calibration and prediction-interval coverage.
5. Change-point detection delay and false alarms per device-day.
6. Missing, duplicate, late and out-of-order event tests.
7. Restart/replay idempotency tests.
8. Model/feature schema mismatch tests.
9. Prompt-injection and unauthorized-tool-call tests.
10. End-to-end cost and latency budgets.

## 10. Five review questions

1. Why is a shuffled train/test split usually invalid for forecasting?
2. What is the difference between a lag feature and a rolling-window feature, and how can either leak future information?
3. In ARIMA(p,d,q), what do p, d and q represent? Why is the MA term not the same as a rolling average?
4. Why can a persistence forecast beat a more complex regression model after a sudden regime change?
5. What do CUSUM's k and h control, and why must the baseline residual distribution be estimated from a clean reference interval?

## 11. Recommended references

1. Hyndman and Athanasopoulos, **Forecasting: Principles and Practice (3rd edition)** — [online book](https://otexts.com/fpp3/). A practical reference for decomposition, forecasting methods, residual diagnostics and time-series cross-validation.
2. scikit-learn — [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Expanding-window temporal validation, test size and gap.
3. statsmodels — [ARIMA API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html). ARIMA and seasonal model parameters.
4. NIST/SEMATECH e-Handbook — [CUSUM control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm). Cumulative-sum monitoring and sensitivity to small process shifts.
5. NIST/SEMATECH e-Handbook — [CUSUM average run length](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm). False-alarm and detection behavior.

## Session takeaway

Time-series ML is not ordinary tabular ML with a timestamp column. Preserve temporal order, construct features only from information available at the forecast origin, compare against naive baselines, evaluate across future windows, and monitor residuals for sustained change. In production, make stream state durable and idempotent, version the feature/model/baseline contract, and route consequential diagnoses through independent checks and human review.
