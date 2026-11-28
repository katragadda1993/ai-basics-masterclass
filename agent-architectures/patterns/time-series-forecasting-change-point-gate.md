# A21 — Time-Series Forecasting and Change-Point Monitoring Gate

**Purpose:** forecast telemetry at a declared horizon and detect sustained changes in forecast residuals without allowing an LLM to control time ordering, thresholds, permissions or remediation.

**Default principle:** deterministic stream and forecasting services own numerical state; agents assist with bounded analysis and evidence explanation.

## When to use

Use this pattern when signals are ordered in time and the operational question involves expected future values, gradual drift, recurring behavior or a sustained process change. Examples include DMA latency, queue depth, PCIe replay counters, SoundWire error rates, codec latency and temperature.

Do not use a time-series model when timestamps are unreliable, sampling semantics are unknown or the signal is better represented as independent events. Repair the data contract first.

## Architecture

~~~mermaid
flowchart TD
    A[Telemetry event] --> B[Identity and schema validation]
    B --> C[Event-time ordering and deduplication]
    C --> D[Watermark and durable window state]
    D --> E[Versioned feature builder]
    E --> F[Forecast service]
    F --> G[Prediction and interval]
    G --> H[Residual calculation]
    H --> I[CUSUM / EWMA / change-point detector]
    I --> J[Deterministic alert policy]
    J --> K{Threshold crossed?}
    K -- No --> L[Persist state and continue]
    K -- Yes --> M[Idempotent investigation case]
    M --> N[Read-only explanation agent]
    N --> O[Human engineering review]
    O --> P[Versioned feedback and model governance]
~~~

## Component contracts

| Component | Input | Output | Authority |
|---|---|---|---|
| Intake/router agent | Engineer request | Typed analysis request | May select an approved workflow; no data permission |
| Event validator | Raw event | Validated event or rejection reason | Enforces identity, schema, units, timestamp and freshness |
| Stream/window service | Validated events | Ordered windows, watermark and checkpoint | Owns deduplication, late-event policy and durable stream state |
| Feature builder | Window plus schema version | Lag/rolling/calendar/exogenous feature vector | Uses only data available at forecast origin |
| Forecast service | Feature vector and model version | Forecast, interval, units and provenance | Read-only inference; no online fitting |
| Residual monitor | Actual, forecast and baseline version | Residual and change score | Owns CUSUM/EWMA state |
| Policy gate | Score, threshold and context | No-op or investigation event | Deterministic severity, rate limits and idempotency |
| Verifier | Event, feature, model and policy metadata | Pass/fail with reasons | Independently checks ordering, compatibility and finite values |
| Explanation agent | Approved evidence bundle | Cited summary and follow-up suggestions | Read-only; cannot execute remediation |
| Human reviewer | Case and evidence | Confirm/reject/escalate decision | Owns consequential diagnosis and approval |

## Typed tool example

A forecast tool should accept bounded, typed fields rather than arbitrary query text:

~~~json
{
  "name": "forecast_signal",
  "input_schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["device_id", "signal_id", "window_end_utc", "horizon_steps", "feature_schema_version"],
    "properties": {
      "device_id": {"type": "string", "minLength": 1, "maxLength": 128},
      "signal_id": {"type": "string", "minLength": 1, "maxLength": 128},
      "window_end_utc": {"type": "string", "format": "date-time"},
      "horizon_steps": {"type": "integer", "minimum": 1, "maximum": 96},
      "feature_schema_version": {"type": "string", "minLength": 1, "maxLength": 64}
    }
  },
  "output_schema": {
    "type": "object",
    "required": ["forecast_id", "model_version", "forecast_value", "lower_bound", "upper_bound", "units", "data_watermark_utc"],
    "properties": {
      "forecast_id": {"type": "string"},
      "model_version": {"type": "string"},
      "forecast_value": {"type": "number"},
      "lower_bound": {"type": "number"},
      "upper_bound": {"type": "number"},
      "units": {"type": "string"},
      "data_watermark_utc": {"type": "string", "format": "date-time"}
    }
  }
}
~~~

The tool service, not the LLM, must enforce caller identity, device scope, allowed horizons, model access and rate limits.

## State and memory model

Keep distinct, versioned state domains:

1. **Stream state:** last event ID, sequence number, watermark, reorder buffer, lag buffer, rolling aggregates, CUSUM/EWMA accumulator and checkpoint.
2. **Model state:** model artifact/version, training interval, feature schema, scaler, horizon, seasonal period, residual baseline and prediction-interval calibration.
3. **Investigation memory:** confirmed incidents, reviewer labels, approved runbook references and provenance. Scope it by tenant/device and retain according to policy.

On restart, restore stream state before consuming new events. Use event ID plus device/signal/window-end/policy-version as an idempotency key. Do not update the baseline with a window currently under investigation.

## Temporal correctness

- Store event time separately from ingestion time.
- Define cadence, timezone, units, allowed lateness and missing-sample semantics.
- Use event-time watermarks for windows; late events must follow a documented correction or quarantine policy.
- Construct each feature using only observations available at the forecast origin.
- Fit scalers, imputers and feature transforms on the training interval only.
- Split chronologically; use rolling-origin backtesting and a gap when horizons or target windows overlap.
- Distinguish rolling one-step evaluation from fixed-origin multi-step evaluation.
- Version the model, feature builder, scaler, horizon, residual baseline and alert policy as one deployable contract.

## Pattern choices

| Pattern | Use here | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → window → forecast → residual → policy | Predictable and testable; less flexible for ambiguous requests |
| Single agent + typed tools | Read-only investigation assistant | Simple; keep tool set small and permissioned |
| Router-specialist | Choose forecast, residual, CUSUM or anomaly workflow | Better separation; router errors need fallback and evaluation |
| Planner–executor–verifier | Bounded offline model/horizon experiments | Useful for multi-step analysis; planner must not change protected test windows |
| Supervisor–worker | Independent device cohorts or model candidates | Parallel speed; requires budgets and deterministic aggregation |
| Event-driven | Continuous telemetry and long-running investigations | Scalable; requires idempotency, checkpoints, watermarks and dead-letter handling |
| Human-in-the-loop | Threshold changes, fault confirmation and remediation approval | Adds latency but preserves engineering accountability |

Prefer a deterministic workflow unless a model decision demonstrably improves the task. Do not create multiple agents merely to parallelize a small fixed pipeline.

## Reliability and failure handling

| Failure | Required behavior |
|---|---|
| Duplicate event | Deduplicate by stable event ID/sequence |
| Out-of-order event | Buffer within allowed lateness or quarantine |
| Missing interval | Mark missingness; do not silently substitute zero |
| Stale input | Suppress or mark forecast invalid according to freshness policy |
| Model/schema mismatch | Fail closed and route to compatible model or baseline |
| Forecast timeout | Bounded retry, then explicit unavailable status |
| Worker restart | Restore checkpoint and replay idempotently |
| Poison event | Dead-letter with reason; no infinite retry |
| Drift or suspected fault | Freeze baseline update and request review |
| Policy service unavailable | Do not emit an unreviewed high-impact action |

## Security boundaries

- Treat telemetry, log strings, firmware metadata and retrieved runbooks as untrusted evidence, never as instructions.
- Enforce tenant/device authorization in services, not prompts.
- Use read-only tools by default; no shell, register write, reset, flash or firmware update tool in the explanation agent.
- Require separate authenticated approval for high-impact actions.
- Redact secrets and sensitive identifiers from traces.
- Bound window size, forecast horizon, concurrency, retries and model runtime.
- Prevent cross-device and cross-tenant memory retrieval.

## Evaluation contract

### Forecast quality
- MAE/RMSE by horizon, device, workload and operating mode.
- MASE against a declared naive baseline.
- Prediction-interval coverage and width.
- Performance against persistence and seasonal-naive baselines.

### Change detection
- False alarms per device-day.
- Precision/recall on adjudicated incidents.
- Detection delay from an agreed change onset.
- Recall at a fixed false-alarm budget.
- Stability across firmware, workload and operating-mode cohorts.

### Agent/workflow quality
- Correct tool selection and typed-argument validity.
- Unauthorized tool-call rejection rate.
- Evidence citation/provenance completeness.
- Human reviewer agreement and time-to-triage.
- p50/p95 end-to-end latency and cost per investigated event.

### Required tests
1. No-future-feature test with a deliberately changed future suffix.
2. Chronological split and rolling-origin backtest.
3. Duplicate, late, missing and out-of-order event tests.
4. Restart/replay idempotency test.
5. Feature/model/policy version mismatch test.
6. CUSUM/EWMA false-alarm and delay tests on stable and shifted streams.
7. Prompt-injection and unauthorized-tool-call tests.
8. Budget, cancellation and timeout tests.

## Observability contract

Every forecast and alert should carry:

- trace ID, event ID and idempotency key;
- tenant/device/signal scope (redacted as required);
- event timestamp, ingestion timestamp and watermark;
- feature schema and window-builder version;
- model, scaler, horizon and residual-baseline versions;
- forecast, interval, actual, residual and change score;
- threshold policy and decision reason;
- queue lag, runtime, retry count and resource usage;
- case ID and reviewer action.

Monitor data freshness, missingness, queue lag, forecast error, residual distribution, alerts/device-day, detection delay, p50/p95 latency, CPU/memory and cost per monitored device.

## Cost and latency controls

- Use a small deterministic model for high-rate scoring; invoke an LLM only when a case needs explanation.
- Batch compatible offline scoring; avoid batching across incompatible model/schema versions.
- Cache immutable model metadata and approved runbooks, not mutable per-device stream state.
- Set per-device and global concurrency limits.
- Propagate deadlines and cancellation.
- Measure cost per monitored device-day and per useful investigation, not just tokens per call.

## Reference implementation exercise

See [Session 015](../../daily-lessons/day-015/README.md) and its runnable [lab.py](../../daily-lessons/day-015/lab.py). The lab compares persistence, seasonal-naive and Ridge lag forecasts, then applies a validation-calibrated CUSUM to held-out seasonal-naive residuals.

## Acceptance criteria for a production implementation

- [ ] Temporal and event-time contract documented.
- [ ] Feature construction is proven not to read future values.
- [ ] Training/validation/test intervals are immutable and chronological.
- [ ] Naive baselines and rolling-origin metrics are recorded.
- [ ] Stream state, model state and investigation memory are separated.
- [ ] Duplicate, late, missing and replay behavior is tested.
- [ ] Threshold false-alarm rate and detection delay are calibrated.
- [ ] Tool schemas and authorization are enforced outside the LLM.
- [ ] Consequential diagnosis and remediation require human approval.
- [ ] Traces, audit records, cost and latency budgets are observable.

## References

- Hyndman and Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/).
- scikit-learn, [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- statsmodels, [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html).
- NIST/SEMATECH, [CUSUM control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm).

**Design rule:** A forecast is not a diagnosis. A change-point alert is not a fault confirmation. Preserve temporal provenance, enforce policy deterministically, and require independent evidence and human review before consequential action.
