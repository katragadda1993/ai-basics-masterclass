# A25 — Sequence Modeling and Stateful Inference Governance Gate

**Track:** Reusable AI agent architectures  
**Related lesson:** Session 019 — RNN, LSTM, BPTT, vanishing gradients and GRU  
**Status:** Reference architecture specification  
**Primary rule:** recurrent numeric state is owned by a deterministic model runtime and isolated per stream; an LLM assistant may inspect evidence but must not mutate hidden state or authorize hardware actions.

## 1. Problem

Sequence models can learn temporal patterns in audio frames, protocol events and hardware telemetry. Production systems add difficult concerns that are not solved by selecting an LSTM layer:

- Event-time ordering, duplicates, gaps and late events.
- Device/session identity and state isolation.
- Variable-length windows and padding.
- Recurrent hidden/cell state lifecycle.
- Truncated BPTT and checkpoint recovery.
- Temporal and device-separated evaluation.
- Model-version compatibility and safe rollout.

This architecture keeps the agentic control plane separate from the numerical sequence-model data plane.

## 2. Architecture

~~~mermaid
flowchart TD
    A[Telemetry or audio event] --> B[Identity and schema validator]
    B --> C[Event-time ordering and window builder]
    C --> D[Versioned feature pipeline]
    D --> E[Model-family router]
    E --> F[RNN candidate worker]
    E --> G[LSTM candidate worker]
    E --> H[GRU candidate worker]
    F --> I[Deterministic training runtime]
    G --> I
    H --> I
    I --> J[Gradient, metric and artifact verifier]
    J --> K[Temporal/device-separated evaluation]
    K --> L{Acceptance policy}
    L -->|Reject| M[Bounded retry or report]
    L -->|Pass| N[Frozen candidate]
    N --> O[Protected test gate]
    O --> P[Human approval]
    P --> Q[Model registry]
    Q --> R[Stateful inference service]
    R --> S[Per-stream versioned state store]
~~~

## 3. Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Event ingestion | Identity, timestamps, offsets, deduplication | Model interpretation |
| Window builder | Ordering, gaps, padding and window contract | Model promotion |
| Feature pipeline | Scaling, feature order and schema version | Tenant authorization |
| Model router | Candidate family selection under policy | Arbitrary tool permissions |
| Training runtime | Tensor operations, BPTT, optimizer and checkpoints | LLM-driven gradient updates |
| Evaluator | Temporal/device splits and metrics | Final approval |
| State store | Per-stream h/c state, version and expiry | Conversation memory |
| Agent assistant | Read-only evidence summary and experiment proposal | Hidden-state mutation or hardware writes |
| Human reviewer | Approval, rejection and rationale | Unlogged side effects |

## 4. Pattern mapping

| Pattern | Where it fits | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → order → window → train/evaluate → release | Best reproducibility; less adaptive |
| Single agent + typed tools | Read-only trace/metric/model inspection | Low orchestration overhead; limited autonomy |
| Router-specialist | Route to RNN/LSTM/GRU/CNN/Transformer candidates | Better specialization; routing errors need fallback |
| Planner–executor–verifier | Plan bounded experiments, execute, independently check | Better control; additional latency |
| Supervisor–worker | Parallel seeds, model families or device cohorts | Throughput; aggregation and compute cost |
| Event-driven | Continuous telemetry and asynchronous chunk processing | Scalable; ordering, replay and idempotency complexity |
| Human-in-the-loop | Model rollout, state reset policy and diagnostic threshold approval | Safer high-impact decisions; slower release |

Prefer the simplest pattern that satisfies the requirement. Parallel workers are appropriate only for independent candidates with explicit merge rules.

## 5. Typed contracts

Example sequence-window request:

~~~json
{
  "schema_version": "1.0",
  "tenant_id": "tenant-scope-from-auth",
  "device_id": "device-identifier",
  "session_id": "capture-session",
  "stream_id": "stream-identifier",
  "first_event_time": "2026-11-28T10:00:00Z",
  "last_event_time": "2026-11-28T10:00:10Z",
  "sequence_offset_start": 1200,
  "sequence_offset_end": 1249,
  "feature_schema": "dma-telemetry-v3",
  "feature_count": 5,
  "sequence_length": 50,
  "model_version": "lstm-dma-v2"
}
~~~

The tenant identity must come from authenticated context, not an LLM-generated argument. Validate monotonic offsets, timestamp policy, finite numeric values, feature order and model compatibility.

Example state record:

~~~json
{
  "stream_id": "stream-identifier",
  "device_id": "device-identifier",
  "model_version": "lstm-dma-v2",
  "feature_schema": "dma-telemetry-v3",
  "last_sequence_offset": 1249,
  "hidden_state_ref": "encrypted-state-reference",
  "cell_state_ref": "encrypted-state-reference",
  "expires_at": "2026-11-28T10:05:00Z",
  "state_version": 2
}
~~~

State records should be encrypted, access-controlled and retained only as long as needed. The example identifiers are placeholders, not real credentials or identities.

## 6. State and memory rules

1. Key recurrent state by tenant, device, session/stream and model version.
2. Never share hidden/cell state across devices, tenants or independent captures.
3. Reset state on a new session, incompatible model/schema version, excessive gap or explicit reset event.
4. Persist last processed offset with state so replay can be idempotent.
5. Treat duplicate offsets as duplicates; define a deterministic policy for late/out-of-order events.
6. Keep conversational memory separate from numerical hidden/cell state.
7. Define state TTL and recovery behavior after process restart.
8. If state is missing or invalid, initialize safely or fall back to a documented cold-start path; do not silently reuse another stream's state.

## 7. Training and checkpoint contract

A reproducible checkpoint manifest should include:

- Model architecture and hyperparameters.
- Model and optimizer state dictionaries.
- Scheduler and mixed-precision scaler state, when used.
- Epoch, global step and truncated-BPTT chunk offset.
- CPU/CUDA RNG states and sampler/data-loader state.
- Dataset fingerprint and immutable split manifest.
- Feature scaler, feature order and acquisition metadata.
- Sequence length, padding and missing-event policy.
- Framework/runtime versions and deterministic settings.
- Gradient clipping threshold and observed gradient statistics.

Truncated BPTT carries the numerical state to the next chunk but detaches the graph at the configured boundary. The chosen chunk length limits memory and gradient horizon; record it in experiment metadata.

## 8. Evaluation contract

Use device-, firmware- and time-separated evaluation where the deployment scenario requires generalization to unseen devices or later firmware. Avoid overlapping windows crossing split boundaries.

Required measurements:

- Classification: per-class precision/recall/F1, PR-AUC, calibration and confusion matrix.
- Detection: false alarms per device-hour, event-level recall and detection delay.
- Forecasting: MAE/RMSE or task-specific loss with rolling-origin evaluation.
- Robustness: missing samples, duplicate events, jitter, sequence truncation, drift and restart.
- Systems: throughput, p50/p95 latency, peak memory and state-store load.
- Compare with persistence, threshold, temporal CNN and simpler recurrent baselines.

A model score is evidence for triage, not a root-cause diagnosis.

## 9. Security and safety

- Authenticate and authorize device/tenant scope outside the model and LLM.
- Treat firmware logs, traces and captured text as untrusted content.
- Expose read-only tools to the assistant by default.
- Never let the assistant choose arbitrary device IDs or retrieve another tenant's state.
- Do not load untrusted serialized checkpoints; use approved artifact registries and safe formats.
- Require independent policy evaluation and human approval for model rollout or high-impact diagnostic actions.
- Redact sensitive payloads from logs while preserving trace IDs, offsets and model versions.

## 10. Reliability and event processing

Use at-least-once event delivery with idempotent state transitions keyed by stream ID and sequence offset. Do not assume exactly-once execution across queues and external state stores.

Define:
- Bounded retries and dead-letter handling.
- Watermarks and late-event policy.
- Checkpoint frequency and recovery point objective.
- State-store transaction/compare-and-swap behavior.
- Cancellation and model rollout rollback.
- Fallback behavior for invalid state or model-version mismatch.

## 11. Observability and cost/latency

Record trace ID, tenant-safe scope, device/session pseudonymous ID, event offset, model version, feature schema, sequence length, queue age, state reset count, dropped/late events, inference duration, memory, and decision outcome.

Monitor p50/p95 latency, throughput, state-store hit/miss rate, GPU utilization and cost per processed sequence. Keep LLM calls off the per-event hot path; invoke the assistant only when investigation or human-readable explanation is useful.

## 12. Failure-mode test matrix

| Failure | Expected behavior |
|---|---|
| Duplicate event offset | Idempotent no-op or deterministic duplicate result |
| Late event | Apply documented reorder/watermark policy |
| New session | Reset recurrent state |
| Model version changed | Reject incompatible state and initialize/migrate explicitly |
| State store unavailable | Fail safely or use documented cold-start fallback |
| NaN/Inf feature | Reject/quarantine; do not score |
| Sequence too short | Apply minimum-length policy or abstain |
| Gradient explosion | Clip, record metric, fail run if still non-finite |
| Worker restart | Restore compatible checkpoint and data cursor |
| Cross-tenant request | Deny before retrieval or model execution |
| LLM prompt injection in trace | Treat trace as data; tool policy remains unchanged |
| Test-set access request | Deny during experiment planning and validation |

## 13. Acceptance criteria

A candidate is eligible for human review only when:

- Data and feature contracts validate.
- Train/validation/test boundaries are immutable and leakage-tested.
- All losses, gradients and parameters remain finite.
- Recurrent state is isolated and version-compatible.
- Temporal/device-separated metrics meet the declared objective.
- Robustness and latency budgets pass.
- Artifact lineage and checkpoint restore tests pass.
- Independent verifier reproduces the evaluation summary.
- A human records approval before registry promotion.

## 14. Reference implementation layout

~~~text
sequence-modeling-gate/
├── contracts/
│   ├── sequence_window.schema.json
│   └── recurrent_state.schema.json
├── ingestion/
│   ├── ordering.py
│   └── idempotency.py
├── features/
│   └── versioned_sequence_features.py
├── models/
│   ├── rnn.py
│   ├── lstm.py
│   └── gru.py
├── training/
│   ├── train.py
│   ├── truncated_bptt.py
│   └── checkpoint.py
├── evaluation/
│   ├── temporal_split.py
│   ├── metrics.py
│   └── robustness.py
├── serving/
│   ├── state_store.py
│   └── streaming_inference.py
├── agent/
│   ├── read_only_tools.py
│   └── investigation_summary.py
└── tests/
    ├── test_state_isolation.py
    ├── test_replay_idempotency.py
    ├── test_temporal_leakage.py
    └── test_model_version_compatibility.py
~~~

## 15. Related lesson

See [Session 019 — Recurrent Neural Networks](../../daily-lessons/day-019/README.md) for the runnable PyTorch lab, equations, gradient-path experiment and five review questions.
