# A24 — CNN and Signal-Representation Training Gate

## Purpose

A24 governs experiments that train convolutional models on images, sampled waveforms or time-frequency representations. An assistant may help choose an approved modality path, propose bounded experiments and explain metrics. Deterministic services own acquisition metadata, preprocessing, tensor execution, evaluation, artifact lineage and release.

**Core principle:** the representation pipeline is part of the model contract. A model trained on one channel order, sample rate, spectrogram configuration or normalization must not silently receive another representation at inference.

## When to use it

Use A24 when:
- The input has meaningful spatial, temporal-local or time-frequency structure.
- Experiments compare 1D waveform CNNs, 2D image/spectrogram CNNs or approved alternatives.
- Training must be reproducible and tied to acquisition and preprocessing metadata.
- An assistant should help engineers without controlling data access, accelerator allocation or release.

For ordinary tabular counters, start with a tree/linear model or MLP. Do not reshape arbitrary counters into an image solely to use Conv2d.

## Architecture

~~~mermaid
flowchart TD
    A[Engineer objective] --> B[Read-only intake agent]
    B --> C[Typed modality and dataset contract]
    C --> D[Identity, schema and split validator]
    D --> E[Versioned representation pipeline]
    E --> F{Input modality router}
    F -->|Image| G[2D CNN worker]
    F -->|Waveform| H[1D CNN worker]
    F -->|Spectrogram| I[2D time-frequency CNN worker]
    G --> J[Deterministic trainer]
    H --> J
    I --> J
    J --> K[Metrics and model artifact]
    K --> L[Independent verifier]
    L --> M[Validation and robustness gate]
    M --> N{Candidate acceptable?}
    N -->|No| O[Bounded retry or reject]
    N -->|Yes| P[Frozen candidate]
    P --> Q[Protected test gate]
    Q --> R[Human approval]
    R --> S[Model registry]
~~~

## Component contracts

| Component | Input | Output | Authority |
|---|---|---|---|
| Intake agent | Engineer goal and approved project metadata | Proposed typed request | No arbitrary code, dataset or accelerator authority |
| Contract validator | Typed request | Validated modality/config or rejection | Enforces identity, schema, ranges, ACL and budget |
| Representation pipeline | Approved raw data and transform config | Versioned tensors/features and manifest | Deterministic, versioned transform |
| Modality router | Validated modality and task contract | Approved 1D, 2D or baseline route | Routing is not authorization; policy validates it |
| CNN worker | Approved tensors, architecture and device lease | Metrics and checkpoint | Reads only authorized training data |
| Evaluator | Frozen candidate and authorized split | Metrics and robustness report | Validation during selection; test after freeze |
| Verifier | Artifact, metadata, metrics and lineage | Pass/fail evidence | Independent deterministic checks |
| Approval/registry gate | Verified artifact and authorized decision | Immutable model version | Requires policy and human owner approval |

## Typed representation contract

Example image request:

~~~json
{
  "run_id": "audio-cnn-018",
  "task": "three_class_shape_demo",
  "modality": "image",
  "tensor_layout": "NCHW",
  "input_shape": [1, 16, 16],
  "dtype": "float32",
  "normalization_version": "unit-range-v1",
  "label_schema": "orientation-v1",
  "dataset_version": "synthetic-shapes-v1",
  "split_manifest_id": "seeded-split-018",
  "architecture_id": "small-cnn-1d8-8d16",
  "budget": {
    "max_wall_seconds": 600,
    "max_gpu_hours": 0.1,
    "max_concurrent_runs": 1
  }
}
~~~

A production waveform/spectrogram contract should additionally carry:
- Sample rate and resampling implementation/version.
- Channel mapping and channel count.
- Sample format, units, clipping policy and time origin.
- Window function, FFT size, hop length and centering policy.
- Frequency scale, mel filter-bank version and log floor if applicable.
- Normalization statistics and their training-data provenance.
- Crop, padding and augmentation policy.
- Label schema, device/session identity and immutable split manifest.

Reject unknown fields, unsupported dimensions, missing acquisition metadata, unauthorized datasets and out-of-policy budgets. Never accept executable Python, shell commands or arbitrary filesystem paths from an LLM.

## Seven agent architecture patterns

| Pattern | A24 application | Trade-off and guardrail |
|---|---|---|
| Deterministic workflow | Validate → transform → train → evaluate → package | Default for numerical and policy-critical steps |
| Single agent + typed tools | Read metrics, inspect feature maps and summarize a run | Start read-only; no arbitrary code execution |
| Router-specialist | Route waveform, spectrogram, image or tabular tasks | Route only among approved paths; deterministic fallback |
| Planner–executor–verifier | Plan bounded kernel/augmentation/architecture experiments | Persist plan; cap trials, tokens, wall time and accelerator hours |
| Supervisor–worker | Parallel independent seeds or candidate architectures | Cap concurrency and memory; use identical split manifests |
| Event-driven | Queue training and publish checkpoint/evaluation events | Durable state, idempotency keys, bounded retries and dead-letter handling |
| Human-in-the-loop | Approve data changes, alert thresholds and model promotion | Auditable reviewer, rationale, timeout and escalation |

Start with a deterministic pipeline and one read-only assistant. Add specialists or workers only when their context, permissions or independent compute requirements justify them.

## State, memory and artifact lineage

Persist:
- Run ID, tenant/project, owner and authorization scope.
- Raw dataset fingerprint, acquisition metadata and immutable split-manifest ID.
- Input modality, tensor layout, shape, dtype and label mapping.
- Complete preprocessing/representation configuration and version.
- Model architecture, initialization seed, optimizer and scheduler.
- Code commit, environment digest, framework version and device metadata.
- Checkpoint hash, validation metrics, robustness tests and selected epoch.
- Test evaluation record created only after candidate freeze.
- Human approval identity, rationale, timestamp and registry artifact version.

Conversation memory is not model state. A chat transcript must never replace the dataset manifest or representation configuration.

## Security and trust boundaries

Threats include prompt injection in filenames/logs/metadata, unauthorized audio or image access, compute exhaustion, test leakage, malicious checkpoints, secret exposure and arbitrary-code execution.

Controls:
1. Treat filenames, image metadata, audio transcripts, logs and dataset rows as untrusted content.
2. Keep raw private media out of LLM context unless explicitly authorized and minimized.
3. Enforce dataset ACLs, tenant boundaries and accelerator quotas in deterministic services.
4. Use typed, allow-listed tools; do not expose unrestricted shell or Python execution.
5. Keep training, validation and protected test permissions separate.
6. Verify artifact hashes and provenance; load only trusted checkpoint artifacts.
7. Require human approval for production promotion and safety-relevant threshold changes.
8. Redact secrets, private waveforms, image content and identifiers from logs/traces.

## Reliability and recovery

- Derive idempotency keys from run ID plus normalized configuration hash.
- Make queue submission and event consumption idempotent.
- Write checkpoints atomically and include artifact hashes/manifests.
- Verify model shape, preprocessing version and label schema before resume or inference.
- Bound image resolution, waveform duration, spectrogram bins, batch size, worker count and concurrent runs.
- Distinguish retryable infrastructure failures from deterministic data/model failures.
- Propagate cancellation and release accelerator allocations.
- Never silently run a checkpoint with a different sample rate, channel order or normalization.

## Evaluation contract

Separate:
1. **Implementation correctness:** tensor shape checks, finite loss/gradients, parameter count, checkpoint load and deterministic smoke tests.
2. **Model quality:** per-class precision/recall, confusion matrix, calibration, abstention and device/session-separated test results.
3. **Robustness:** noise, amplitude, time shift, missing channel, resampling and operating-mode changes.
4. **Operational fitness:** preprocessing time, examples/second, inference p50/p95, peak memory, accelerator utilization and cost per sample.

For audio/silicon diagnostics, measure false alarms per device-hour and missed-event rate. A high image accuracy on synthetic data is not evidence of field performance.

## Observability

Trace fields should include:
- Trace ID, run ID, event ID, idempotency key and actor.
- Dataset fingerprint, split manifest and acquisition metadata version.
- Modality, shape, dtype, layout and preprocessing hash.
- Model architecture/configuration hash, seed and code/environment version.
- Train/validation metrics and test-gate transition.
- Batch size, data wait, transfer, forward/backward and checkpoint durations.
- Device, accelerator memory, utilization and failure/retry count.
- Artifact hash, reviewer decision and registry version.
- Assistant model/prompt version, tool calls, token usage and latency.

Redact raw private media and secrets. Measure p50/p95 end-to-end latency and cost per successful experiment, not just model-forward time.

## Cost and latency controls

- Use small CPU smoke tests before GPU allocation.
- Profile data decoding, feature extraction, host-to-device transfer and convolution separately.
- Bound resolution, duration, spectrogram size, batch size and concurrency.
- Benchmark channels-last and mixed precision only on supported devices and verify numerical quality.
- Compare CNNs against an appropriate simpler baseline on identical splits.
- Use LLMs at coarse experiment boundaries, never per frame or training batch.
- Propagate deadlines and cancellation through queues and workers.

## State machine

~~~text
REQUESTED
  -> VALIDATING
  -> REPRESENTATION_READY
  -> QUEUED
  -> RUNNING
  -> CHECKPOINTED <-> RUNNING
  -> VALIDATING_CANDIDATE
  -> TEST_FROZEN
  -> TESTED
  -> AWAITING_APPROVAL
  -> REGISTERED

Failure branches:
  VALIDATING -> REJECTED
  RUNNING -> RETRYABLE_FAILURE -> QUEUED (bounded)
  RUNNING -> FAILED
  AWAITING_APPROVAL -> REJECTED or EXPIRED
~~~

Every transition records actor, timestamp, reason and artifact hashes.

## Reference implementation

Session 018's [synthetic PyTorch CNN lab](../../daily-lessons/day-018/lab.py) demonstrates NCHW tensor contracts, Conv2d, ReLU, pooling, shape assertions, parameter counting, deterministic training and validation-selected evaluation. It is a teaching reference, not a production audio or silicon classifier.

## Required acceptance criteria

- Typed modality contract rejects missing/invalid acquisition and tensor metadata.
- Representation output shape and preprocessing hash are checked.
- Dataset and split manifest are immutable for a run.
- CPU and supported accelerator paths pass smoke tests.
- No non-finite loss, gradient or parameter values.
- Artifact can be loaded only with matching preprocessing and label schema.
- Validation selection never reads protected test labels.
- Independent verifier confirms artifact lineage and metrics.
- Robustness tests cover relevant acquisition variations.
- Human approval is required for registry promotion.
- Traces expose preprocessing, transfer, compute, memory, latency and cost without leaking private media.

## References

- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Learning — Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [PyTorch Channels Last Memory Format](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
