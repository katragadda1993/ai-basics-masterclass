# A23 — PyTorch Training Runtime and GPU-Aware Execution Gate

## Purpose

A23 is a production-minded architecture for training PyTorch models under explicit data, compute, security and release controls. An assistant may help translate an engineer's goal into a typed experiment request, explain metrics and investigate failures. Deterministic services own tensor execution, device allocation, checkpointing, evaluation and model promotion.

**Core principle:** an LLM is not the training runtime, scheduler, security boundary or release authority. It must not generate and execute arbitrary Python, select unauthorized datasets, exceed accelerator quotas, inspect protected test labels or approve its own production artifact.

## When to use it

Use A23 when:
- Training jobs need CPU/GPU scheduling, bounded resources and recovery.
- Runs must be reproducible and traceable to data, code, environment and configuration.
- Multiple experiments or workers share accelerator resources.
- An assistant should help engineers plan and interpret experiments without controlling critical operations.

For a tiny notebook or one-off experiment, start with a deterministic script. Do not introduce queues, agents, registries or a distributed scheduler before the workload requires them.

## Architecture

~~~mermaid
flowchart TD
    A[Engineer request] --> B[Read-only intake agent]
    B --> C[Typed experiment contract]
    C --> D[Policy and dataset validator]
    D --> E[Deterministic training controller]
    E --> F[CPU/GPU worker]
    F --> G[Metrics and atomic checkpoints]
    G --> H[Validation evaluator]
    H --> I[Independent verifier]
    I --> J{Acceptance policy}
    J -->|Fail| K[Bounded retry or failure report]
    J -->|Pass| L[Frozen candidate]
    L --> M[Protected test gate]
    M --> N[Human approval]
    N --> O[Model registry]
~~~

## Component contracts

| Component | Input | Output | Authority |
|---|---|---|---|
| Intake agent | Engineer objective and approved project metadata | Proposed typed request | No shell, arbitrary code, data or accelerator authority |
| Contract validator | Typed request | Validated config or structured rejection | Enforces schema, ranges, identity and policy |
| Dataset validator | Dataset reference and split manifest | Fingerprint, schema and leakage report | Enforces dataset ACL and split immutability |
| Training controller | Validated config, queue state and budget | Scheduled, paused, resumed or cancelled job | Owns CPU/GPU quotas, deadline, concurrency and retries |
| PyTorch worker | Approved tensors, config and device lease | Metrics, checkpoint and status events | Reads only authorized training data; no registry promotion |
| Checkpoint store | Model/training state and lineage manifest | Atomic, versioned checkpoint | Integrity, retention and access control |
| Evaluator | Frozen candidate and declared split | Deterministic metrics | Validation during selection; test only after candidate freeze |
| Verifier | Artifact, config, metrics and provenance | Pass/fail evidence with reasons | Checks shapes, finite values, gradient smoke tests and lineage |
| Approval/registry gate | Verified artifact and authorized decision | Immutable registered version | Requires policy and human model-owner approval |

## Typed experiment contract

Example request; validate it using a JSON Schema or typed application model before scheduling:

~~~json
{
  "run_id": "telemetry-mlp-017",
  "dataset_version": "telemetry-v4",
  "split_manifest_id": "device-group-split-2026-11",
  "task": "binary_classification",
  "architecture": {
    "input_dim": 12,
    "hidden_units": 32,
    "activation": "relu",
    "output_dim": 1
  },
  "training": {
    "epochs": 80,
    "batch_size": 128,
    "learning_rate": 0.001,
    "seed": 17017,
    "device_policy": "approved_cuda_or_cpu"
  },
  "budget": {
    "max_wall_seconds": 900,
    "max_gpu_hours": 0.25,
    "max_concurrent_runs": 2
  }
}
~~~

Reject unknown fields, unsupported activations, invalid tensor dimensions, unapproved datasets and budgets outside the caller's entitlement. Never accept Python source, shell commands, arbitrary imports or a filesystem path from the LLM as an executable tool argument.

## Seven agent architecture patterns

| Pattern | A23 application | Trade-off and guardrail |
|---|---|---|
| Deterministic workflow | Validate → schedule → train → checkpoint → evaluate → package | Default for numerical and policy-critical operations |
| Single agent + typed tools | Read metrics, inspect checkpoint metadata, explain failures | Lowest orchestration complexity; start read-only |
| Router-specialist | Route data, performance, model-quality and deployment questions | Use only when contexts or permissions differ; provide deterministic fallback |
| Planner–executor–verifier | Plan a bounded ablation, execute fixed configurations, verify artifacts | Persist plan; cap steps, tokens, wall time and accelerator budget |
| Supervisor–worker | Run independent seeds or hyperparameter trials | Parallelism costs memory and compute; cap concurrency and keep split manifests identical |
| Event-driven | Queue jobs and emit checkpoint/completion/failure events | Requires durable events, idempotency, cancellation and dead-letter handling |
| Human-in-the-loop | Approve expensive runs, exceptional data access and production promotion | Auditable approval state, timeout and escalation policy |

Start with a deterministic workflow and one read-only assistant. Add workers only when experiments are independent and comparable.

## State, memory and checkpoint contract

Persist:
- Run ID, owner, tenant/project and authorization scope.
- Normalized configuration and configuration hash.
- Dataset fingerprint, schema version and immutable split-manifest ID.
- Feature/preprocessing version and code commit.
- Container/environment digest, Python/PyTorch/CUDA versions and hardware metadata.
- Model, optimizer, scheduler and mixed-precision scaler states.
- Epoch, global step, gradient-scaler state and training/validation history.
- CPU/CUDA RNG states, DataLoader/sampler generator state and augmentation RNG state where required.
- Checkpoint hash, selected checkpoint and resume history.
- Test-evaluation record created only after candidate freeze.
- Human approval identity, timestamp, rationale and registry artifact version.

Conversation memory is not durable training state. On restart, reconstruct from a validated checkpoint manifest and verify architecture, dataset, preprocessing, optimizer and environment compatibility.

Model weights alone are sufficient for inference but not faithful training resumption. Optimizer moments, scheduler/scaler and RNG/sampler state may be necessary to continue the same run.

## Security and trust boundaries

Threats include prompt injection in data rows/logs/notebooks, unauthorized dataset selection, compute exhaustion, test leakage, malicious checkpoints, secret exposure and arbitrary-code execution.

Controls:
1. Treat dataset content, filenames, stack traces and notebook output as untrusted data, never instructions.
2. Keep the LLM context to minimized, redacted metadata and approved summaries.
3. Enforce identity, dataset ACLs and accelerator quotas in the controller/service layer.
4. Expose typed, allow-listed tools; do not expose unrestricted shell or Python execution.
5. Keep train, validation and protected test permissions separate.
6. Verify artifact hashes and provenance; scan dependencies and model packages.
7. Load only trusted checkpoint artifacts. Prefer state_dict-based loading and safe loading options; do not deserialize arbitrary untrusted Python objects.
8. Require an authorized human owner for production promotion.
9. Redact secrets, private examples and identifiers from prompts, logs and traces.

## Reliability and recovery

- Derive idempotency keys from run ID plus normalized configuration hash.
- Make job submission and event consumption idempotent.
- Write checkpoints atomically and include a manifest/checksum.
- Validate checkpoint hash, shapes, dtype and configuration before resume.
- Bound wall time, accelerator hours, memory, batch size, workers and concurrent jobs.
- Distinguish retryable infrastructure failures from deterministic data/model failures.
- Use cancellation tokens and release orphaned accelerator allocations.
- Send poison events to a dead-letter queue with investigation metadata.
- Never silently resume a checkpoint against different data, model or preprocessing.

## Evaluation contract

Separate three kinds of quality:

1. **Implementation correctness:** tensor shape checks, finite loss/gradients/parameters, gradient-check smoke tests, checkpoint round-trip and deterministic unit tests.
2. **Model quality:** agreed metrics on validation, baseline comparison, then one protected final test evaluation after candidate freeze.
3. **Operational fitness:** examples/second, data wait, accelerator utilization, memory, latency, reproducibility, recovery and cost per successful run.

An LLM review is not an independent numerical verifier. Use deterministic checks and, where necessary, an independently implemented evaluator.

## Observability

Every trace/event should include:
- trace ID, run ID, idempotency key and parent event ID;
- dataset fingerprint, split-manifest ID, code commit and environment digest;
- configuration hash, architecture, seed, optimizer and scheduler;
- epoch/step, train/validation loss and task metrics;
- gradient norm, activation statistics and NaN/Inf counters;
- device type, accelerator ID, utilization and peak memory;
- checkpoint hash, save duration and resume count;
- queue delay, data-wait time, examples/second and total wall time;
- LLM model/prompt version, token usage, tool calls and assistant latency;
- approval and registry transition events.

Measure p50/p95 job-start and end-to-end latency, throughput, failure rate, time to recover and cost per successful experiment. Do not log raw private samples or secrets.

## Cost and latency controls

- Run smoke tests on CPU or a small accelerator before expensive sweeps.
- Profile data loading, host-to-device copies, forward, backward, optimizer and checkpointing separately.
- Bound concurrency to avoid GPU memory contention.
- Tune batch size and DataLoader workers empirically.
- Use pinned memory/non-blocking copies only when the data path benefits.
- Stop unpromising runs using declared validation criteria, never hidden test feedback.
- Cache deterministic preprocessing only with data and transform versions in the cache key.
- Use an LLM at coarse workflow boundaries for planning/explanation; never call it per training batch.
- Propagate deadlines and cancellation through queue, worker and tool layers.

## Example state machine

~~~text
REQUESTED
  -> VALIDATING
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

Every transition records actor, timestamp, reason and input/output artifact hashes.

## Reference implementation

Session 017's [PyTorch XOR lab](../../daily-lessons/day-017/lab.py) demonstrates a local deterministic workflow: seeded data, CPU/CUDA selection, DataLoader, autograd training, validation-selected checkpoints, safe state loading and a held-out test evaluation. It is a teaching reference, not a distributed production trainer.

## Required acceptance criteria for an implementation

- Typed contract validation rejects malformed or over-budget requests.
- Dataset and split manifest are immutable for a run.
- CPU and supported accelerator paths pass smoke tests.
- No non-finite loss, gradient or parameter values.
- Checkpoint round-trip and interrupted-run recovery tests pass.
- Idempotent event/job replay does not create duplicate side effects.
- Validation selection never reads protected test labels.
- Independent verifier confirms artifact lineage and metrics.
- Human approval is required for registry promotion.
- Traces expose queue delay, throughput, memory, latency and cost without leaking sensitive samples.

## References

- [PyTorch autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
