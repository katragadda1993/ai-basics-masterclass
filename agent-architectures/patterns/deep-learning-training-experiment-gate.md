# A22 — Deep-Learning Training and Release Gate

## Purpose

A22 is a production-minded architecture for using AI assistants to help plan, inspect and explain deep-learning experiments while deterministic services retain control of data access, tensor computation, training, evaluation, checkpointing and model promotion.

The first reference workload is a small feed-forward classifier. The same boundaries apply to larger CNN, RNN, transformer and multimodal training systems.

**Core principle:** an LLM may propose an experiment or summarize evidence. It must not execute arbitrary generated code, alter the protected test set, grant itself data/GPU permissions or approve its own production release.

## When to use it

Use A22 when:
- Training runs are long enough to require durable state and recovery.
- Multiple experiments need comparable configurations and immutable splits.
- Dataset access, compute budgets or model promotion require governance.
- An assistant should help engineers interpret runs without becoming the training runtime.

For a one-off notebook, use a deterministic script first. Do not introduce agents, queues or a model registry merely to train a tiny model.

## Architecture

~~~mermaid
flowchart TD
    A[Engineer objective] --> B[Intake agent]
    B --> C[Typed experiment contract]
    C --> D[Dataset and split validator]
    D --> E[Deterministic training workflow]
    E --> F[Training worker]
    F --> G[Checkpoint and metrics store]
    G --> H[Validation evaluator]
    H --> I[Independent verifier]
    I --> J{Acceptance policy}
    J -- Reject --> K[Failure report and bounded retry]
    J -- Accept --> L[Frozen candidate artifact]
    L --> M[Protected test gate]
    M --> N[Human model owner approval]
    N --> O[Versioned model registry]
~~~

## Component contracts

| Component | Input | Output | Authority |
|---|---|---|---|
| Intake agent | Engineer request and allowed project context | Proposed typed experiment request | No dataset, shell or release authority |
| Contract validator | Typed request | Validated configuration or structured rejection | Enforces schema, ranges and policy |
| Dataset validator | Dataset reference and split manifest | Dataset fingerprint and validation report | Enforces data scope and split immutability |
| Training runtime | Validated tensors/configuration | Checkpoints, metrics and training status | Can read only authorized training data |
| Experiment controller | Run state and budget | Scheduled, paused, resumed or cancelled run | Enforces deadline, accelerator and retry budgets |
| Evaluator | Frozen candidate and declared split | Deterministic metrics | Validation during selection; test only after freeze |
| Verifier | Run metadata, artifact and metrics | Pass/fail evidence with reasons | Checks lineage, shapes, finite values and policy |
| Registry gate | Verified candidate and approval | Immutable registered artifact | Writes only after policy and approval |

Each component should have a narrow API and an explicit owner. The training runtime must not call the LLM to decide tensor shapes or loss calculations.

## Typed experiment contract

Example request (validate with a JSON Schema or equivalent typed model before execution):

~~~json
{
  "run_id": "mlp-xor-001",
  "dataset_version": "xor-synthetic-v1",
  "split_manifest_id": "split-2026-01",
  "task": "binary_classification",
  "architecture": {
    "input_dim": 2,
    "hidden_units": 8,
    "activation": "tanh",
    "output_dim": 1
  },
  "training": {
    "max_epochs": 1500,
    "batch_size": 32,
    "learning_rate": 0.08,
    "seed": 99
  },
  "budget": {
    "max_wall_seconds": 300,
    "max_gpu_hours": 0
  }
}
~~~

Reject unknown task/activation names, non-positive dimensions, unreasonable batch sizes, excessive epochs, unapproved datasets and budgets above the caller's entitlement. Do not accept a code string, shell command or arbitrary import as an experiment tool argument.

## Seven architecture patterns

| Pattern | A22 application | Trade-off / guardrail |
|---|---|---|
| Deterministic workflow | Validate → train → checkpoint → evaluate → package | Most reproducible; use for all numerical and policy-critical steps |
| Single agent + typed tools | Read metrics, compare runs, explain failures | Lowest orchestration complexity; tools should be read-only unless explicitly authorized |
| Router-specialist | Route to data, training, performance or evaluation specialist | Separate contexts and expertise; deterministic eligibility checks and safe fallback required |
| Planner–executor–verifier | Plan bounded ablations, execute, verify outcomes | Useful for multi-step research; persist plans and cap steps, tokens, wall time and cost |
| Supervisor–worker | Run independent seeds or hyperparameter trials in parallel | Better wall-clock time; cap concurrency and compare on identical splits/budgets |
| Event-driven | Queue jobs, checkpoint events, completion/failure notifications | Decoupled and scalable; requires durable events, idempotency and dead-letter handling |
| Human-in-the-loop | Approve data access, expensive runs and production promotion | Strong governance; define reviewer timeout, escalation and auditable decision state |

Start with a deterministic workflow plus one read-only assistant. Add parallel workers only when experiments are independent and the evaluation protocol is identical.

## State, memory and checkpointing

Persist an experiment record with:
- Run ID, owner, tenant/project and authorization scope.
- Dataset fingerprint, schema version and immutable split-manifest ID.
- Code commit, environment/container digest and framework versions.
- Architecture, initialization seed, loss, optimizer and complete configuration.
- Epoch, step, random-generator state and optimizer state.
- Training/validation history, checkpoint hashes and selected checkpoint.
- Test-evaluation record created only after candidate freeze.
- Approval identity, timestamp, rationale and registry artifact version.

Conversation history is not the source of truth. On restart, load a validated durable state record and verify checkpoint/configuration compatibility.

Checkpoint model weights alone are insufficient for exact training resumption. Where supported, save optimizer state, scheduler state, epoch/step and random-number-generator state as well.

## Security and trust boundaries

Threats:
- Prompt injection embedded in dataset rows, filenames, notebooks, stack traces or experiment logs.
- A model proposing an unapproved dataset or excessive compute request.
- Leakage of secrets, private samples or customer identifiers into prompts and traces.
- Training/test contamination or post-hoc split changes.
- Untrusted checkpoint or model artifacts.
- Tool abuse through shell, arbitrary Python or registry-write access.

Controls:
1. Treat all dataset content and logs as untrusted data, never as instructions.
2. Keep LLM context to minimized, redacted metadata and approved summaries.
3. Enforce identity, dataset ACLs and GPU quotas in the service layer.
4. Use typed allow-listed tools; do not expose unrestricted shell or Python execution.
5. Keep train, validation and protected test permissions separate.
6. Hash/sign artifacts and record provenance; scan dependencies and model packages.
7. Require an authorized human owner for production promotion.
8. Redact secrets and sensitive sample content from logs and traces.

## Reliability and recovery

- Use an idempotency key derived from run ID plus normalized configuration hash.
- Make job submission and event consumption idempotent.
- Checkpoint at a configured step/epoch interval.
- Validate checkpoint hash, shapes, dtype and configuration before resume.
- Bound retries, total wall time, GPU hours, memory and concurrent workers.
- Distinguish retryable infrastructure failures from deterministic data/model failures.
- Use cancellation tokens and clean up orphaned accelerator allocations.
- Send poison events to a dead-letter queue with enough metadata for investigation.
- Never silently resume a checkpoint with a different dataset or architecture.

## Evaluation contract

Separate three questions:

1. **Implementation correctness:** gradient checks, tensor shapes, finite values and deterministic unit tests.
2. **Model quality:** task metrics on validation and then one frozen final test evaluation.
3. **Operational fitness:** throughput, memory, latency, reproducibility, robustness and deployment constraints.

Example acceptance gates:
- All input/output tensor shapes match the declared architecture.
- No NaN/Inf values in loss, gradients or parameters.
- Gradient-check tolerance passes on a small reference batch.
- Training loss improves on a known smoke-test dataset.
- Validation metrics beat an agreed baseline under a fixed budget.
- Test data is not used for architecture/threshold/hyperparameter selection.
- Artifact lineage and approval record are complete.

A second LLM review is not an independent numerical verifier. Prefer deterministic checks and, where needed, an independently implemented evaluator.

## Observability

Every event should include:
- trace ID, run ID, idempotency key and parent event ID;
- dataset fingerprint, split-manifest ID and code/environment version;
- model architecture, seed, optimizer and configuration hash;
- epoch/step, training and validation loss, task metrics;
- gradient norm, activation statistics, NaN/Inf counters;
- checkpoint hash and resume count;
- queue delay, training duration, CPU/GPU utilization and peak memory;
- token usage, tool calls and LLM latency for assistant actions;
- approval and registry transition events.

Measure p50/p95 job-start latency, examples/second, GPU utilization, cost per successful experiment, failure rate and time to recover. Do not log raw private samples or secrets.

## Cost and latency policy

- Run smoke tests on CPU or a small accelerator before expensive sweeps.
- Profile data loading, host-to-device copies and compute separately.
- Bound concurrent workers to avoid GPU memory contention.
- Stop unpromising runs using declared validation criteria, not hidden test feedback.
- Cache deterministic preprocessing only with a key containing data and transform versions.
- Use an LLM only for planning/explanation where it adds value; never call it per training batch.
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

Session 016's [NumPy MLP lab](../../daily-lessons/day-016/lab.py) is a deliberately small deterministic training runtime. It demonstrates:
- explicit forward and backward passes;
- train-only preprocessing;
- gradient checking with finite differences;
- validation-based checkpoint selection;
- a final held-out test evaluation;
- finite-value and performance assertions.

It is a teaching reference, not a production training platform. A production implementation should add durable storage, framework-native checkpointing, job orchestration, security controls, monitoring and release approvals.

## Failure-mode test matrix

| Test | Expected behavior |
|---|---|
| Dataset fingerprint changes on resume | Reject resume; require a new run |
| Test split manifest is modified | Reject and emit audit event |
| Unknown activation or task | Typed contract rejection |
| NaN appears in loss/gradient | Stop run, preserve diagnostic checkpoint/log |
| Worker event is delivered twice | Idempotency prevents duplicate training side effect |
| GPU worker is interrupted | Resume only from compatible validated checkpoint |
| Budget is exhausted | Cancel safely and record partial status |
| Prompt injection appears in a sample/log | Treat as data; no tool/permission change |
| Verifier fails artifact hash | Block test and registry gates |
| Human approval expires | Do not promote; move to expired/escalation state |

## Rollout checklist

- [ ] Typed request/response schemas and versioning
- [ ] Dataset ACLs and immutable split manifests
- [ ] Deterministic smoke-test dataset and gradient checks
- [ ] Idempotent run/job/event IDs
- [ ] Checkpoint and resume compatibility tests
- [ ] CPU/GPU budgets, deadlines and cancellation
- [ ] Redacted traces and audit events
- [ ] Independent validation/test evaluator
- [ ] Artifact signing, provenance and registry controls
- [ ] Human approval for production release
- [ ] Cost, latency, throughput and failure-rate dashboards

## Further reading

- PyTorch autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html
- PyTorch checkpointing and saving models: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework

