# Session 017 — PyTorch Training in Practice: Autograd, Datasets, Batching, GPU and Checkpointing

**Phase:** 4 — Deep Learning  
**Duration:** 60 minutes  
**Prerequisite:** Session 016 — the NumPy MLP, forward propagation, backpropagation and gradient checking.

## Why this lesson follows Session 016

Session 016 built a small multilayer perceptron from NumPy so every matrix operation and derivative was visible. This lesson ports the same conceptual task into PyTorch and focuses on the engineering runtime around a model: tensors, automatic differentiation, datasets, mini-batches, device placement, training/evaluation modes, checkpoint recovery and profiling.

The lab retains XOR. A linear model is the control experiment; a nonlinear MLP should learn the quadrant decision boundary. This keeps the data and task simple while changing the implementation from hand-written gradients to PyTorch autograd.

## Learning objectives

By the end of this session, you will be able to:

1. Explain PyTorch tensors, shape, dtype and device.
2. Describe how autograd records a dynamic computation graph and calculates gradients.
3. Implement the canonical training sequence: zero gradients → forward → loss → backward → optimizer step.
4. Explain why gradients accumulate and why model training mode differs from evaluation mode.
5. Build a TensorDataset and DataLoader for mini-batch training.
6. Move models and batches safely between CPU and CUDA devices.
7. Use inference mode for evaluation without building an autograd graph.
8. Save and restore model, optimizer, epoch and random-generator state.
9. Measure throughput and understand why GPU timing needs synchronization.
10. Design a governed, observable training workflow with bounded agent assistance.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Connect the NumPy MLP to PyTorch |
| 5–15 min | Tensors, devices and autograd |
| 15–25 min | Dataset, DataLoader and the training loop |
| 25–35 min | CPU/GPU execution, evaluation and profiling |
| 35–42 min | Checkpoints, recovery and reproducibility |
| 42–55 min | Run and inspect the PyTorch XOR lab |
| 55–60 min | Production training architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Tensor | Multi-dimensional array with dtype, shape and device |
| Device | CPU or accelerator location where a tensor resides |
| Autograd | PyTorch automatic differentiation engine |
| Computation graph | Operations and dependencies recorded for gradient calculation |
| Leaf tensor | A tensor such as a trainable parameter whose gradient is retained by default |
| requires_grad | Indicates whether autograd should track operations for a tensor |
| .grad | Accumulated gradient stored on a leaf tensor after backward |
| backward | Propagates derivatives through the recorded graph |
| Optimizer | Applies parameter updates using gradients |
| Dataset | Abstraction for indexing examples and labels |
| DataLoader | Batches, optionally shuffles and loads dataset examples |
| Epoch | One complete pass over the training dataset |
| Mini-batch | A subset of examples used for one optimizer update |
| Checkpoint | Persisted model/training state used for recovery or evaluation |
| state_dict | Mapping of parameter/buffer names to tensor values |
| Inference mode | Context that disables autograd bookkeeping for inference-only computation |
| Throughput | Examples or tokens processed per unit time |

## 1. From the NumPy network to PyTorch

The Session 016 network was:

\[
X\rightarrow Linear(2,8)\rightarrow tanh\rightarrow Linear(8,1)\rightarrow sigmoid
\]

The PyTorch lab uses a slightly wider hidden layer (16 units) to make training robust across deterministic seeds. The output layer returns logits:

\[
z_1=XW_1+b_1,\qquad a_1=\tanh(z_1)
\]

\[
z_2=a_1W_2+b_2
\]

BCEWithLogitsLoss consumes z2 directly. It combines sigmoid and binary cross-entropy in a numerically stable implementation. Apply sigmoid only when a probability is needed for reporting or a decision rule.

~~~python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 1),  # one logit per example
)
criterion = nn.BCEWithLogitsLoss()
~~~

The model is a Python object containing registered parameters. nn.Linear initializes its weights and bias. The optimizer discovers parameters through model.parameters().

## 2. Tensors: shape, dtype and device

A tensor has at least three properties to inspect:

- Shape: dimensions, for example [64, 2] for 64 two-feature examples.
- Dtype: for example torch.float32 for model inputs and parameters.
- Device: for example cpu or cuda:0.

For a batch of 64 XOR examples:

| Value | Shape | Meaning |
|---|---|---|
| xb | [64, 2] | 64 examples, 2 input features |
| yb | [64, 1] | 64 binary targets |
| First linear output | [64, 16] | 16 hidden activations per example |
| Final logits | [64, 1] | One raw score per example |

Shape mismatches are a common source of silent broadcasting bugs. Assert expected dimensions at model boundaries, especially when combining model outputs and labels.

A model and its inputs must be on compatible devices:

~~~python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
xb = xb.to(device)
yb = yb.to(device)
~~~

Moving a model to CUDA does not automatically move later input batches. The data pipeline must place each batch on the selected device.

## 3. Autograd: how PyTorch calculates gradients

When autograd is enabled, PyTorch records the operations needed to differentiate a result with respect to tensors that require gradients. The graph is built dynamically as the forward code runs.

~~~python
x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x
loss = y
loss.backward()
print(x.grad)  # dy/dx = 2*x + 3 = 7 at x=2
~~~

Conceptually:

~~~text
x (requires_grad)
       |
    multiply ----
       |         \
       +          add ----> loss
       |                    |
       +---------------- backward()
                            |
                       x.grad = 7
~~~

Important rules:

1. Gradients accumulate in .grad by default. They are not automatically cleared after an optimizer step.
2. Clear gradients before the next update, commonly with optimizer.zero_grad(set_to_none=True).
3. loss.backward() calculates gradients; it does not update parameters.
4. optimizer.step() performs the update using the gradients currently stored on parameters.
5. The graph is normally freed after backward unless retain_graph=True is requested. Retaining graphs unnecessarily can increase memory use.
6. Use detach() when a tensor must be separated from its history, but understand that doing so stops gradients through that path.

### The canonical training step

~~~python
model.train()
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
~~~

The sequence is deliberate:

- zero_grad: remove gradients from the previous mini-batch.
- forward: calculate predictions.
- criterion: calculate a scalar loss.
- backward: calculate parameter gradients.
- step: update parameters.

Gradient clipping is optional. It bounds a gradient norm; it is not a replacement for fixing exploding gradients or a bad learning rate.

## 4. Dataset and DataLoader

Dataset defines how one example is retrieved. DataLoader builds batches and can shuffle the training order.

~~~python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
~~~

Why mini-batches?

- Full-batch training may require too much memory and produces fewer optimizer updates per epoch.
- Very small batches add overhead and produce noisy gradients.
- Larger batches can improve accelerator utilization but use more memory and may change optimization behavior.

The right batch size depends on model size, data pipeline, accelerator memory and convergence behavior. Benchmark rather than assuming the largest batch is best.

For GPU pipelines, pin_memory=True can help host-to-device transfer in appropriate workloads. Pair it with .to(device, non_blocking=True) where applicable. More DataLoader workers can improve throughput for expensive CPU preprocessing, but worker startup and inter-process overhead can make them slower for small in-memory datasets.

For reproducibility, seed the DataLoader generator explicitly. If using multiple workers or stochastic augmentations, also seed worker-side Python/NumPy generators and persist sampler state when exact recovery is required.

## 5. Training mode, evaluation mode and inference mode

These are distinct controls:

| Mechanism | What it changes |
|---|---|
| model.train() | Sets modules to training behavior; affects layers such as Dropout and BatchNorm |
| model.eval() | Sets modules to evaluation behavior; Dropout is disabled and BatchNorm uses stored statistics |
| torch.no_grad() | Disables gradient recording in a context |
| torch.inference_mode() | Disables autograd bookkeeping more aggressively for inference-only computation |

Calling model.eval() does not disable autograd by itself. For evaluation:

~~~python
model.eval()
with torch.inference_mode():
    logits = model(xb)
~~~

For a simple MLP without Dropout or BatchNorm, train() and eval() may produce the same numerical output, but retain the correct mode transitions in reusable code.

## 6. CPU and GPU execution

A GPU can accelerate large batches of parallel tensor operations. It is not automatically faster for every model or workload.

### Device-aware training

~~~python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for xb, yb in train_loader:
    xb = xb.to(device, non_blocking=(device.type == "cuda"))
    yb = yb.to(device, non_blocking=(device.type == "cuda"))
    ...
~~~

Consider:

- Compute size: a tiny MLP may be faster on CPU because kernel launch and transfer overhead dominate.
- Transfer cost: repeated CPU↔GPU copies can erase acceleration benefits.
- Memory: parameters, gradients, optimizer states, activations and batches all consume accelerator memory.
- Batch size: larger batches may improve utilization until memory or convergence constraints intervene.
- Precision: mixed precision can improve throughput on supported accelerators, but requires numerical checks and often gradient scaling for FP16. BF16 has a wider exponent range on supported hardware.
- Reproducibility: some accelerator operations may be nondeterministic or unavailable in deterministic mode.

### Timing GPU work correctly

CUDA operations are often asynchronous from the CPU's perspective. A naive wall-clock timer can measure launch time rather than completed device work. Warm up the model and synchronize before/after a timed region:

~~~python
for _ in range(10):
    _ = model(xb)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    _ = model(xb)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
~~~

Measure the full pipeline: data wait, transfer, forward, backward, optimizer and checkpointing. The slowest stage is the bottleneck worth optimizing.

## 7. Checkpoints and recovery

Saving only model weights is sufficient for inference, but not for faithful training resumption.

A training checkpoint should normally include:

- Model state_dict.
- Optimizer state_dict (momentum/Adam moments are part of training state).
- Scheduler and mixed-precision scaler state, if used.
- Epoch and global step.
- CPU and accelerator RNG states.
- DataLoader/sampler generator state, and worker/augmentation state where needed.
- Model and training configuration.
- Dataset fingerprint, split-manifest ID and feature/preprocessing version.
- Code commit, dependency/environment versions and checkpoint hash.

A simplified example:

~~~python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "torch_rng_state": torch.get_rng_state(),
}, "checkpoint.pt")
~~~

Resume by reconstructing the same model and optimizer, loading their state dictionaries, restoring the RNG/sampler state and continuing from the next epoch. Verify that architecture, dataset fingerprint, preprocessing, optimizer and code configuration are compatible before resuming.

The lab uses weights_only=True to load a checkpoint containing tensors and primitive values. Only load checkpoint files from trusted sources; artifact provenance and hashes still matter. A model state_dict is generally preferable to serializing an arbitrary Python model object.

For production, write checkpoints atomically (temporary file then rename/object-store commit), keep a manifest and checksum, retain a bounded number of versions and test recovery. A partially written checkpoint must never be treated as valid.

## 8. Reproducibility and debugging

The lab seeds Python, NumPy, PyTorch and DataLoader shuffling. It requests deterministic PyTorch algorithms where available. This improves repeatability but does not promise bit-for-bit identity across PyTorch versions, devices, hardware or every operation.

Useful diagnostics:

- Print tensor shapes, dtypes and devices at model boundaries.
- Check torch.isfinite(loss) and inspect gradient norms.
- Confirm parameters change after optimizer.step().
- Compare train and validation metrics for overfitting.
- Use a tiny batch to overfit intentionally as a pipeline smoke test.
- Use torch.autograd.gradcheck on small double-precision functions when validating custom differentiable code.
- Profile before optimizing; do not assume the GPU is the bottleneck.

## 9. Hands-on lab: PyTorch XOR classifier

### Dataset and model

The lab creates deterministic synthetic XOR-like quadrant data:

\[
y=\mathbb{1}[x_1x_2>0]
\]

| Split | Samples |
|---|---:|
| Training | 1,200 |
| Validation | 300 |
| Test | 300 |

Two models use the same split:

- Linear baseline: Linear(2, 1).
- MLP: Linear(2, 16) → Tanh → Linear(16, 1).

The MLP is trained with BCEWithLogitsLoss and AdamW. The best checkpoint is selected using validation loss, restored, and then evaluated once on the held-out test split.

### Reproduced run

Reference environment: PyTorch 2.10.0 CPU build. Timings vary with hardware, runtime load and PyTorch build; use them only as a local smoke-test observation.

| Model | Test accuracy | Test F1 | Test loss |
|---|---:|---:|---:|
| Linear baseline | 54.67% | 0.5108 | 0.69338 |
| MLP | 99.67% | 0.9966 | 0.03067 |

The MLP's nonlinear hidden layer learns the XOR-like decision boundary; the linear baseline cannot represent it. The lab also verifies a small autograd derivative using finite-difference gradcheck, round-trips a checkpoint, checks finite losses and asserts that the MLP beats the linear control by a margin.

**Interpretation:** this is a synthetic teaching task with a clean nonlinear boundary. It demonstrates framework mechanics, not expected performance on real silicon telemetry. Real diagnostic data requires leakage-safe splits, realistic operating modes, imbalance-aware metrics and independent validation.

### Run it

~~~bash
cd daily-lessons/day-017
python -m pip install -r requirements.txt
python lab.py
~~~

The script automatically uses CUDA when available and otherwise CPU. It prints the actual device, framework version, metrics, checkpoint round-trip result and runtime observed in the current environment.

### Exercises

1. Change HIDDEN_UNITS from 16 to 2, 4 and 32. Compare convergence and validation behavior.
2. Change the batch size to 16, 64 and 256. Record training time, examples/second and test quality.
3. Replace Tanh with ReLU. If needed, change initialization and learning rate; explain why an activation change can alter optimization.
4. Interrupt a longer run after an epoch boundary, restore model + optimizer + RNG + DataLoader generator state, and compare with an uninterrupted run.
5. Add a torch.profiler trace. Separate data loading, host-to-device transfer, forward, backward and optimizer time.
6. If CUDA is available, compare CPU and GPU with warm-up and synchronization. Do not claim a speedup unless the complete timed workload is faster.

## 10. Production pattern A23 — PyTorch Training Runtime and GPU-Aware Execution Gate

### Architecture

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

### Component boundaries

| Component | Responsibility | Authority boundary |
|---|---|---|
| Intake agent | Convert the engineer's goal into a proposed request | No arbitrary code, shell or dataset authority |
| Contract validator | Validate architecture, shapes, dataset and budget | Reject unknown fields and out-of-policy values |
| Training controller | Schedule, pause, cancel and resume runs | Enforce CPU/GPU quota, deadline and concurrency |
| PyTorch worker | Execute versioned tensor code | Access only the approved training data and assigned device |
| Checkpoint store | Persist recoverable state and artifact hashes | Atomic writes, integrity checks and retention policy |
| Evaluator | Compute declared metrics on authorized splits | Validation during selection; test only after candidate freeze |
| Verifier | Check finite values, shapes, lineage and acceptance rules | Independent deterministic checks, not a second LLM opinion |
| Approval/registry gate | Record approval and promote an artifact | Human owner and policy required for production promotion |

### How the seven agent patterns apply

| Pattern | A23 application | Trade-off / control |
|---|---|---|
| Deterministic workflow | Validate → train → checkpoint → evaluate → package | Default for all numeric and policy-critical operations |
| Single agent + typed tools | Read metrics, inspect checkpoint metadata, explain a failure | Low complexity; start read-only |
| Router-specialist | Route data, performance, model-quality and deployment questions | Use only when contexts/permissions differ; fallback to deterministic docs/runbook |
| Planner–executor–verifier | Plan a bounded ablation, execute fixed configs, verify artifacts | Persist plan; cap steps, tokens, wall time and GPU budget |
| Supervisor–worker | Run independent seeds or hyperparameter experiments in parallel | Parallelism costs memory/compute; cap concurrency and use identical split manifests |
| Event-driven | Queue jobs and emit checkpoint/completion/failure events | Durable events, idempotency keys, bounded retries and dead-letter handling |
| Human-in-the-loop | Approve expensive runs, exceptional data access and model promotion | Auditable approval state, timeout and escalation policy |

### Typed tool contract example

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

Validate dimensions, allowed activation names, dataset ACLs and resource limits in ordinary application code. Do not accept a Python source string or arbitrary module name from the model.

### State, memory and recovery

Persist run ID, owner/tenant, configuration hash, dataset fingerprint, split manifest, code commit, container digest, PyTorch/CUDA versions, model/optimizer/scheduler/scaler states, epoch/global step, CPU/CUDA RNG states, sampler state, metrics, checkpoint hashes and approval record.

Conversation memory is not durable training state. The worker must recover from the validated checkpoint manifest, not from a prior chat transcript.

### Security and reliability

- Treat dataset rows, filenames, notebook output and stack traces as untrusted data; they may contain prompt injection.
- Never expose arbitrary shell, Python, filesystem or unrestricted cloud credentials to the agent.
- Enforce dataset permissions and accelerator quotas in the controller, not in prompts.
- Keep training and test permissions separate; the LLM cannot request hidden test labels.
- Validate artifact hashes and checkpoint compatibility before loading.
- Use idempotency keys for job submission and event processing.
- Write checkpoints atomically and retain a last-known-good checkpoint.
- Distinguish transient infrastructure failures from deterministic code/data errors; retry only bounded transient failures.
- Propagate cancellation and clean up abandoned accelerator allocations.

### Evaluation and observability

Correctness gates: tensor shape contract, finite loss/gradients/parameters, gradient-check smoke test, deterministic tiny-batch overfit, checkpoint round-trip and artifact lineage.

Model gates: validation metrics and agreed baselines, followed by one protected test evaluation after freezing the candidate.

Operational metrics: queue delay, examples/second, data-wait time, accelerator utilization, peak memory, checkpoint duration, resume count, failure rate, p50/p95 end-to-end latency, cost per successful run and LLM token/tool cost.

Redact private examples, credentials and sensitive log content. Every trace should carry run ID, config hash, dataset/split IDs, model version, device, checkpoint hash and parent event ID.

### Cost and latency

Use CPU for tiny smoke tests and only request a GPU when measured throughput justifies it. Profile the complete input-to-checkpoint pipeline. Bound batch size, workers, memory, concurrency, wall time and accelerator hours. Avoid LLM calls inside the batch loop; use the assistant for setup, diagnosis and explanation at coarse workflow boundaries.

## 11. Five review questions

1. What is the difference between loss.backward() and optimizer.step()?
2. Why must gradients be cleared before each independent mini-batch update?
3. Why is model.eval() not the same as torch.inference_mode()?
4. Why does a resumable training checkpoint need optimizer and RNG/sampler state in addition to model weights?
5. Why should an LLM propose and explain experiments while a deterministic controller owns device quotas, training, evaluation and model promotion?

## References

1. [PyTorch — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
2. [PyTorch — Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
3. [PyTorch — Optimization loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
4. [PyTorch — Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
5. [PyTorch — Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
6. [PyTorch — Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
7. [PyTorch — CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
8. [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## Next session — Session 018

**Convolutional Neural Networks: Convolution, Filters, Stride, Padding, Pooling and Feature Maps**

We will move from dense layers to spatially structured data, understand how filters learn local patterns and build a small CNN. The next step will connect convolution to hardware acceleration, memory movement and GPU execution.
