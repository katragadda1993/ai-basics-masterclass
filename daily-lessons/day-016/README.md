# Session 016 — Neural Networks from First Principles: Perceptrons, MLP, Forward Propagation and Backpropagation

**Phase:** 4 — Deep Learning  
**Duration:** 60 minutes  
**Prerequisites:** Sessions 003–007 for vectors, matrices, derivatives, chain rule, probability, cross-entropy and optimization; Sessions 008–009 for supervised learning and classification.

## Why this lesson follows time-series ML

Sessions 008–015 built a practical classical-ML toolkit: problem formulation, regression, trees, distance/probability/margin classifiers, clustering, PCA, anomaly detection and temporal forecasting. We now begin deep learning.

The key shift is that a neural network learns intermediate representations. Instead of manually writing every nonlinear decision rule, we compose trainable layers and use gradients to adjust their parameters.

The lab uses a small synthetic XOR problem. It is intentionally simple enough to inspect every matrix operation, but nonlinear enough to show why a single linear classifier is insufficient.

## Learning objectives

By the end of this session, you will be able to:

1. Explain a perceptron as a weighted sum followed by an activation.
2. Distinguish a linear model from a multilayer perceptron (MLP).
3. Calculate tensor shapes in a dense neural network.
4. Perform forward propagation through a hidden layer.
5. Explain logits, sigmoid and binary cross-entropy.
6. Derive the backpropagation equations using the chain rule.
7. Verify analytical gradients with finite differences.
8. Train and evaluate a tiny MLP using NumPy.
9. Identify initialization, activation, overfitting and numerical-stability concerns.
10. Design a reproducible, observable deep-learning experiment workflow.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | From classical ML to learned representations |
| 5–13 min | Perceptron, weights, bias and activations |
| 13–22 min | MLP architecture and forward propagation |
| 22–35 min | Loss and backpropagation from the chain rule |
| 35–42 min | Initialization, training loop and common failure modes |
| 42–55 min | NumPy XOR lab and gradient check |
| 55–60 min | Production training-agent architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Neuron / unit | A weighted sum, bias and activation |
| Weight | Trainable coefficient controlling an input connection |
| Bias | Trainable offset added before activation |
| Logit | Raw model score before a probability transform |
| Activation | Nonlinear function applied to a layer's pre-activation |
| Perceptron | A single trainable linear threshold unit |
| MLP | Feed-forward network with one or more hidden layers |
| Forward propagation | Computing activations and predictions from inputs |
| Loss | Scalar objective measuring prediction error |
| Backpropagation | Efficient chain-rule calculation of parameter gradients |
| Gradient | Partial derivatives of the loss with respect to parameters |
| Epoch | One pass over the training dataset |
| Mini-batch | Small subset used for one parameter update |
| Initialization | Initial values assigned to trainable parameters |
| Checkpoint | Saved model/training state used for recovery or evaluation |
| Inference | Using a trained model to produce predictions |

## 1. A neuron is a mathematical function

A simplified biological analogy can be motivating, but an artificial neuron is not a realistic model of a biological neuron. It is a small mathematical computation.

For input vector \(x\), weights \(w\), bias \(b\), and activation \(\phi\):

\[
z=w^Tx+b
\]

\[
a=\phi(z)
\]

The weighted sum \(z\) is the pre-activation or logit. The activation transforms it.

A two-input example:

\[
z=2x_1-0.5x_2+0.1
\]

For \(x_1=0.4\), \(x_2=0.8\):

\[
z=2(0.4)-0.5(0.8)+0.1=0.5
\]

With a sigmoid activation:

\[
\sigma(z)=\frac{1}{1+e^{-z}}\approx0.6225
\]

The neuron has not discovered meaning. It has applied a parameterized function whose parameters can be learned from examples.

## 2. Activations introduce nonlinearity

Without nonlinear activations, stacking dense layers does not create a more expressive function class: a composition of linear/affine transformations is still affine.

Common activations:

| Activation | Definition | Typical use / caveat |
|---|---|---|
| Sigmoid | \(1/(1+e^{-z})\) | Binary output probability; can saturate in hidden layers |
| Tanh | \((e^z-e^{-z})/(e^z+e^{-z})\) | Zero-centred hidden activation; can saturate |
| ReLU | \(\max(0,z)\) | Simple hidden activation; can produce inactive units |
| GELU | \(z\Phi(z)\) | Smooth activation used in many transformer blocks |
| Softmax | \(e^{z_i}/\sum_j e^{z_j}\) | Normalized multiclass output distribution |

For a binary classifier, a sigmoid output can be interpreted as an estimated positive-class probability when training and calibration are appropriate. A threshold such as 0.5 turns that score into a class decision; the threshold is a separate policy choice.

For numerical stability, production libraries often accept logits directly. For example, PyTorch's BCEWithLogitsLoss combines sigmoid and binary cross-entropy using a stable formulation. Do not manually apply sigmoid before passing logits to that loss.

## 3. A perceptron versus an MLP

A single perceptron has a linear decision boundary:

\[
w^Tx+b=0
\]

It can separate linearly separable classes. It cannot represent XOR: the two diagonal positive/negative regions cannot be separated by one straight line.

An MLP inserts a hidden layer:

\[
x \rightarrow (W_1,b_1) \rightarrow \phi
\rightarrow (W_2,b_2) \rightarrow \text{output}
\]

The hidden units transform the input into a learned representation where a simple output layer can solve a more complex problem.

~~~text
Input layer        Hidden layer        Output
 x1  ───────────┐
                ├──> h1 ───┐
 x2  ───────────┤          ├──> logit ──> probability
                ├──> h2 ───┤
                ├──> ...  │
                └──> hH ──┘
~~~ 

For a batch of \(N\) examples, input dimension \(D\), hidden width \(H\), and one binary output:

- \(X\): \(N\times D\)
- \(W_1\): \(D\times H\)
- \(b_1\): \(1\times H\)
- \(A_1\): \(N\times H\)
- \(W_2\): \(H\times1\)
- \(b_2\): \(1\times1\)
- Output: \(N\times1\)

Shape checking catches many neural-network implementation bugs before training.

## 4. Forward propagation

The lab uses a 2–8–1 MLP:

\[
Z_1=XW_1+b_1
\]

\[
A_1=\tanh(Z_1)
\]

\[
Z_2=A_1W_2+b_2
\]

\[
P=\sigma(Z_2)
\]

Each row in \(P\) is the predicted probability for one example.

The forward pass is deterministic for fixed inputs and parameters. Training changes the parameters; inference normally does not.

## 5. Binary cross-entropy loss

For target \(y\in\{0,1\}\) and predicted probability \(p\):

\[
L=-\left[y\log(p)+(1-y)\log(1-p)\right]
\]

For a batch, average the per-example losses.

- If \(y=1\), the loss is \(-\log(p)\).
- If \(y=0\), the loss is \(-\log(1-p)\).
- Confident, incorrect predictions receive large penalties.

The derivative simplifies when sigmoid is paired with binary cross-entropy:

\[
\frac{\partial L}{\partial z_2}=\frac{p-y}{N}
\]

This simplification is one reason the pairing is common.

## 6. Backpropagation: apply the chain rule backward

Backpropagation is not a separate learning rule. It is an efficient way to calculate derivatives of the loss with respect to every parameter by reusing intermediate derivatives.

For the output layer:

\[
\delta_2=\frac{P-Y}{N}
\]

\[
\frac{\partial L}{\partial W_2}=A_1^T\delta_2
\]

\[
\frac{\partial L}{\partial b_2}=\sum_{rows}\delta_2
\]

Propagate the gradient into the hidden layer:

\[
\delta_1=(\delta_2W_2^T)\odot(1-A_1^2)
\]

The factor \(1-A_1^2\) is the derivative of tanh because \(A_1=\tanh(Z_1)\).

Then:

\[
\frac{\partial L}{\partial W_1}=X^T\delta_1
\]

\[
\frac{\partial L}{\partial b_1}=\sum_{rows}\delta_1
\]

A gradient-descent update is:

\[
W\leftarrow W-\eta\frac{\partial L}{\partial W}
\]

where \(\eta\) is the learning rate. Session 007 covered optimizer variants; today's focus is how the network produces the gradients those optimizers consume.

### Why the chain rule matters

For a composition \(L(a(z(w)))\):

\[
\frac{dL}{dw}=\frac{dL}{da}\frac{da}{dz}\frac{dz}{dw}
\]

A deep network repeats this principle across many layers. The chain rule lets us calculate all parameter gradients without separately perturbing every parameter.

## 7. Gradient checking

A backpropagation implementation can look plausible and still contain a transposed matrix or incorrect derivative.

For a parameter \(\theta_i\), approximate its derivative using central finite differences:

\[
g_i^{numeric}\approx
\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}
\]

Compare this with the analytical gradient. Use a small test batch and a small network. Do not use finite differences for production training: it is far more expensive than backpropagation.

A useful relative error is:

\[
r_i=\frac{|g_i^{numeric}-g_i^{analytic}|}
{\max(\varepsilon,|g_i^{numeric}|+|g_i^{analytic}|)}
\]

The lab checks one element from each parameter tensor. A more thorough implementation checks multiple randomly selected elements.

## 8. Initialization and common training failures

### Initialization

If all hidden units start with identical weights, they receive identical gradients and learn the same feature. Random initialization breaks this symmetry.

Common starting points:

- Xavier/Glorot initialization is often suitable for tanh-like activations.
- He initialization is often suitable for ReLU-like activations.

These are starting heuristics, not guarantees. Architecture, normalization, activation and optimizer all matter.

### Failure modes

| Symptom | Possible cause | Investigation |
|---|---|---|
| Loss does not decrease | Incorrect gradients, learning rate, labels or data | Gradient check; inspect a tiny batch |
| NaN or infinity | Unstable loss, invalid input or exploding activations | Check finite values and use stable logits-based loss |
| Training good, validation poor | Overfitting or leakage | Simplify model, regularize, audit split |
| All predictions near 0.5 | Under-training, saturated/poor initialization or weak signal | Inspect logits, gradients and baseline |
| Dead ReLU units | Units remain on the inactive side | Inspect activation statistics and initialization |
| Very slow training | Poor batching, Python loops or unsuitable device | Vectorize and profile before adding hardware |

Track both training and validation loss. Do not select a model using the final test set.

## 9. Hands-on lab: solve XOR with NumPy

The runnable lab is in [lab.py](./lab.py) and uses NumPy only.

### Dataset

The synthetic dataset has four Gaussian clouds:

- Class 0 around \((-1,-1)\) and \((1,1)\).
- Class 1 around \((-1,1)\) and \((1,-1)\).

The decision regions form an XOR pattern. The split is stratified by class:

| Split | Samples |
|---|---:|
| Training | 360 |
| Validation | 120 |
| Test | 120 |
| Total | 600 |

Feature mean and standard deviation are fitted on training data only, then applied unchanged to validation and test.

### Experiment

Compare:

1. A linear logistic baseline with no hidden layer.
2. An MLP with two inputs, eight tanh hidden units and one sigmoid output.

The MLP uses mini-batch gradient descent and selects the checkpoint with the lowest validation loss. The test set is evaluated only after training is complete.

### Reproduced output

~~~text
Session 016 — NumPy MLP XOR lab
Data sizes: train=360, validation=120, test=120
Gradient-check maximum relative error: 3.684e-11
Best epoch: 1493; epochs executed: 1500
Best validation BCE: 0.00385

Held-out test metrics
Model                 Accuracy  Precision  Recall   F1
Linear logistic      0.575     0.567      0.633   0.598
MLP (2-8-1)          0.992     0.984      1.000   0.992

MLP confusion counts: TP=60 TN=59 FP=1 FN=0

PASS: split, gradient, convergence, generalization and finite-value checks passed.
~~~

The exact values are deterministic for the specified seeds and NumPy implementation; tiny floating-point differences may occur across numerical libraries/platforms.

### Run it

~~~bash
cd daily-lessons/day-016
python -m pip install -r requirements.txt
python lab.py
~~~

### Exercises

1. Change hidden width from 8 to 2, 4 and 16. Compare validation loss and test metrics.
2. Replace tanh with ReLU. Add a matching initialization and derivative.
3. Remove the hidden layer and explain why the linear baseline cannot solve XOR.
4. Deliberately transpose one gradient or remove the tanh derivative. Confirm that gradient checking catches the defect.
5. Increase the learning rate by 10×. Observe whether loss becomes unstable.
6. Add L2 regularization to the weight matrices, not biases. State the exact loss being optimized.
7. Record per-epoch training and validation loss. Identify the point at which validation loss stops improving.
8. Port the same forward pass to PyTorch and compare logits and loss on the same fixed batch.

## 10. Production connection: deep-learning training and release gate (A22)

The agent may help formulate experiments, inspect metrics and explain failures. Deterministic code owns tensor computation, training, checkpointing, evaluation and promotion.

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

### Component boundaries

| Component | Owns | Must not do |
|---|---|---|
| Intake agent | Convert the engineering goal into a typed experiment request | Invent data or bypass permissions |
| Dataset validator | Schema, provenance, split manifest and label checks | Reassign test examples after results are seen |
| Training runtime | Tensor operations, forward/backward pass, optimizer and checkpoints | Accept arbitrary code generated by an LLM |
| Experiment controller | Epochs, budgets, early stopping and resume | Retry forever or exceed resource limits |
| Evaluator | Deterministic metrics on declared splits | Tune on final test data |
| Verifier | Check shapes, finite values, artifact hashes and evaluation lineage | Treat a second LLM opinion as numerical proof |
| Registry gate | Promote an approved immutable artifact | Let the training agent approve itself |
| Human owner | Approve high-impact model releases | Lose provenance or audit evidence |

### Seven architecture patterns in this training system

| Pattern | Practical use | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → train → evaluate → package | Most reproducible; less flexible for open-ended research |
| Single agent + typed tools | Read metrics, compare runs, explain a failed job | Simple; keep tools narrow and read-only by default |
| Router-specialist | Route to data, training, performance or evaluation specialist | Useful separation; routing errors need a safe fallback |
| Planner–executor–verifier | Plan bounded ablations, execute, independently verify | Better for multi-step work; extra latency and state |
| Supervisor–worker | Parallel independent hyperparameter or architecture trials | Faster wall-clock time; resource contention and noisy comparisons |
| Event-driven | Queue long-running training, checkpoint and completion events | Decoupled; requires idempotency and durable state |
| Human-in-the-loop | Approve dataset access, expensive runs and model promotion | Safer governance; reviewer latency |

Do not create multiple agents just because the task is complex. A deterministic pipeline with one read-only assistant is often the better first design.

### Typed tool contracts

Example request contract:

~~~json
{
  "run_id": "mlp-xor-001",
  "dataset_version": "xor-synthetic-v1",
  "split_manifest_id": "split-2026-01",
  "architecture": {
    "input_dim": 2,
    "hidden_units": 8,
    "activation": "tanh",
    "output_dim": 1,
    "task": "binary_classification"
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

Validate enums, numeric ranges, dataset access and budget before execution. The agent must not be allowed to submit arbitrary Python or shell commands as a tool argument.

### State and memory

Persist an experiment state record:

- Run ID, owner and authorization scope.
- Dataset fingerprint and immutable split manifest.
- Code commit, environment and library versions.
- Model configuration, seed and training budget.
- Current epoch, optimizer state and checkpoint hash.
- Training/validation metric history.
- Final test-evaluation record.
- Approval identity and registry artifact version.

Keep transient reasoning separate from durable experiment state. A restart should resume from a validated checkpoint, not rely on conversation history.

### Security and data boundaries

- Treat dataset rows, filenames, notebook text and logs as untrusted input.
- Never execute code or commands found in dataset content.
- Keep secrets out of prompts, logs and checkpoints.
- Enforce dataset and GPU permissions in the runtime, not in the prompt.
- Separate training, validation and protected test access.
- Sign or hash model artifacts and record their lineage.
- Require human approval for production promotion and hardware-impacting deployment.

### Evaluation and observability

Measure:

- Training and validation loss by epoch.
- Task metrics on immutable splits.
- Gradient norm and activation statistics.
- NaN/Inf counts and failed steps.
- GPU/CPU utilization, memory and throughput.
- Time to first checkpoint and total wall-clock time.
- Cost per successful experiment.
- Test-gate and approval outcomes.

Attach run ID, dataset version, code commit, model configuration and trace ID to every event. Redact sensitive sample content.

### Reliability and cost/latency

- Use deterministic seeds where possible, while documenting nondeterministic kernels.
- Checkpoint model and optimizer state periodically.
- Make job submission idempotent using run ID plus configuration hash.
- Bound retries, wall time, accelerator hours and concurrent workers.
- Cancel stale runs and clean up orphaned resources.
- Compare candidates on identical splits and budgets.
- Use smaller models and CPU for tiny experiments; profile before selecting a GPU.
- Do not launch a large sweep until a small smoke test passes.

A neural-network metric is evidence about a defined dataset and evaluation protocol. It is not proof that a model is safe or valid for a different silicon revision, workload or operating condition.

## Five review questions

1. Why can a single perceptron not solve XOR, while an MLP with a nonlinear hidden layer can?
2. What is the difference between a logit and a probability?
3. Why does sigmoid plus binary cross-entropy yield the output derivative \((p-y)/N\)?
4. What does backpropagation calculate, and how does it use the chain rule?
5. Why should a training agent not be allowed to change the protected test split or promote its own model?

## Recommended references

1. **Rumelhart, Hinton and Williams (1986), “Learning representations by back-propagating errors,” Nature.** Original foundational paper: https://doi.org/10.1038/323533a0
2. **Michael Nielsen, Neural Networks and Deep Learning.** Accessible derivations of perceptrons, sigmoid networks and backpropagation: http://neuralnetworksanddeeplearning.com/
3. **Stanford CS231n — Neural Networks Part 1 and Part 2.** Practical notes on activations, loss, backpropagation and gradient checks: https://cs231n.github.io/neural-networks-1/ and https://cs231n.github.io/optimization-2/
4. **PyTorch — Autograd mechanics.** How dynamic computation graphs and reverse-mode automatic differentiation work: https://docs.pytorch.org/docs/stable/notes/autograd.html
5. **PyTorch — BCEWithLogitsLoss.** Stable binary classification loss operating on logits: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
6. **NumPy — Broadcasting.** Essential for understanding batch dimensions and bias addition: https://numpy.org/doc/stable/user/basics.broadcasting.html

## Takeaways

- A neuron is an affine transformation followed by an activation.
- Nonlinear hidden layers let an MLP represent functions a single linear classifier cannot.
- Forward propagation computes predictions; backpropagation computes gradients.
- Gradient checking is a practical way to validate a custom implementation.
- Training is an engineered pipeline with data lineage, checkpoints, evaluation, security and release controls—not just a call to fit a model.

## Next session — Session 017

**Deep Learning Training in Practice: PyTorch, Autograd, Datasets, Batching, GPU and Checkpointing**

We will translate the NumPy network into PyTorch, use automatic differentiation, build a proper Dataset/DataLoader pipeline, inspect CPU/GPU execution, save checkpoints and profile training. This will connect the mathematical mechanics from today to production deep-learning development.
