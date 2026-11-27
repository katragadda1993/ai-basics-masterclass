# Day 007 — Optimization Algorithms: SGD, Momentum, RMSProp, Adam and AdamW

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 2 — Mathematics for AI  
**Previous lesson:** [Day 006 — Information Theory](../day-006/README.md)

## Learning objectives

By the end of this session, you should be able to:
1. Explain why an optimizer is needed after backpropagation computes gradients.
2. Distinguish batch, stochastic and mini-batch gradient descent.
3. Interpret momentum, RMSProp and Adam updates.
4. Explain Adam's first- and second-moment bias correction.
5. Distinguish L2 regularization from decoupled weight decay in AdamW.
6. Select and tune learning rates and schedules responsibly.
7. Diagnose exploding gradients, unstable loss and misleading comparisons.
8. Separate neural-network optimization from an agent runtime's resource-budget controller.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Recall gradients and cross-entropy |
| 5–13 min | Optimizers; batch vs stochastic vs mini-batch |
| 13–23 min | SGD and momentum |
| 23–33 min | AdaGrad and RMSProp |
| 33–42 min | Adam, bias correction and AdamW |
| 42–48 min | Schedules, clipping and checkpointing |
| 48–56 min | Python comparison lab |
| 56–60 min | Agent architecture connection and quiz |

## 1. From gradients to parameter updates

Day 004 showed how backpropagation computes the gradient of a loss with respect to model parameters. Day 006 connected cross-entropy to classification and next-token prediction.

A gradient gives the local direction of steepest increase. To reduce loss, move in the opposite direction.

For parameters \\(\\theta\\), loss \\(L(\\theta)\\), and learning rate \\(\\eta\\):

\\[
\\theta_{t+1}=\\theta_t-\\eta\\nabla_\\theta L(\\theta_t)
\\]

The **optimizer** defines how gradients become parameter updates. It may keep state from earlier steps, adapt update sizes per parameter, or apply regularization.

The learning rate is critical: too large can make training diverge; too small can make progress slow. Optimizers do not eliminate the need to validate hyperparameters.

## 2. Batch, stochastic and mini-batch gradient descent

For a dataset of \\(N\\) examples:

\\[
L(\\theta)=\\frac{1}{N}\\sum_{i=1}^{N}\\ell_i(\\theta)
\\]

- **Batch gradient descent:** compute gradients over all examples per update. Stable, but potentially expensive in memory and compute.
- **Stochastic gradient descent:** use one example per update. Cheap steps, but noisy.
- **Mini-batch SGD:** use a small group of examples per update. This balances gradient noise, hardware utilization and memory.

In modern deep learning, “SGD” often means mini-batch SGD. Inspect the actual batch size and implementation.

GPUs and accelerators benefit from parallel tensor operations over mini-batches. Larger batches can improve throughput but consume more memory and may require learning-rate and regularization changes. Bigger is not automatically better.

## 3. Plain SGD

For mini-batch gradient \\(g_t\\):

\\[
\\theta_{t+1}=\\theta_t-\\eta g_t
\\]

SGD is simple, memory-efficient and a useful baseline. It can zig-zag in narrow valleys and progress slowly when parameter directions have very different curvature scales.

### Worked example

Let \\(L(w)=\\frac{1}{2}(w-3)^2\\), so \\(g=w-3\\). Start at \\(w_0=0\\) with \\(\\eta=0.1\\).

\\[
g_0=-3,\\qquad w_1=0-0.1(-3)=0.3
\\]

The update moves toward the minimum \\(w=3\\).

## 4. Momentum

Momentum accumulates a moving history of gradients:

\\[
v_t=\\beta v_{t-1}+g_t
\\]
\\[
\\theta_{t+1}=\\theta_t-\\eta v_t
\\]

The coefficient \\(\\beta\\) is often initialized near 0.9, but must be tuned. Some implementations scale the new gradient by \\(1-\\beta\\); understand the exact convention before comparing settings.

**Intuition:** a ball rolling downhill gains velocity in directions where gradients consistently point the same way. Momentum can reduce zig-zagging and accelerate progress along consistent directions.

**Trade-off:** momentum stores an extra state tensor per parameter and can overshoot when learning rate and momentum are too aggressive.

## 5. AdaGrad and RMSProp

### AdaGrad

AdaGrad accumulates squared gradients:

\\[
r_t=r_{t-1}+g_t^2
\\]
\\[
\\theta_{t+1}=\\theta_t-\\frac{\\eta}{\\sqrt{r_t}+\\epsilon}g_t
\\]

The effective learning rate decreases for parameters that repeatedly receive large gradients. This can help sparse features, but the accumulated sum only grows, so the effective learning rate may become too small during long training.

### RMSProp

RMSProp uses an exponential moving average of squared gradients:

\\[
s_t=\\rho s_{t-1}+(1-\\rho)g_t^2
\\]
\\[
\\theta_{t+1}=\\theta_t-\\frac{\\eta}{\\sqrt{s_t}+\\epsilon}g_t
\\]

Unlike AdaGrad's unbounded accumulation, the moving average can adapt as gradient scales change. Squares, division and square roots are element-wise for parameter tensors. Epsilon stabilizes division; implementation details can vary.

## 6. Adam: momentum plus adaptive scaling

Adam maintains moving averages of gradients and squared gradients:

\\[
m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t
\\]
\\[
v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2
\\]

Because both estimates start at zero, they are biased toward zero early in training. Adam corrects this:

\\[
\\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\qquad
\\hat v_t=\\frac{v_t}{1-\\beta_2^t}
\\]

Then:

\\[
\\theta_{t+1}=\\theta_t-
\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}
\\]

Common starting values are \\(\\beta_1=0.9\\), \\(\\beta_2=0.999\\), and a small epsilon. They are starting points, not universal laws.

Adam is popular because it combines momentum-like behavior with per-parameter adaptive scaling. It uses additional memory and is not guaranteed to outperform SGD on every task.

## 7. AdamW: decoupled weight decay

Regularization discourages overly complex parameter values and can improve generalization. Distinguish:

- **L2 penalty:** add a term proportional to squared parameter norm to the objective, so its gradient contributes to the update.
- **Weight decay:** shrink parameters directly during the update.

For ordinary SGD, these can be equivalent when coefficients are scaled consistently. With adaptive methods such as Adam, adding an L2 penalty to the gradient is not generally equivalent to decoupled weight decay.

AdamW applies a separate shrinkage term, conceptually:

\\[
\\theta_{t+1}=
\\theta_t-
\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}
-\\eta\\lambda\\theta_t
\\]

Here \\(\\lambda\\) is the weight-decay coefficient. This is a simplified update; implementations can differ in ordering and parameter-group handling.

Tune weight decay on validation data. Biases and normalization parameters are often placed in separate parameter groups rather than decayed identically to all weights.

## 8. Learning-rate schedules

The learning rate can change during training.

**Step decay:** reduce the learning rate at selected milestones.

**Exponential decay:**

\\[
\\eta_t=\\eta_0\\gamma^t
\\]

**Cosine decay:**

\\[
\\eta_t=\\eta_{\\min}+
\\frac{1}{2}(\\eta_{\\max}-\\eta_{\\min})
\\left(1+\\cos\\frac{\\pi t}{T}\\right)
\\]

**Warm-up:** start with a smaller learning rate and increase it during an initial period. This can stabilize early optimization for some large models and batch configurations.

A schedule must be tied to the correct unit: optimizer steps, batches or epochs. Advancing a scheduler once per batch differs from advancing it once per epoch.

## 9. Gradient clipping and numerical stability

Global norm clipping rescales a gradient vector when its norm exceeds a threshold:

\\[
g'=g\\cdot\\min\\left(1,\\frac{c}{\\|g\\|_2}\\right)
\\]

Clipping limits the update contribution; it does not fix every underlying problem. Investigate bad data, unstable activations, loss scaling, learning rate and numerical overflow too.

Monitor training and validation loss, gradient norms, parameter norms, learning rate, NaN/Inf counts, throughput, memory and time per step.

## 10. Optimizer state and checkpointing

Momentum, RMSProp, Adam and AdamW maintain state in addition to model parameters. A resumable checkpoint should normally preserve:
- Model parameters and buffers
- Optimizer state
- Learning-rate scheduler state
- Mixed-precision gradient-scaler state, if used
- Training step/epoch and random-number-generator states
- Configuration, data and code version identifiers

Saving only model weights may be enough for inference, but not for an exact training resume. Reproducibility also depends on data-loader order, hardware kernels and deterministic settings.

## 11. Practical lab: compare optimizers

Run the standard-library lab:

~~~bash
cd daily-lessons/day-007
python lab.py
~~~

The lab minimizes an ill-conditioned quadratic:

\\[
L(x,y)=\\frac{1}{2}(x^2+25y^2)
\\]

Its gradient is:

\\[
\\nabla L=[x,25y]
\\]

The curvature differs by a factor of 25 between axes. The lab implements SGD, momentum, RMSProp, Adam with bias correction, AdamW with decoupled weight decay, cosine learning-rate scheduling, global norm clipping, finite-difference gradient checking and deterministic assertions.

The chosen learning rates are illustrative, not a fair universal benchmark. Optimizers use different hyperparameters and state. Compare them on multiple objectives, budgets and validation metrics before drawing conclusions.

## 12. PyTorch training-loop example

~~~python
import torch
import torch.nn as nn

model = MyModel()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
)
criterion = nn.CrossEntropyLoss()

for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )
    optimizer.step()
~~~

For evaluation, switch to evaluation mode and use no-gradient context. Do not update optimizer state on validation batches. Add scheduler stepping and checkpoint saving according to the selected schedule and recovery requirements.

## 13. Architecture connection: budget-aware agent execution

Training optimizers update model parameters from gradients. An agent runtime controller is a different system: it allocates time, tokens, tool calls and money across a task. The shared idea is controlled resource allocation, not shared mathematics.

~~~mermaid
flowchart TD
  U[Task request] --> I[Validate goal, risk and deadline]
  I --> B[Create hard budget envelope]
  B --> P[Planner proposes bounded steps]
  P --> G[Policy and budget gate]
  G -->|Allowed| E[Executor selects approved model/tool]
  G -->|Over budget or unsafe| H[Escalate or stop]
  E --> V[Verifier checks evidence and acceptance]
  V -->|Pass| D[Deliver result]
  V -->|Repair within budget| P
  V -->|Budget exhausted| H
  E -. metrics .-> O[Cost, latency, retries and traces]
  V -. metrics .-> O
~~~

### Component boundaries

- **Planner:** proposes steps and resource estimates; cannot grant itself more budget.
- **Budget controller:** deterministic service owning remaining tokens, tool calls, wall-clock time and spend.
- **Policy engine:** checks identity, permissions, data scope and side-effect risk.
- **Executor:** calls typed, allow-listed tools and approved model endpoints.
- **Verifier:** checks output against explicit acceptance criteria.
- **State store:** persists task status, budget reservations, idempotency keys and artifacts.
- **Human reviewer:** approves high-impact actions or budget increases.

### Production rules

1. Set hard budgets before execution; model-generated cost estimates are not enforcement.
2. Reserve budget before parallel workers launch to avoid oversubscription.
3. Use deterministic fallback or abstention when budget is exhausted.
4. Retry only transient errors, with bounded attempts and backoff.
5. Make side-effecting tools idempotent and require approval for irreversible actions.
6. Record model/tool versions, token counts, latency, retries, decision reasons and evidence.
7. Evaluate cost per successful task, not merely cost per model call.

### Architecture trade-offs

| Pattern | Advantage | Risk / cost |
|---|---|---|
| Deterministic workflow | Predictable cost and latency | Less flexible for ambiguity |
| Single agent + tools | Simple control surface | Context/tool overload |
| Router + specialists | Focused prompts and permissions | Routing errors and extra hops |
| Planner–executor–verifier | Multi-step work with checks | More calls and latency |
| Supervisor–workers | Parallel throughput | Coordination and budget contention |
| Event-driven | Decouples slow work | Duplicate delivery and recovery complexity |
| Human-in-the-loop | Controls high-impact actions | Approval latency |

Use the simplest architecture that meets quality, safety and deadline requirements. More agents do not automatically mean better results.

## 14. Key terminology

| Term | Meaning |
|---|---|
| Optimizer | Algorithm that turns gradients into parameter updates |
| Learning rate | Scale of a parameter update |
| Mini-batch | Small subset of examples used for one update |
| Momentum | Moving accumulation of past gradients |
| Adaptive optimizer | Uses per-parameter gradient statistics |
| First moment | Exponential moving average of gradients |
| Second moment | Exponential moving average of squared gradients |
| Bias correction | Adjusts zero-initialized moment estimates |
| Weight decay | Direct shrinkage of parameter values |
| Scheduler | Changes learning rate according to a policy |
| Gradient clipping | Limits gradient magnitude or norm |
| Optimizer state | Extra tensors needed by an optimizer |
| Budget envelope | Runtime-enforced limits on an agent task |

## 15. Five review questions

1. Why can mini-batch SGD be faster in practice than full-batch gradient descent on accelerator hardware?
2. What problem does momentum address, and what can happen if momentum and learning rate are too high?
3. How does RMSProp differ from AdaGrad?
4. Why does Adam use bias correction, and how is AdamW different from adding an L2 penalty to Adam's gradient?
5. Why must an agent's budget controller be deterministic and separate from the LLM planner?

## 16. Completion checklist

- [ ] Explain batch, stochastic and mini-batch gradient descent.
- [ ] Derive the momentum update.
- [ ] Explain RMSProp's moving average of squared gradients.
- [ ] Explain Adam's moments and bias correction.
- [ ] Explain decoupled weight decay.
- [ ] Run the Python lab and inspect all optimizer results.
- [ ] Explain why runtime budget enforcement is separate from model optimization.
- [ ] Answer the five review questions.

## 17. References

1. [PyTorch — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html): practical training-loop sequence and optimizer usage.
2. [PyTorch — torch.optim](https://docs.pytorch.org/docs/stable/optim.html): optimizer implementations and parameters.
3. [Kingma and Ba — Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980): original Adam paper.
4. [Loshchilov and Hutter — Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW and the distinction between L2 regularization and decoupled weight decay.
5. [PyTorch — Gradient Clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html): global gradient norm clipping API.

## Next lesson

**Day 008 — Machine Learning Foundations: Problem Formulation, Data Splits, Features, Labels and Baselines.**

We will move from mathematical optimization into the end-to-end machine-learning workflow: defining a prediction problem, preparing data, avoiding leakage, building a baseline and evaluating generalization.
