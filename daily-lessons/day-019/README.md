# Session 019 — Recurrent Neural Networks: RNN, Hidden State, BPTT, Vanishing Gradients and LSTM

**Phase:** 4 — Deep Learning  
**Duration:** 60 minutes  
**Prerequisite:** Session 018 — CNNs, convolution, feature maps, pooling and PyTorch model training.

## Why this lesson follows Session 018

Session 018 introduced convolutional networks, which exploit local structure and shared filters. This session introduces a different inductive bias: a recurrent network processes an ordered sequence while carrying a state from one time step to the next.

This is useful when order and temporal context matter: audio frames, sensor histories, protocol events, DMA latency traces and event streams. We will derive the vanilla RNN, understand backpropagation through time (BPTT), diagnose vanishing/exploding gradients, and compare LSTM and GRU cells.

## Learning objectives

By the end of this session, you will be able to:

1. Explain why sequence order requires a temporal model.
2. Calculate the hidden-state update of a vanilla RNN.
3. Explain unrolling and backpropagation through time.
4. Derive why repeated Jacobian products cause vanishing or exploding gradients.
5. Explain gradient clipping and truncated BPTT.
6. Describe LSTM input, forget and output gates and the cell state.
7. Compare RNN, LSTM and GRU trade-offs.
8. Interpret PyTorch recurrent tensor and state shapes.
9. Handle variable-length sequences and stateful streaming safely.
10. Design a governed recurrent-model training and inference workflow.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Why sequence models; connection to CNNs |
| 5–13 min | Vanilla RNN and hidden state |
| 13–23 min | Unrolling, BPTT and gradient pathologies |
| 23–35 min | LSTM gates and GRU comparison |
| 35–42 min | PyTorch shapes, packed sequences and streaming state |
| 42–54 min | Run the sequence-memory lab |
| 54–60 min | Production architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Sequence | Ordered observations, often indexed by time |
| Time step | One position in a sequence |
| Hidden state | Recurrent summary passed from one step to the next |
| Cell state | LSTM's separate memory pathway |
| Unrolling | Drawing a recurrent cell once per sequence position |
| BPTT | Backpropagation through the unrolled time steps |
| Vanishing gradient | Gradient magnitude shrinks across many recurrent steps |
| Exploding gradient | Gradient magnitude grows excessively across steps |
| Gradient clipping | Limits gradient magnitude before an optimizer update |
| Truncated BPTT | Backpropagates through a bounded sequence chunk |
| Gate | Learned control value that regulates information flow |
| Bidirectional RNN | Processes a sequence in forward and reverse directions |
| Stateful inference | Carries recurrent state across streaming chunks |
| Packed sequence | PyTorch representation that avoids recurrent work on padded positions |

## 1. Why a recurrent model?

A CNN is effective when local patterns can be detected with shared filters. A recurrent model is designed to process a sequence and carry information from earlier positions.

Examples:
- An audio frame depends partly on preceding frames.
- A DMA timeout may be preceded by a rising retry count.
- A SoundWire fault may follow a sequence of clock or synchronization events.
- A sensor value may be normal in one operating mode but unusual after a particular transition.

A recurrent model receives one time step at a time and updates its state.

### Sequence task types

| Task | Input → output | Example |
|---|---|---|
| Many-to-one | Sequence → one label/value | Classify a telemetry window |
| Many-to-many | Sequence → output at each step | Label each audio frame |
| One-to-many | One context → output sequence | Generate a sequence |
| Sequence-to-sequence | Input sequence → output sequence | Translate or transform a sequence |

## 2. Vanilla RNN from first principles

At time step t, a vanilla tanh RNN calculates:

\[
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
\]

A prediction head may calculate:

\[
o_t=W_{hy}h_t+b_y
\]

Where:
- \(x_t\) is the current input vector.
- \(h_{t-1}\) is the previous hidden state.
- \(h_t\) is the updated hidden state.
- \(W_{xh}\) maps input to hidden state.
- \(W_{hh}\) is the recurrent weight matrix.
- The same weights are reused at every time step.

The hidden state is a learned summary, not a perfect copy of the full history. Its capacity is limited by hidden size and the dynamics learned during training.

### Worked scalar example

For a simplified scalar recurrent unit, suppose:

\[
h_t=\tanh(0.5x_t+0.8h_{t-1})
\]

Let \(h_0=0\), \(x_1=1\), and \(x_2=0\).

\[
h_1=\tanh(0.5)\approx0.4621
\]

\[
h_2=\tanh(0.8\times0.4621)\approx0.3549
\]

Even though the second input is zero, the second hidden state is non-zero because the network carries information from the previous step.

## 3. Unrolling and BPTT

A recurrent cell reuses the same parameters at every step. To train it, we conceptually unroll the recurrence into a chain:

~~~text
x1        x2        x3                 xT
 |         |         |                  |
 v         v         v                  v
[RNN] --> [RNN] --> [RNN] --> ... --> [RNN]
  h1        h2        h3                 hT
  |                                    |
  +------------------------------------+
                 loss
~~~

The forward pass computes the hidden states and predictions. Backpropagation then moves backward through the unrolled steps.

For a sequence loss:

\[
L=\sum_{t=1}^{T}L_t
\]

The gradient for shared recurrent parameters includes contributions from every time step at which those parameters were used.

### Why gradients vanish or explode

The gradient from a later hidden state to an earlier one contains a product of recurrent Jacobians:

\[
\frac{\partial h_T}{\partial h_t}
=
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
\]

For a tanh RNN, each local Jacobian includes the recurrent matrix and the derivative of tanh. If the repeated products tend to contract, gradients can shrink toward zero. If they expand, gradients can grow rapidly.

Consequences:
- Early sequence positions may receive almost no learning signal.
- Training can become unstable.
- The network may learn short patterns but fail to retain long-range dependencies.

This is not simply a software bug. It is a mathematical difficulty of optimization through long recurrent chains.

### Two practical controls

**Gradient clipping**

For global gradient norm \(g\) and maximum norm \(\tau\):

\[
g' = g\min\left(1,\frac{\tau}{\|g\|}\right)
\]

Clipping limits exploding updates. It does not restore information already lost through vanishing gradients.

**Truncated BPTT**

Process a long stream in chunks. Carry the hidden state forward, but detach it at chunk boundaries so the backward graph does not extend indefinitely.

~~~python
# Conceptual truncated-BPTT pattern
state = None
for x_chunk, y_chunk in stream:
    optimizer.zero_grad(set_to_none=True)
    output, state = model(x_chunk, state)
    loss = criterion(output, y_chunk)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    state = tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
~~~

The state is preserved numerically, but gradients do not cross the detach boundary. Chunk length is therefore a modeling and systems trade-off, not just a memory setting.

## 4. LSTM — a controlled memory pathway

LSTM means Long Short-Term Memory. It introduces a cell state and gates that regulate what to retain, write and expose.

~~~text
                 +-------------------------------+
                 |                               |
c(t-1) --------> forget gate ----+                |
                                 |                v
x(t), h(t-1) --> input gate --> candidate --> [add] --> c(t)
                                                  |
                                                  v
                                              tanh(c)
                                                  |
x(t), h(t-1) --> output gate ------------------ multiply --> h(t)
~~~

The common LSTM equations are:

\[
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
\]

\[
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)
\]

\[
g_t=\tanh(W_g[x_t,h_{t-1}]+b_g)
\]

\[
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)
\]

\[
c_t=f_t\odot c_{t-1}+i_t\odot g_t
\]

\[
h_t=o_t\odot\tanh(c_t)
\]

Where:
- \(i_t\): input gate — how much candidate information to write.
- \(f_t\): forget gate — how much previous cell state to retain.
- \(g_t\): candidate cell content.
- \(o_t\): output gate — how much cell information to expose.
- \(c_t\): cell state.
- \(h_t\): hidden/output state.

The additive cell-state update provides a path through which information and gradients can flow more easily than through repeated nonlinear hidden-state transformations alone. Gates can learn to preserve or replace information.

LSTM reduces the long-dependency optimization problem; it does not guarantee that every long sequence will be learned.

### Forget-gate intuition

If a forget-gate value is near 1, the corresponding cell-state component is mostly retained. If it is near 0, it is mostly removed.

A positive forget-gate bias is a common initialization heuristic. It is not a substitute for validation and is not a universal requirement.

## 5. GRU — a lighter gated alternative

A Gated Recurrent Unit combines memory into a single hidden state and uses fewer gates than an LSTM.

A simplified formulation:

\[
r_t=\sigma(W_r[x_t,h_{t-1}])
\]

\[
z_t=\sigma(W_z[x_t,h_{t-1}])
\]

\[
\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}])
\]

\[
h_t=(1-z_t)\odot\tilde h_t+z_t\odot h_{t-1}
\]

- Reset gate \(r_t\) controls how much previous state contributes to the candidate.
- Update gate \(z_t\) controls the mix of old state and candidate state.

Exact gate conventions and algebraic arrangements vary across implementations; PyTorch documents its GRU equations and parameter layout.

### RNN vs LSTM vs GRU

| Model | State | Strength | Trade-off |
|---|---|---|---|
| Vanilla RNN | Hidden state | Simple, fewer parameters | Long-range gradients can be difficult |
| LSTM | Hidden + cell | Explicit gated memory | More gates and parameters |
| GRU | Hidden state | Fewer gates than LSTM | Different memory parameterization; not always faster in every runtime |

For input size \(I\), hidden size \(H\), one layer and PyTorch's two bias vectors, approximate recurrent parameter counts are:

\[
RNN=H I+H^2+2H
\]

\[
GRU=3(HI+H^2+H)
\]

\[
LSTM=4(HI+H^2+H)
\]

For \(I=8\) and \(H=64\):

| Cell | Recurrent parameters |
|---|---:|
| RNN | 4,736 |
| GRU | 14,016 |
| LSTM | 18,688 |

These counts exclude a task-specific output head. Actual speed depends on sequence length, batch size, hidden size, kernel implementation, device and data pipeline.

## 6. PyTorch tensor shapes and state

With §batch_first=True§, a recurrent input is:

\[
X:[N,L,F]
\]

Where N = batch size, L = sequence length, F = input feature count.

For hidden size H and one unidirectional layer:
- RNN output: \([N,L,H]\)
- RNN final hidden state: \([1,N,H]\)
- LSTM output: \([N,L,H]\)
- LSTM final hidden state: \([1,N,H]\)
- LSTM final cell state: \([1,N,H]\)

For a bidirectional model, output feature width becomes \(2H\), and the state direction dimension becomes 2.

**Important PyTorch detail:** §batch_first=True§ changes input and output layout, but does not change the hidden-state layout.

### Minimal PyTorch example

~~~python
import torch
from torch import nn

batch, steps, features, hidden = 8, 50, 2, 32
x = torch.randn(batch, steps, features)

rnn = nn.RNN(features, hidden, batch_first=True)
rnn_output, rnn_h = rnn(x)

lstm = nn.LSTM(features, hidden, batch_first=True)
lstm_output, (lstm_h, lstm_c) = lstm(x)

gru = nn.GRU(features, hidden, batch_first=True)
gru_output, gru_h = gru(x)

assert rnn_output.shape == (batch, steps, hidden)
assert lstm_output.shape == (batch, steps, hidden)
assert gru_output.shape == (batch, steps, hidden)
~~~

For variable-length sequences, use padding plus §pack_padded_sequence§ where appropriate. This avoids recurrent computation over padded time steps. Be careful when using final states, bidirectional outputs and unsorted batch lengths.

## 7. Stateful streaming inference

For offline classification, the entire sequence can be supplied at once. For streaming inference, the model may process chunks:

~~~text
stream chunk 1 --> recurrent model --> state 1
stream chunk 2 + state 1 --> recurrent model --> state 2
stream chunk 3 + state 2 --> recurrent model --> state 3
~~~

The serving system must define:
- Which stream/device owns each state.
- When a state is initialized or reset.
- How state is versioned with the model.
- How gaps, duplicates and late events are handled.
- How long idle states are retained.
- How state is restored after process restart.

Never reuse hidden state across unrelated devices, tenants, sessions or independent sequences. A stale or cross-device state can silently contaminate predictions.

For online decisions, a bidirectional model is usually unsuitable because its reverse direction requires future observations. Bidirectional models are useful when the complete sequence is available offline.

## 8. Practical lab — delayed-marker memory task

The lab creates a synthetic sequence-classification problem:
- Sequence length: 50 time steps.
- Input features: 2 per step.
- The class is encoded at the first time step.
- Later time steps contain distractor noise.
- The model predicts the original class from the final recurrent output.

This forces the model to preserve information across intervening steps.

### Dataset

| Split | Samples |
|---|---:|
| Training | 3,000 |
| Validation | 600 |
| Test | 600 |
| Sequence length | 50 |
| Input features | 2 |
| Classes | 2 |

The lab trains vanilla RNN and LSTM candidates with the same data split, batch size and optimizer family. It also verifies RNN/LSTM/GRU tensor shapes, bidirectional state shapes and packed variable-length sequences.

### Reproduced CPU reference run

Environment: PyTorch 2.10.0 CPU build, deterministic algorithms enabled, one CPU thread.

| Model | Validation accuracy | Test accuracy | Test loss |
|---|---:|---:|---:|
| Vanilla RNN | 100.00% | 100.00% | 0.000217 |
| LSTM | 100.00% | 100.00% | 0.000232 |

Both models solve this deliberately controlled task. This is not evidence that vanilla RNNs generally match LSTMs on long real-world sequences.

### Gradient-path illustration

At random initialization, the lab measures the gradient norm from the final output to the initial recurrent state. It compares vanilla RNN \(h_0\) with the LSTM cell-state path \(c_0\), using a positive forget-gate bias for the LSTM diagnostic.

| Steps | Vanilla RNN gradient to \(h_0\) | LSTM gradient to \(c_0\) |
|---:|---:|---:|
| 10 | 2.157238e-02 | 3.686270e-01 |
| 30 | 2.235428e-07 | 2.056554e-02 |
| 60 | 5.359278e-15 | 8.036405e-05 |
| 100 | 0 | 1.610556e-07 |

This is an illustrative initialization diagnostic, not a controlled quality benchmark: the RNN hidden state and LSTM cell state are different variables, and results depend on initialization and gate settings. It demonstrates how a direct gated cell-state path can preserve a stronger gradient signal over more steps in this example.

### Run the lab

~~~bash
cd daily-lessons/day-019
python -m pip install -r requirements.txt
python lab.py
~~~

Optional device selection:

~~~bash
python lab.py --device cpu
python lab.py --device cuda
~~~

The script checks tensor shapes, packed-sequence handling, finite losses and gradients, and held-out learning. CUDA is selected only when available. CUDA execution was not hardware-tested in the reference run.

### Extension exercises

1. Increase sequence length from 50 to 100 and compare convergence.
2. Remove the positive LSTM forget-gate bias initialization and compare learning.
3. Change hidden size from 32 to 8 and 64.
4. Add a third telemetry feature and document the input contract.
5. Implement gradient clipping off/on and record maximum gradient norm.
6. Implement truncated BPTT over chunks and show how detaching state changes gradient reach.
7. Try GRU on the delayed-marker task.
8. Compare a unidirectional and bidirectional LSTM for offline classification.
9. Build a streaming inference wrapper that stores state per device ID and resets state on a new session.
10. Replace the synthetic sequence with a real, properly de-identified waveform or hardware trace only after defining a leakage-safe split.

## 9. Application to audio and silicon diagnostics

### Audio

An RNN/LSTM can process:
- Mel-spectrogram frames.
- Audio feature vectors.
- Short-time energy and spectral features.
- Streaming voice or event sequences.

A 1D CNN captures local waveform patterns; a recurrent model can summarize temporal progression. CNN-RNN hybrids are also common.

### Silicon telemetry

A sequence may contain:

\[
x_t=[DMA\_retries, AXI\_errors, queue\_depth, latency, temperature]
\]

The model can learn patterns such as a retry-rate increase followed by queue growth and latency degradation.

However, sequence correlation is not root-cause proof. Evaluate by device, firmware and time period; avoid windows from the same capture leaking across train and test sets.

### Choosing a model

| Requirement | Candidate |
|---|---|
| Short local patterns, low latency | 1D CNN |
| Sequential state and moderate history | GRU or LSTM |
| Offline sequence labeling with future context | Bidirectional LSTM/GRU |
| Very long context and parallel training | Transformer |
| Tiny MCU/strict power budget | Benchmark compact GRU/RNN or temporal CNN |

The correct choice depends on measured quality, memory, latency, energy, sequence length and deployment constraints.

## 10. New reusable agent architecture — A25

**Sequence Modeling and Stateful Inference Governance Gate**

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

### Pattern mapping

| Agent pattern | Application |
|---|---|
| Deterministic workflow | Validate → order → window → train/evaluate → release |
| Single agent + typed tools | Read-only inspection of traces, metrics and model metadata |
| Router-specialist | Route to RNN, LSTM, GRU, CNN or transformer candidate |
| Planner–executor–verifier | Plan bounded sequence experiments and independently verify outputs |
| Supervisor–worker | Parallel seeds, model families or device cohorts |
| Event-driven | Ingest telemetry and process chunks asynchronously |
| Human-in-the-loop | Approve model rollout, state-reset policy and diagnostic thresholds |

### Production engineering contract

**Component boundaries**
- Ingestion owns event identity, timestamps and ordering.
- Window builder owns sequence assembly and missing-step policy.
- Model runtime owns tensor conversion and recurrent execution.
- State store owns per-stream hidden/cell state and version compatibility.
- Agent owns explanation and read-only investigation, not model-state mutation or hardware control.

**Typed tools and state**
- Define input contracts: device/session ID, timestamp, feature schema, sequence length and model version.
- Persist state with device/tenant identity, model version, last processed offset and expiry.
- Reset state at session boundaries and reject incompatible state versions.
- Keep conversational memory separate from numeric recurrent state.

**Security**
- Enforce tenant/device authorization outside the LLM.
- Treat firmware logs and traces as untrusted data.
- Prevent cross-device hidden-state reuse.
- Restrict the assistant to read-only tools unless a separately authorized workflow approves an action.

**Evaluation**
- Use device-, firmware- and time-separated splits.
- Report detection delay, false alarms per device-hour, precision/recall and calibration where relevant.
- Test missing events, duplicate events, timestamp disorder and abrupt sequence resets.
- Compare against persistence, threshold and temporal-CNN baselines.

**Reliability**
- Clip gradients and monitor finite values.
- Use truncated BPTT with documented detach boundaries.
- Checkpoint model, optimizer, scheduler, RNG and data cursor for training recovery.
- Make event updates idempotent using stream ID and sequence offset.
- Fall back to a safe baseline if model state is missing or incompatible.

**Observability and cost/latency**
- Track sequence length, queue age, missing-step rate, state resets, hidden-state version, p50/p95 inference latency, memory and throughput.
- Attribute cost to model family, device and sequence length.
- Batch where throughput matters; use smaller stateful chunks where latency matters.
- Keep recurrent inference deterministic and cheap; invoke an LLM only when a human-readable investigation is useful.

**Release rule:** the assistant may recommend and explain a candidate. Deterministic evaluation and an independent verifier decide whether it meets policy; a human approves production promotion.

## 11. Five review questions

1. Why does a vanilla RNN reuse the same recurrent weights at every time step?
2. What mathematical operation in BPTT causes vanishing and exploding gradients?
3. What is the purpose of the LSTM cell state, and how do the forget and input gates affect it?
4. What is the difference between carrying state across chunks and carrying gradients across chunks in truncated BPTT?
5. Why must streaming recurrent state be keyed by device/session and model version rather than shared globally?

## 12. Recommended references

1. [PyTorch RNN documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
2. [PyTorch LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
3. [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
4. [PyTorch packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
5. Hochreiter and Schmidhuber, [Long Short-Term Memory (1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
6. Cho et al., [Learning Phrase Representations using RNN Encoder–Decoder (EMNLP 2014)](https://aclanthology.org/D14-1179/)
7. [Deep Learning — Sequence Modeling: Recurrent and Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html)

## 13. Next session — Session 020

**Attention Mechanism and Transformers: From Query-Key-Value to Self-Attention**

- Why recurrent models process tokens sequentially.
- Query, key and value vectors.
- Scaled dot-product attention.
- Attention masks and causal masking.
- Multi-head attention.
- Positional encodings.
- Transformer block architecture.
- Complexity and memory trade-offs.
- Implement a small self-attention layer in PyTorch.
- Connect the attention mechanism to modern LLMs.

**Course milestone:** 19 sequential lessons published. Phase 4 has progressed from MLPs and PyTorch through CNNs to recurrent sequence models. Session 020 begins the transition to attention and transformers.
