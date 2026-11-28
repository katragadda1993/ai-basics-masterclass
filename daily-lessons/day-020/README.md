# Session 020 — Attention Mechanism and Transformers: From Query-Key-Value to Self-Attention

**Phase:** 5 — Transformers and generative AI  
**Duration:** 60 minutes  
**Prerequisite:** Session 019 — RNN, LSTM, GRU, BPTT and stateful sequence modeling.

## Why this lesson follows Session 019

RNNs carry a hidden state from one time step to the next. This is useful for streaming sequences, but it makes training sequential across time and can make long-range dependencies difficult to learn. The Transformer replaces recurrence with attention: each position can combine information from other positions, subject to an attention mask.

We will derive attention, explain query/key/value projections, causal and padding masks, multi-head attention, positional information and the Transformer block. Then we implement attention in PyTorch and test that a causal mask prevents future-token leakage.

## Learning objectives

By the end of this session, you should be able to:

1. Explain why attention is an alternative to recurrent state propagation.
2. Interpret query, key and value vectors.
3. Derive scaled dot-product attention and calculate a small example.
4. Distinguish self-attention from cross-attention.
5. Explain causal and padding masks and why mask semantics must be tested.
6. Explain multi-head attention and tensor reshaping.
7. Describe why positional information is needed.
8. Draw encoder, decoder and decoder-only Transformer blocks.
9. Estimate attention compute and memory growth with sequence length.
10. Implement and verify causal multi-head attention in PyTorch.
11. Design safe, observable Transformer experiment and inference workflows.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | RNN recap and why attention |
| 5–15 min | Query, key, value and scaled dot-product attention |
| 15–23 min | Masks and causal information flow |
| 23–32 min | Multi-head attention and positional information |
| 32–40 min | Transformer blocks and complexity |
| 40–53 min | PyTorch implementation and tests |
| 53–60 min | Production architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Attention | Weighted combination of value vectors using query-key compatibility |
| Query (Q) | Learned representation of what a position is seeking |
| Key (K) | Learned representation used to match against queries |
| Value (V) | Content mixed into the attention output |
| Self-attention | Q, K and V are projected from the same sequence representation |
| Cross-attention | Queries come from one sequence; keys and values come from another |
| Scaled dot-product attention | Dot-product scores divided by square root of key dimension, then softmax-weighted values |
| Causal mask | Prevents a position from reading future positions |
| Padding mask | Prevents padded positions from contributing as keys/values |
| Attention head | One learned Q/K/V projection subspace |
| Multi-head attention | Several attention heads computed in parallel and combined |
| Positional encoding | Information that identifies token or frame positions |
| Residual connection | Adds a block input to its transformed output |
| Layer normalization | Normalizes activations across a feature dimension |
| Feed-forward network (FFN) | Position-wise MLP applied after attention |
| Context window | Maximum number of tokens a model can process together |
| KV cache | Stored key/value tensors reused during autoregressive generation |

## 1. From recurrence to attention

In Session 019, the RNN update was:

\[
h_t=f(x_t,h_{t-1})
\]

Information from earlier positions must pass through recurrent state transitions. A Transformer instead lets a token representation combine information from many other positions in the current context.

~~~text
RNN:
x1 -> h1 -> h2 -> h3 -> ... -> hT

Self-attention:
x1 <----> x2
 | \       / |
 |  \     /  |
x3 <----> x4
(each query can score multiple keys, subject to a mask)
~~~

The important distinction is not that Transformers have unlimited memory. They are limited by context window, attention mask, positional representation and compute. Their advantage is that training-time attention can be computed for many positions in parallel.

## 2. Query, key and value

Let the input sequence representation be:

\[
X\in\mathbb{R}^{B\times T\times d_{model}}
\]

where B is batch size, T is sequence length and d_model is embedding dimension.

Learned linear projections create:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

For one attention head:

\[
Q,K\in\mathbb{R}^{B\times T\times d_k}
\]

\[
V\in\mathbb{R}^{B\times T\times d_v}
\]

An intuition:

- Query: what information does this position want?
- Key: what information does this position advertise for matching?
- Value: what content should be passed along if it is relevant?

These are learned vector spaces, not literal database fields or human-readable labels.

### Scaled dot-product attention

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

The matrix QKᵀ contains compatibility scores between every query position and every key position. Softmax converts each row into weights. Those weights form a weighted sum of V.

Why divide by square-root of d_k? If query and key components have roughly unit variance and are independent, the variance of their dot product grows with d_k. Scaling keeps scores from becoming unnecessarily large and pushing softmax toward saturation.

### Worked numerical example

Use one query and two keys:

\[
q=[1,0],\quad k_1=[1,0],\quad k_2=[0,1]
\]

Let the corresponding scalar values be 10 and 20. Here d_k=2.

The scaled scores are:

\[
s=[1/\sqrt{2},0]\approx[0.7071,0]
\]

Softmax gives approximately:

\[
\alpha=[0.6698,0.3302]
\]

The output is:

\[
0.6698(10)+0.3302(20)\approx13.302
\]

The output is not a hard selection of one value. It is a differentiable weighted mixture.

## 3. Self-attention and cross-attention

**Self-attention:** the same input sequence supplies Q, K and V. It lets positions exchange information with other positions in that sequence.

**Cross-attention:** Q comes from one representation while K and V come from another. In an encoder-decoder Transformer, decoder queries attend to encoder outputs.

~~~text
Self-attention:
X ----> WQ ----> Q
 \----> WK ----> K
  \---> WV ----> V

Cross-attention:
Decoder states ----> WQ ----> Q
Encoder outputs ---> WK ----> K
                 \-> WV ----> V
~~~

Self-attention is not automatically causal. Whether a position can see earlier, later or all positions is determined by the mask.

## 4. Attention masks: correctness is a quality and safety boundary

### Causal mask

For autoregressive next-token prediction, position t must not see positions greater than t.

For four positions, the allowed pattern is:

\[
M_{allowed}=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
\]

Disallowed future scores are set to negative infinity before softmax, making their weights zero.

### Padding mask

Batches often contain sequences of different lengths. Padding positions are placeholders, not real tokens. A key-padding mask prevents valid queries from attending to padded key/value positions. Loss computation must also ignore padded target positions.

### A PyTorch mask trap

Different PyTorch APIs use different boolean-mask conventions:

- In PyTorch functional scaled-dot-product attention, boolean True means the position is allowed to participate.
- In PyTorch MultiheadAttention boolean key-padding/attention mask conventions, True means the position is masked out or not allowed.

Do not copy a boolean mask from one API to another without checking its contract. An inverted causal mask can silently leak future tokens or hide valid context.

A production test should perturb future tokens and assert that earlier causal outputs remain unchanged.

## 5. Multi-head attention

One attention head learns one projection subspace. Multi-head attention runs several heads in parallel:

\[
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)
\]

\[
MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W_O
\]

If d_model=128 and there are 4 heads, each head commonly uses d_k=d_v=32.

~~~text
Input [B,T,128]
       |
   Q / K / V projections
       |
  split into 4 heads
       |
  head 1  head 2  head 3  head 4
       \     |      |     /
          concatenate
               |
        output projection
               |
         [B,T,128]
~~~

Heads are not guaranteed to have human-interpretable roles. It is safer to say they provide multiple learned representation subspaces than to claim one head always detects syntax or a particular concept.

## 6. Positional information

Self-attention without positional information is permutation-equivariant: reordering input tokens reorders outputs in the same way. It has no built-in notion of token order.

A Transformer therefore adds or injects positional information.

### Sinusoidal positional encoding

For position pos and dimension index i:

\[
PE(pos,2i)=\sin\left(pos/10000^{2i/d_{model}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right)
\]

Other approaches include learned absolute embeddings and relative/rotary position methods. Modern LLM families frequently use rotary position embeddings (RoPE); we will study these more deeply later.

For an audio spectrogram or telemetry sequence, position may correspond to frame index or sample-window index. The preprocessing pipeline must define what a position means.

## 7. The Transformer block

A simplified encoder block contains:

1. Multi-head self-attention.
2. Residual connection and layer normalization.
3. Position-wise feed-forward network.
4. Another residual connection and normalization.

~~~text
Input
  |
Multi-head self-attention
  |
Residual + LayerNorm
  |
Feed-forward network (Linear -> GELU/ReLU -> Linear)
  |
Residual + LayerNorm
  |
Output
~~~

Two common normalization arrangements:

- Post-LayerNorm: normalization after residual addition, used in the original paper's block presentation.
- Pre-LayerNorm: normalization before the attention/FFN sublayer, common in many modern decoder architectures.

### Encoder, decoder and decoder-only models

| Architecture | Attention behavior | Typical use |
|---|---|---|
| Encoder | Usually bidirectional self-attention | Representation learning, classification |
| Encoder-decoder | Encoder self-attention, decoder causal self-attention and cross-attention | Sequence-to-sequence tasks |
| Decoder-only | Causal self-attention | Autoregressive language generation |

A decoder-only language model predicts the next token from preceding context. At training time, a causal mask lets all positions in a batch be processed in parallel while preserving the correct information boundary.

## 8. Complexity and systems trade-offs

For dense full self-attention with sequence length T and model dimension d:

- Attention score computation scales approximately as O(T²d).
- The attention score/probability matrix has O(T²) elements per head and batch item.
- Q/K/V and output projections add terms roughly proportional to O(Td²).

Doubling T approximately quadruples the pairwise attention matrix size. This is why long context windows can be expensive.

### Training versus generation

**Training:** all token positions in a training sequence can be processed in parallel, because the causal mask blocks future information.

**Autoregressive generation:** new tokens are generated one after another. A KV cache stores keys and values from earlier positions so the model does not recompute them from scratch at every step.

The KV cache reduces repeated computation but consumes memory that grows with batch size, number of layers, KV-head count, head dimension and cached sequence length.

### Practical optimization vocabulary

- FlashAttention / memory-efficient SDPA: compute attention with improved memory traffic and reduced materialization of large intermediate matrices.
- Grouped-query attention (GQA): multiple query heads share fewer key/value heads.
- Multi-query attention (MQA): query heads share one key/value head.
- Sliding-window attention: restricts each position to a bounded neighborhood.
- KV-cache quantization: reduces cache memory at a quality/complexity trade-off.

These are engineering variants, not required to understand the original Transformer.

## 9. Hands-on lab — implement attention in PyTorch

The lab in lab.py implements:

1. Manual scaled dot-product attention.
2. Causal masking with a lower-triangular allowed-position matrix.
3. A custom multi-head self-attention layer.
4. A pre-LayerNorm Transformer block.
5. Comparison with PyTorch SDPA.
6. A causal leakage test.

### Reference run

Environment: PyTorch 2.10.0 CPU build.

| Check | Result |
|---|---:|
| Manual attention vs PyTorch SDPA maximum error | 3.576e-07 |
| Causal future-token leakage maximum error | 0 |
| Non-causal earlier-output change after future perturbation | 6.812 |
| Transformer block output shape | (4, 12, 32) |
| Transformer block parameters | 12,704 |

The non-causal output changes when future positions change; the causal output for earlier positions does not. This is the intended behavior.

The lab is a mechanics and correctness exercise, not a trained language model or benchmark.

### Run locally

~~~bash
cd daily-lessons/day-020
python -m pip install -r requirements.txt
python lab.py
python lab.py --device cpu
python lab.py --device cuda
~~~

The CUDA path is available when a compatible PyTorch CUDA build and GPU are installed. The reference run was CPU-only; CUDA was not hardware-tested.

### Exercises

1. Change d_model from 32 to 64 and keep four heads. Verify head dimension and parameter count.
2. Change number of heads from 4 to 8 while keeping d_model=32.
3. Remove the causal mask and observe why earlier outputs change when future tokens are perturbed.
4. Add a padding mask for sequences with different lengths.
5. Implement sinusoidal positional encodings and add them to token embeddings.
6. Compare one attention head with four heads on the same input.
7. Measure memory and runtime as sequence length increases from 128 to 256 to 512.

## 10. Connection to audio and silicon diagnostics

Transformers are not limited to text.

### Audio

A spectrogram can be converted into time-frequency patches or frame embeddings. Attention can combine information across distant frames, useful for speech, acoustic event detection and audio understanding.

### Silicon telemetry

A sequence can be represented as timestamped feature vectors:

\[
x_t=[DMA\ retries,\ AXI\ errors,\ queue\ depth,\ latency,\ temperature]
\]

A Transformer can learn interactions across a longer context than a small recurrent state, subject to context and compute limits.

### Protocol traces

Ordered SoundWire, PCIe or DMA events can be represented as event tokens with type, timestamp, channel, register or state metadata. The model can learn patterns across events, but it must be evaluated against deterministic protocol rules and known-good traces.

Do not serialize arbitrary register dumps into a language prompt and treat the result as a verified hardware diagnosis. Keep structured parsing and protocol invariants deterministic.

## 11. New reusable agent architecture — A26

**Transformer Training and Inference Governance Gate**

~~~mermaid
flowchart TD
    A[Engineer objective] --> B[Read-only intake agent]
    B --> C[Typed token and sequence contract]
    C --> D[Dataset / tokenizer / split validator]
    D --> E[Deterministic representation pipeline]
    E --> F{Model task router}
    F --> G[Encoder experiment worker]
    F --> H[Decoder-only experiment worker]
    F --> I[Encoder-decoder worker]
    G --> J[Training runtime]
    H --> J
    I --> J
    J --> K[Mask and gradient verifier]
    K --> L[Validation and robustness evaluator]
    L --> M{Acceptance policy}
    M -->|Reject| N[Bounded retry or report]
    M -->|Accept| O[Frozen candidate]
    O --> P[Protected test gate]
    P --> Q[Human approval]
    Q --> R[Model registry]
    R --> S[Inference service]
    S --> T[Context and KV-cache manager]
~~~

### Pattern mapping

| Agent pattern | Application |
|---|---|
| Deterministic workflow | Token validation → masking → training → evaluation → release |
| Single agent + typed tools | Read-only inspection of tokenizer, traces, metrics and model metadata |
| Router-specialist | Route encoder, decoder-only, encoder-decoder, audio or telemetry tasks |
| Planner–executor–verifier | Plan bounded architecture experiments and independently verify masks/metrics |
| Supervisor–worker | Parallel seeds, model configurations or dataset cohorts |
| Event-driven | Queue training jobs, checkpoint events and inference requests |
| Human-in-the-loop | Approve model release, context policies and consequential diagnostic use |

### Production engineering contract

**Typed boundaries**
- Tokenizer ID and vocabulary hash.
- Special-token IDs (BOS, EOS, PAD and task-specific markers).
- Sequence length, truncation and padding policy.
- Input tensor shape, dtype and device.
- Attention-mask semantics and positional-encoding version.
- Model architecture/configuration and checkpoint version.

**State and memory**
- Keep user conversation memory separate from numerical KV cache.
- Scope KV cache to request/session, model version and security principal.
- Expire cache on session end, model swap or policy boundary.
- Never reuse cache across tenants or unrelated users.
- Record context truncation and cache eviction.

**Security**
- Treat retrieved documents, logs and tokenized content as untrusted data.
- Enforce tool permissions outside the model.
- Do not interpret attention weights as authorization or proof of reasoning.
- Test prompt-injection and cross-tenant context isolation.
- Keep hardware writes, resets and firmware actions behind separate deterministic approval gates.

**Evaluation**
- Verify causal masks with future-token perturbation tests.
- Test padding and variable-length behavior.
- Evaluate held-out documents/devices/time periods.
- Track next-token loss/perplexity where appropriate, task quality, hallucination and safety metrics.
- Test long-context degradation and context truncation.

**Reliability and observability**
- Version tokenizer, positional method, model and preprocessing together.
- Make training jobs idempotent and checkpoint recoverable.
- Track tokens/sec, time-to-first-token, inter-token latency, p50/p95, GPU memory, KV-cache occupancy and error rates.
- Trace model version, tokenizer hash, request ID and policy decision; redact sensitive prompt content.
- Bound context length, output tokens, retries, concurrency and compute budgets.

**Cost and latency**
- Use batching for throughput where latency permits.
- Use KV caching for autoregressive generation.
- Evaluate grouped-query attention and efficient SDPA for memory/throughput trade-offs.
- Route simple deterministic tasks to tools or smaller models.
- Invoke an LLM only when language understanding or synthesis adds value.

**Release rule:** an agent may propose experiments and explain evidence. Deterministic evaluation and independent verification decide whether acceptance criteria pass; a human approves production promotion.

## 12. Five review questions

1. What do Q, K and V represent, and why are they separate learned projections?
2. Why is the dot product divided by the square root of the key dimension?
3. What is the difference between a causal mask and a padding mask?
4. Why does a Transformer need positional information?
5. Why does autoregressive generation still proceed token by token even though Transformer training can parallelize across positions?

## 13. Recommended references

1. Vaswani et al., [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762).
2. [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
3. [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
4. [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
5. [PyTorch TransformerDecoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html).
6. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)](https://arxiv.org/abs/2205.14135).
7. [Stanford CS25 — Transformers United](https://web.stanford.edu/class/cs25/).

## 14. Next session — Session 021

**Tokenization, Embeddings and Next-Token Language Modeling**

- Why text must become token IDs.
- Characters, bytes, words and subword tokenization.
- BPE and byte-level BPE intuition.
- Vocabulary, special tokens and tokenizer contracts.
- Embedding matrices and embedding lookup.
- Next-token prediction and teacher forcing.
- Causal language-model loss and shifted labels.
- Perplexity and its limitations.
- Build a tiny character/subword language model.
- Connect tokenizer and embedding design to context length, memory and inference cost.

**Course milestone:** 20 sequential lessons published. Phase 5 has begun. The next session moves from Transformer architecture into the text representation and training objective that make a language model work.
