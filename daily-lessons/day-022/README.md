# Session 022 — Causal Language Model Pretraining, Sampling and Text Generation

**Phase 5 · Transformers and Generative AI · 60-minute lesson**

> **Continuity:** Session 020 derived attention; Session 021 covered tokenization, embeddings, shifted next-token targets and perplexity. This session turns those pieces into a complete small GPT-style training and generation workflow. The lab uses a character vocabulary so the training mechanics remain inspectable; production subword tokenizer contracts are covered in Session 021.

## Learning objectives

By the end of this lesson, you will be able to:

1. Explain the stages of a causal language-model pretraining pipeline.
2. Build fixed-length training examples with causal next-token targets.
3. Explain document packing, EOS boundaries and data leakage risks.
4. Configure AdamW, learning-rate warmup and cosine decay.
5. Calculate effective batch size when using gradient accumulation.
6. Explain mixed precision, loss scaling and gradient clipping.
7. Save and restore a training checkpoint with enough state to resume.
8. Compare greedy decoding, temperature, top-k and top-p sampling.
9. Add stop conditions and context-window handling to generation.
10. Design a production-minded, agent-assisted pretraining and generation workflow.

## 60-minute schedule

| Time | Topic |
|---|---|
| 0–5 min | Recap: token IDs → embeddings → next-token objective |
| 5–15 min | Corpus preparation, sequence packing and causal labels |
| 15–25 min | Training loop, AdamW, warmup and gradient accumulation |
| 25–33 min | Mixed precision, clipping and checkpoints |
| 33–40 min | Greedy, temperature, top-k and top-p decoding |
| 40–53 min | Train and sample from a tiny GPT in PyTorch |
| 53–60 min | Production architecture, review and next steps |

## 1. From a language-model objective to a training system

A causal language model estimates:

\[
P(x_t\mid x_0,\ldots,x_{t-1})
\]

For a token sequence [BOS, dma, retry, EOS], the model receives the prefix and learns the next token:

| Input position | Context presented | Target |
|---:|---|---|
| 0 | BOS | dma |
| 1 | BOS, dma | retry |
| 2 | BOS, dma, retry | EOS |

The input and target tensors are shifted by one:

~~~python
input_ids = tokens[:, :-1]
target_ids = tokens[:, 1:]
~~~

The model emits vocabulary logits with shape [batch, time, vocabulary]. Cross-entropy is calculated against the shifted target IDs. Padding targets, if present, must be excluded from the loss.

The important distinction is that **the Transformer architecture is only one component**. A useful language model also needs a well-governed corpus, tokenizer, training schedule, checkpoint system, evaluation procedure and inference decoder.

## 2. Corpus preparation and sequence packing

A practical pretraining pipeline is:

~~~text
Raw documents
    |
    v
License / privacy / quality policy
    |
    v
Deduplication and source-level split
    |
    v
Version-matched tokenizer
    |
    v
Document boundaries + EOS
    |
    v
Fixed-length token blocks
    |
    v
Causal next-token training
~~~

### Split before packing

Split at the appropriate source boundary **before** creating windows or packed token streams. Depending on the use case, group by document, conversation, repository, device, customer or time. Near-duplicate content across splits can make held-out metrics misleading.

### Packing documents

Token streams are often packed into fixed-length blocks to reduce padding. An EOS marker can separate documents:

~~~text
doc-A tokens ... EOS doc-B tokens ... EOS
~~~

Packing is not automatically safe. Decide whether attention is allowed to cross document boundaries. A simple concatenated stream lets later tokens attend to earlier documents in the same block. If that is not intended, use block-diagonal/document-aware masks or keep examples separate. Never let a training window cross into validation or test data.

### Context length

For block length \(T\), the model sees at most T tokens in one training example. Increasing T can improve access to longer dependencies but raises attention activation and memory costs. It also changes the number of windows and the effective data mixture.

**Worked example:** a 1,024-token block produces 1,023 next-token targets when using a one-token shift. With batch size 8 and 4 gradient-accumulation microbatches, one optimizer update processes \(8\times1,024\times4=32,768\) input-token positions (approximately the same number of targets, minus the first-position convention).

## 3. Training loop and optimizer

A standard update is:

1. Read a batch of token blocks.
2. Run the model forward pass.
3. Calculate causal cross-entropy.
4. Backpropagate gradients.
5. Clip gradients if configured.
6. Update parameters with the optimizer.
7. Advance the learning-rate schedule only when an optimizer update actually occurs.

### AdamW

AdamW maintains moving averages of gradients and squared gradients, with decoupled weight decay. It is a common starting optimizer for Transformer training.

The learning rate controls the update scale. Too high can destabilize training; too low can make progress very slow.

### Warmup and cosine decay

A warmup schedule starts with a small learning rate and increases it during the first training steps. It can reduce instability at the beginning of training.

After warmup, cosine decay gradually reduces the learning rate:

\[
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\left(\pi\frac{t-t_w}{T-t_w}\right)\right)
\]

where \(t_w\) is the warmup boundary and T is the planned total update count. This is one schedule, not a universal rule. Always record the schedule and actual optimizer-step count.

### Gradient accumulation

If a GPU cannot fit the desired batch, split it into smaller microbatches and accumulate gradients before stepping.

\[
B_{effective}=B_{micro}\times N_{accum}\times N_{devices}
\]

For a single device, microbatch 8 and accumulation 4 gives an effective batch of 32 sequences. Divide each microbatch loss by the accumulation count before backward so the accumulated gradient is an average rather than a sum.

Gradient accumulation reduces activation memory for each microbatch, but it does not make the optimizer see more independent data than was actually sampled. It also adds microbatch-loop overhead.

## 4. Mixed precision and gradient clipping

Mixed precision uses lower-precision arithmetic for suitable operations while retaining higher precision where needed.

- FP16 has a narrower exponent range; gradient scaling is commonly used to reduce underflow.
- BF16 has a wider exponent range and often does not require gradient scaling, but hardware support matters.
- FP32 remains a useful baseline and fallback.

A typical CUDA FP16 pattern is:

~~~python
scaler = torch.amp.GradScaler("cuda")

with torch.autocast("cuda", dtype=torch.float16):
    logits, loss = model(x, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
~~~

Unscale before clipping so the threshold is applied to real gradients. Do not assume AMP is always faster; benchmark the actual model, sequence length and hardware.

Gradient clipping limits unusually large gradient norms:

\[
g'=g\min\left(1,\frac{\tau}{\|g\|}\right)
\]

It is a stability measure, not a fix for incorrect data, loss or model code.

## 5. Checkpointing and restartability

A resumable training checkpoint should include more than model weights:

- Model state dictionary.
- Optimizer state.
- Learning-rate scheduler state.
- AMP GradScaler state, if used.
- Global optimizer step and epoch.
- RNG states (Python, NumPy, CPU and accelerator).
- Sampler or data-loader state / data cursor.
- Tokenizer revision and corpus/split manifest hashes.
- Model and training configuration.
- Software versions and run identifier.

If only weights are restored, AdamW moments and the learning-rate schedule may be lost. A resumed run can then diverge from the original trajectory.

Write checkpoints atomically (temporary file then rename), validate artifact hashes, and load only trusted artifacts. For untrusted model files, do not enable unrestricted pickle loading.

The lab performs a small model/optimizer/scheduler checkpoint round-trip and verifies identical logits after restoring the model. A production checkpoint must preserve the full state listed above.

## 6. Text generation and decoding

After the prompt is tokenized, the model produces logits for the next token. A decoding policy converts those logits into a selected token. Append it and repeat until a stop condition is met.

### Greedy decoding

\[
x_{next}=\arg\max_i logits_i
\]

Always selects the highest-scoring token. It is deterministic (for a fixed model and runtime) and useful for simple structured outputs, but can produce repetitive or dull continuations.

### Temperature

\[
p_i=softmax(logits_i/\tau)
\]

- \(\tau<1\): sharper distribution, less random.
- \(\tau=1\): unchanged distribution.
- \(\tau>1\): flatter distribution, more random.
- \(\tau\to0^+\): approaches greedy selection.

Temperature changes randomness, not model knowledge. A high temperature does not make a model more intelligent.

### Top-k sampling

Keep only the k highest-scoring tokens, renormalize their probabilities and sample from them. A fixed candidate count is simple but may be too restrictive for some distributions and too permissive for others.

### Top-p (nucleus) sampling

Sort tokens by probability and keep the smallest set whose cumulative probability reaches p. The candidate set adapts to distribution sharpness.

Top-k and top-p are often combined. Apply filters to logits before softmax sampling. Implementations differ in boundary details; test the exact behavior.

### Stop and safety conditions

Generation should stop on:
- EOS or an approved stop sequence.
- Maximum new-token budget.
- Context-window limit.
- Request deadline or cancellation.
- Policy-defined output constraints.

For a fixed context window, crop or reject an overlong prompt according to an explicit policy. Never silently discard the system/developer instructions or critical evidence. For a real serving system, use the model's exact chat template and tokenizer, not an ad hoc concatenation.

## 7. Worked decoding example

Suppose the next-token logits are:

\[
[2.0,\;1.0,\;0.1,\;-1.0]
\]

Greedy selects token 0. With temperature 2, the logits are divided by 2 before softmax, producing a flatter probability distribution. Top-k with k=2 removes tokens 2 and 3 before sampling. Top-p then retains the smallest high-probability prefix whose cumulative mass reaches the chosen threshold.

The same model can therefore produce different continuations under different decoding settings. Evaluation must record the decoding configuration and random seed.

## 8. Hands-on lab: train a tiny GPT and generate text

The runnable lab.py builds a small decoder-only Transformer in PyTorch and trains it on a synthetic diagnostic-style corpus. It uses a character codec deliberately so Session 022 can focus on training and decoding mechanics; Session 021 covers subword tokenizers and their compatibility contracts.

### Lab contents

- Deterministic document generation and document-level train/validation/test split.
- Character vocabulary fitted on training documents only.
- Fixed-length next-token batches.
- Causal self-attention using PyTorch scaled-dot-product attention.
- Pre-LayerNorm Transformer blocks and tied token/output embeddings.
- AdamW, linear warmup followed by cosine decay.
- Gradient accumulation and norm clipping.
- Optional CUDA FP16 autocast and GradScaler; CPU uses FP32.
- Validation loss, held-out test loss and perplexity.
- Greedy, temperature, top-k and top-p decoding.
- A checkpoint round-trip test.

### Reference CPU run

Environment: PyTorch 2.10.0 CPU. Seed: 22022.

| Measurement | Result |
|---|---:|
| Training / validation / test documents | 192 / 24 / 24 |
| Character vocabulary | 37 |
| Training stream characters | 10,982 |
| Model parameters | 106,560 |
| Optimizer updates | 220 |
| Microbatch / accumulation | 32 / 2 |
| Effective sequences per update | 64 |
| Training loss before → after | 3.6245 → 0.3093 |
| Validation loss before → after | 3.6226 → 0.3421 |
| Test loss | 0.3429 |
| Test perplexity | 1.4090 |
| Restored-checkpoint max logit difference | 0.0 |

The corpus is small, synthetic and repetitive. These results demonstrate that the code path runs and the model can learn the toy distribution; they are not a benchmark of general language ability or a production pretraining result.

Greedy and sampled decoding both returned bounded continuations. The sample quality is intentionally not scored: a tiny character model trained on a narrow corpus can repeat patterns and produce incomplete text.

### Run locally

~~~bash
cd daily-lessons/day-022
python -m pip install -r requirements.txt
python lab.py
python lab.py --device cpu
python lab.py --device cuda
~~~

The CUDA path is implemented but was not hardware-tested in the reference run. Use a CUDA-enabled PyTorch build and a compatible GPU for that path.

### Exercises

1. Increase context length from 64 to 128 and record memory and step time.
2. Compare microbatch 16 × accumulation 4 against microbatch 32 × accumulation 2.
3. Disable gradient clipping and inspect gradient norms and loss stability.
4. Change warmup from 20 to 0 steps and compare early training.
5. Compare greedy, temperature 0.7, temperature 1.2, top-k 8 and top-p 0.9 using fixed seeds.
6. Add an EOS token and stop generation on EOS.
7. Save and restore the full optimizer/scheduler/RNG/data-cursor state, then compare the next update against an uninterrupted run.
8. Replace the character codec with the versioned subword tokenizer from Session 021; preserve tokenizer/model compatibility and special-token handling.

## 9. A28 — Causal LM Pretraining and Generation Governance Gate

### Architecture

~~~mermaid
flowchart TD
    A[Training or generation request] --> B[Read-only intake agent]
    B --> C[Typed run contract and policy]
    C --> D[Corpus and split validator]
    D --> E[Tokenizer/model compatibility gate]
    E --> F[Deterministic packer]
    F --> G[Pretraining controller]
    G --> H[GPU/CPU training worker]
    H --> I[Checkpoint and metrics store]
    I --> J[Independent evaluator]
    J --> K{Validation policy}
    K -->|Reject| L[Bounded retry or stop]
    K -->|Accept| M[Frozen candidate]
    M --> N[Protected test gate]
    N --> O[Human approval]
    O --> P[Model registry]
    P --> Q[Versioned inference service]
    Q --> R[Decoding policy and stop controller]
~~~

### Component boundaries

| Component | Owns | Does not own |
|---|---|---|
| Corpus service | Source allow-list, licensing/PII policy, dedup and manifests | Model release |
| Tokenizer contract | Vocabulary, normalization, special IDs and hash | Data authorization |
| Packer | EOS boundaries, context blocks, split isolation | Test-label changes |
| Training controller | Optimizer, schedule, accumulation, AMP, clipping and checkpointing | Human approval |
| Evaluator | Held-out loss, perplexity, regression and generation tests | Training data mutation |
| Decoder | Sampling, stop sequences, context and output budgets | Tool permissions |
| Agent assistant | Explain runs and propose bounded experiments | Arbitrary shell, secrets or release authority |

### Seven reusable agent patterns

| Pattern | Use in this workflow | Trade-off |
|---|---|---|
| Deterministic workflow | Corpus → tokenizer → packer → train → evaluate → release | Reproducible, less flexible |
| Single agent + typed tools | Read run status, metrics and approved sample outputs | Simple; limited specialization |
| Router-specialist | Route corpus, tokenizer, training, evaluation and serving questions | Focused; requires abstain/fallback |
| Planner–executor–verifier | Plan bounded experiments, execute and independently verify | Better control; more latency/state |
| Supervisor-worker | Parallel seeds, sequence lengths or bounded hyperparameter trials | Faster exploration; higher compute and aggregation complexity |
| Event-driven | Queue jobs, checkpoint events and evaluation results | Scalable; needs durable state/idempotency |
| Human-in-the-loop | Approve corpus exceptions, expensive runs and model promotion | Safer; adds review latency |

Use the simplest pattern that meets the requirement. Parallel workers must use immutable manifests, isolated output paths and explicit merge/selection rules.

### Typed tool contracts (illustrative)

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class PretrainRequest:
    corpus_manifest_sha256: str
    tokenizer_revision: str
    model_config_revision: str
    context_length: int
    micro_batch_size: int
    accumulation_steps: int
    max_optimizer_steps: int
    seed: int

@dataclass(frozen=True)
class GenerateRequest:
    model_revision: str
    tokenizer_revision: str
    prompt_ref: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float
~~~

Validate bounds and artifact compatibility in deterministic services. A model-generated plan is untrusted data, not executable authority.

### State and memory

Persist:
- Immutable corpus/source/license and split manifests.
- Tokenizer artifact hash and special-token map.
- Model config, context length and positional method.
- Optimizer, scheduler, scaler, RNG, sampler/data cursor and global step.
- Checkpoint hashes, evaluation results and approval record.
- Generation configuration, stop reason and request trace.

Keep agent conversation memory, training state and inference KV cache separate. Scope KV cache by tenant, authorized request/session and model revision. Never reuse cached state after model, tokenizer or authorization changes.

### Security

- Treat corpus documents as untrusted training data, not instructions to the training agent.
- Enforce licensing, PII, secrets and retention policies before ingestion.
- Sandbox data processing and training; deny arbitrary shell/network access by default.
- Use least-privilege, tenant-scoped dataset and artifact access.
- Do not expose protected test corpora or labels to planner/worker agents.
- Load only trusted checkpoints; verify artifact hashes and use safe serialization.
- Treat prompts and generated text as untrusted when passed to downstream tools.
- Keep hardware writes, firmware flashing, resets and model promotion outside the LLM's authority.

### Evaluation and reliability

Required checks:
- Document-level split isolation, dedup and source contamination checks.
- Tokenizer/model/special-token compatibility.
- Correct shifted targets and causal-mask behavior.
- EOS and padding handling.
- Finite loss/gradients and gradient-norm monitoring.
- Held-out token NLL/perplexity with corpus/tokenizer metadata.
- Generation regression tests: repetition, stop behavior, length, format and domain coverage.
- Checkpoint restore and deterministic evaluation.
- Human review for consequential model releases.

A lower perplexity does not prove factuality, reasoning, safety or diagnostic correctness. Add task-specific evaluations before using the model as an assistant.

Use immutable job IDs, idempotent submissions, atomic checkpoints, bounded retries, deadlines, cancellation and recovery. Retry transient infrastructure failures, not deterministic data or model-contract errors.

### Observability, cost and latency

Track:
- Documents, bytes and tokens ingested.
- Training tokens/sec and optimizer updates/sec.
- Loss, validation loss, gradient norm and learning rate.
- GPU utilization, peak allocated/reserved memory and data-loader wait.
- Checkpoint size, age, save duration and resume count.
- Time-to-first-token, inter-token latency and generated tokens/sec.
- Prompt/output tokens, truncation and stop reason.
- Cost per million training tokens, per generated token and per successful task.

Budget context length, output tokens, concurrency, GPU hours and retry count. Longer context increases attention/KV-cache cost. Gradient accumulation changes effective batch size but does not remove total compute. Mixed precision should be enabled only after correctness and performance checks.

**Release rule:** the LLM may explain experiments and suggest settings; deterministic systems own data access, training, evaluation and decoding limits. Independent verification and human approval control production promotion.

## Five review questions

1. Why must document/source splits happen before packing token windows?
2. How does gradient accumulation change effective batch size, and why should the loss be divided by the accumulation count?
3. What is the difference between temperature, top-k and top-p sampling?
4. Why should a resumable AdamW checkpoint include optimizer, scheduler, RNG and sampler/data-cursor state?
5. Why is a lower validation perplexity not sufficient evidence that a model is safe or useful for silicon diagnostics?

## Recommended references

1. [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — causal LM training and generation.
2. [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) — greedy, sampling, temperature, top-k and top-p.
3. [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html) — autocast and GradScaler.
4. [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) — clipping and gradient accumulation.
5. [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — decoupled weight decay.
6. [PyTorch CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) — cosine learning-rate decay.
7. [PyTorch scaled-dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — causal attention implementation.
8. [nanoGPT](https://github.com/karpathy/nanoGPT) — compact educational GPT training reference; inspect version and license before reuse.

## Next lesson — Session 023

**Instruction Tuning, Supervised Fine-Tuning and Chat Templates**

- Pretraining versus instruction tuning.
- Instruction/response dataset construction.
- Chat message roles and templates.
- Assistant-only loss masking.
- EOS and turn boundaries.
- Supervised fine-tuning (SFT).
- Catastrophic forgetting and data quality.
- LoRA preview.
- Build a small instruction-tuning pipeline.
- Evaluate instruction following and format adherence.

**Milestone:** 22 sequential lessons published. The course now covers the complete toy causal-LM path from corpus and tokenizer through training, checkpointing and decoding.
