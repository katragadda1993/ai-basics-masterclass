# A28 — Causal LM Pretraining and Generation Governance Gate

**Status:** reusable reference architecture  
**Related lesson:** [Session 022 — Causal Language Model Pretraining, Sampling and Text Generation](../../daily-lessons/day-022/README.md)  
**Scope:** corpus-to-checkpoint training, evaluation, model registration and bounded autoregressive generation.

## Problem

A causal language model is not just a Transformer checkpoint. It is a coupled system of corpus policy, tokenizer revision, document packing, model configuration, optimizer/scheduler state, evaluation evidence and decoding policy. An LLM assistant may help plan and explain work, but must not become the authority for data access, training execution, protected evaluation or release.

## Architecture

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

## Component contracts

| Component | Owns | Must not own |
|---|---|---|
| Corpus service | Source allow-list, licensing/PII policy, deduplication and immutable manifests | Model release |
| Tokenizer contract | Vocabulary, normalization, special-token IDs, merge ranks and artifact hash | Dataset authorization |
| Packer | EOS boundaries, context blocks, split isolation and target alignment | Test-label changes |
| Training controller | Optimizer, schedule, accumulation, AMP, clipping and checkpointing | Human approval |
| Evaluator | Held-out NLL/perplexity, regression and generation tests | Training data mutation |
| Decoder | Temperature, top-k/top-p, stop sequences, context and output budgets | Tool permissions |
| Agent assistant | Explain reports and propose bounded experiments | Arbitrary shell, secrets or release authority |

## Seven reusable patterns

| Pattern | Application | Trade-off |
|---|---|---|
| Deterministic workflow | Corpus → tokenizer → packer → train → evaluate → release | Reproducible; less flexible for exploratory work |
| Single agent + typed tools | Read run status, metrics and approved sample outputs | Low orchestration overhead; limited specialization |
| Router-specialist | Route corpus, tokenizer, training, evaluation and serving questions | Focused expertise; needs abstain/fallback |
| Planner–executor–verifier | Plan bounded experiments, execute and independently verify | Stronger control; adds latency and state |
| Supervisor-worker | Parallel seeds, context lengths or bounded hyperparameter trials | Faster exploration; higher compute and merge complexity |
| Event-driven | Queue jobs, checkpoint events and evaluation results | Scales asynchronous work; needs durable state and idempotency |
| Human-in-the-loop | Approve corpus exceptions, expensive runs and model promotion | Safer high-impact decisions; adds review latency |

Prefer the simplest pattern that satisfies the task. Parallel workers require immutable manifests, isolated output paths, independent job IDs and explicit selection/merge rules. Do not allow workers to mutate one shared tokenizer or checkpoint.

## Typed tool contracts

Illustrative Python contracts:

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

Validate numeric bounds, caller identity, artifact revisions and resource quotas in deterministic services. Treat model-generated plans as untrusted data, never executable authority.

## State and memory boundaries

Persist a run manifest containing:
- Corpus/source/license manifest and deduplication report.
- Document-level train/validation/test split manifest.
- Tokenizer artifact hash, special-token map and normalization/pre-tokenization revision.
- Model config, positional method, context length and parameter count.
- Optimizer, scheduler, AMP scaler, RNG, sampler/data cursor and global optimizer step.
- Checkpoint hashes, metrics, software versions and approval record.
- Generation configuration, stop reason and request trace.

Keep three state classes separate:
1. **Agent conversation state:** user goals, tool results and approval workflow.
2. **Training state:** model/optimizer/scheduler/scaler/RNG/sampler/cursor checkpoint.
3. **Inference state:** request-scoped KV cache and decoder state.

Scope KV cache by tenant, authorized request/session and model revision. Invalidate it after model, tokenizer, position or authorization changes. A tokenizer revision mismatch is a hard compatibility error, not a warning.

## Security and data governance

- Enforce source licensing, PII, secrets and retention policy before ingestion.
- Treat corpus documents as untrusted data, not instructions to the training agent.
- Sandbox data processing and training; deny arbitrary shell/network access by default.
- Use least-privilege and tenant-scoped corpus/artifact access.
- Keep protected test corpora and labels inaccessible to planners and workers.
- Verify checkpoint hashes; load only trusted artifacts using safe serialization.
- Treat generated text as untrusted when passed to downstream tools.
- Keep hardware writes, firmware flashing, resets and model promotion outside LLM authority.

## Evaluation gates

Minimum gates:
- Document/source split isolation, deduplication and contamination checks.
- Tokenizer/model/special-token compatibility.
- EOS, padding, shifted-target and causal-mask tests.
- Finite loss/gradients and gradient-norm monitoring.
- Held-out token NLL/perplexity with corpus, tokenizer and valid-token metadata.
- Generation tests for repetition, stop behavior, output length, format and domain coverage.
- Checkpoint restore and deterministic evaluation.
- Human-reviewed task-specific quality, safety and factuality evaluations before assistant use.

Perplexity is a predictive metric, not proof of factuality, reasoning, safety or diagnostic correctness. Do not compare raw token-level perplexity across different tokenizers without a stated normalization method.

## Reliability

- Immutable run IDs and input manifests.
- Idempotent job submission and artifact writes.
- Atomic checkpoints and restore tests.
- Bounded retries for transient infrastructure errors only.
- Deadlines, cancellation, retry budgets and failure classification.
- Durable event offsets and deduplication keys.
- Frozen validation candidate before protected test evaluation.
- Human approval and auditable registry promotion.

## Observability, cost and latency

Track:
- Documents, bytes and tokens ingested; tokenization throughput.
- Training tokens/sec, optimizer updates/sec, loss and learning rate.
- Gradient norm, skipped AMP steps, GPU utilization and peak memory.
- Data-loader wait, checkpoint duration/size and resume count.
- Time-to-first-token, inter-token latency and generated tokens/sec.
- Prompt/output tokens, truncation, stop reason and cache hit rate.
- Cost per million training tokens, generated tokens and successful task.

Budget context length, output tokens, concurrency, GPU hours and retry count. Longer contexts increase attention and KV-cache costs. Gradient accumulation lowers microbatch activation memory but does not remove total compute. Mixed precision must be benchmarked on target hardware.

## Acceptance criteria

A release candidate is eligible for human review only if:
1. Corpus, split, tokenizer and model manifests are immutable and compatible.
2. All data/causal/target-alignment tests pass.
3. Training and evaluation are reproducible within declared tolerances.
4. Checkpoint restore reproduces model outputs.
5. Held-out metrics meet predeclared thresholds.
6. Generation, safety and domain-specific evaluations pass.
7. Artifact hashes and provenance are complete.
8. Resource/cost/latency budgets are met.

**Release rule:** the assistant may explain runs and propose bounded experiments. Deterministic services own data access, training, evaluation and decoding limits. Independent verification and human approval own production promotion.
