# A27 — Tokenizer and Language-Model Training Governance Gate

**Purpose:** Govern the path from an approved text corpus through a versioned tokenizer and causal language-model training to an independently evaluated, human-approved model artifact.

This pattern extends the Transformer governance work in A26. It focuses on the interfaces before and after attention: corpus policy, tokenizer identity, token IDs, embeddings, shifted targets, causal loss, perplexity and generation compatibility.

## When to use

Use this gate when:
- Training a new causal language model or adapting an existing one.
- Changing vocabulary, BPE merges, normalization, special tokens or chat templates.
- Building a domain tokenizer for technical logs, code, multilingual text or audio transcripts.
- Comparing tokenizer/model combinations.
- Promoting a model into a serving environment where token counts affect context, latency and cost.

Do not use an LLM agent as the tokenizer implementation, corpus-policy engine, gradient trainer, evaluator of record or model-release authority.

## Architecture

~~~mermaid
flowchart TD
    A[Training objective] --> B[Read-only intake agent]
    B --> C[Typed corpus and tokenizer contract]
    C --> D[Corpus policy, dedup and split validator]
    D --> E[Deterministic tokenizer pipeline]
    E --> F[Versioned tokenizer artifact]
    F --> G[Tokenization coverage and round-trip tests]
    G --> H[Deterministic causal-LM trainer]
    H --> I[Checkpoint and token-level metrics]
    I --> J[Independent verifier]
    J --> K[Validation evaluator]
    K --> L{Acceptance policy}
    L -->|Reject| M[Bounded retry or reject]
    L -->|Accept| N[Frozen candidate]
    N --> O[Protected test gate]
    O --> P[Human approval]
    P --> Q[Model registry]
    Q --> R[Version-matched inference service]
~~~

## Component contracts

| Component | Owns | Does not own |
|---|---|---|
| Corpus intake | Source allow-list, license/PII policy, immutable corpus manifest | Model weights |
| Tokenizer pipeline | Normalization, pre-tokenization, vocabulary, merge ranks, special IDs | Authorization policy |
| Dataset builder | Document splits, BOS/EOS policy, windows, PAD masks | Test-label changes |
| Trainer | Embeddings, Transformer, loss, optimizer, checkpoints | Release approval |
| Evaluator | Held-out token NLL, perplexity, coverage and regression tests | Rewriting training data |
| Registry | Immutable model/tokenizer pair and lineage | Unreviewed model mutation |
| Assistant | Explain reports and propose bounded experiments | Arbitrary corpus access or release authority |

## Typed tool contracts

Illustrative Python contracts:

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenizeRequest:
    text: str
    tokenizer_revision: str
    add_special_tokens: bool

@dataclass(frozen=True)
class TrainLMRequest:
    dataset_manifest_sha256: str
    tokenizer_sha256: str
    context_length: int
    batch_size: int
    max_steps: int
    seed: int
~~~

Validate bounds, revisions, tenant identity and caller permissions in deterministic services. A model-generated plan is data to validate, not executable authority.

## Seven orchestration patterns

| Pattern | Application | Trade-off |
|---|---|---|
| Deterministic workflow | Corpus validation → tokenizer → dataset → training → evaluation | Most reproducible; less flexible for ambiguous research questions |
| Single agent + typed tools | Read-only corpus statistics and experiment reports | Low orchestration overhead; limited specialization |
| Router-specialist | Route tokenizer, data-quality, training and evaluation questions | Better focus; routing needs fallback |
| Planner–executor–verifier | Plan bounded experiments, execute, independently verify | Stronger control; more latency and state |
| Supervisor-worker | Parallel tokenizer configurations or model seeds | Higher throughput; requires isolated jobs and explicit aggregation |
| Event-driven | Queue tokenization/training jobs and publish checkpoint/metric events | Scales asynchronous work; needs idempotency and durable state |
| Human-in-the-loop | Approve corpus-policy exceptions, tokenizer compatibility changes and model promotion | Safer high-impact decisions; adds review latency |

Prefer the simplest pattern that satisfies the task. Parallel workers must use immutable manifests and disjoint job IDs; they must not race to mutate a shared tokenizer or model artifact.

## State, memory and artifact identity

Persist:
- Corpus snapshot, source/license manifest and deduplication report.
- Train/validation/test split manifest.
- Tokenizer artifact, cryptographic hash, normalization and pre-tokenizer configuration.
- Vocabulary, merge ranks, special-token strings and IDs.
- Model config, tokenizer compatibility record and positional method.
- Optimizer, scheduler, RNG, sampler and training-step checkpoint state.
- Token-level evaluation report, valid-token count and approval record.

Keep conversational memory separate from training state and inference KV cache. A tokenizer revision change invalidates token-ID compatibility. Never silently pair a model checkpoint with a different vocabulary mapping.

## Security and data governance

- Treat corpus text as untrusted data, not instructions to the training agent.
- Scan for secrets, PII, licensing restrictions and disallowed content before ingestion.
- Do not execute source snippets, shell commands or links found in training documents.
- Sandbox tokenization and training jobs with resource and filesystem limits.
- Enforce least-privilege, tenant-scoped dataset access.
- Protect validation/test corpora and labels.
- Load only trusted checkpoints; verify artifact hashes and use safe serialization.
- Keep tool permissions outside the LLM and audit every side effect.

## Evaluation gates

Required deterministic checks:
1. Fixed tokenizer revision produces fixed token IDs.
2. Encode/decode and special-token behavior pass.
3. All emitted IDs are within the declared vocabulary range.
4. Vocabulary size matches embedding and output-head dimensions.
5. Input/target shift is correct.
6. Causal future-token leakage is zero within tolerance.
7. PAD targets are excluded from loss.
8. Train/validation/test sources are separated and deduplicated.
9. Held-out NLL and perplexity include corpus, tokenizer and valid-token metadata.
10. Checkpoint restoration reproduces evaluation within declared numerical tolerance.
11. Regression coverage includes code identifiers, punctuation, multilingual scripts and domain-specific strings.

Perplexity alone is insufficient evidence of a useful assistant. Add task-specific evaluation for factuality, retrieval grounding, instruction following, tool correctness and safety in later application-engineering modules.

## Reliability, observability and cost

**Reliability**
- Immutable corpus and tokenizer manifests.
- Idempotent job submission and bounded retries.
- Checkpoint recovery with optimizer/RNG/sampler state.
- Explicit cancellation and deadlines.
- Compatibility checks before loading a model/tokenizer pair.
- Human approval for corpus-policy exceptions and release.

**Observability**
- Documents, bytes and tokens ingested.
- Tokens per byte/word by language and data type.
- Unknown/fallback and truncation rates.
- Tokenizer throughput and CPU time.
- Training tokens/sec, loss, gradient norm and accelerator memory.
- Checkpoint age, resume count and failed jobs.
- Time-to-first-token, inter-token latency and generated tokens/sec.
- Cost per million training/inference tokens and per successful task.

**Cost/latency**
- Larger vocabulary increases embedding and output-projection cost.
- Longer context increases attention and KV-cache memory.
- More tokenizer pieces per source document increase training/inference sequence length.
- Tokenizer throughput can become a CPU bottleneck.
- Cap context, output tokens and concurrency; benchmark end-to-end rather than only model kernels.

## Example configuration

See [the A27 YAML example](../examples/tokenizer-language-model-experiment.yaml). It records corpus, tokenizer, model, training, evaluation and approval constraints. It is a declarative example, not a ready-to-run job scheduler configuration.

## Release rule

An LLM may explain tokenization and propose experiments. Deterministic pipelines own token IDs, masks, loss and training. Independent evaluation and human approval own model promotion.
