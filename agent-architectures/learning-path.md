# AI Agent Architecture Learning Path

This track is introduced after the foundations and LLM application concepts. Each module should include an architecture diagram, a runnable reference implementation, failure-mode tests, a security review and cost/latency measurements.

| Sequence | Pattern / engineering topic | Main design question | Expected artifact |
|---|---|---|---|
| A1 | Deterministic workflow + model steps | Which steps are fixed and which need model judgment? | State-machine workflow with typed model adapter |
| A2 | Single agent + typed tools | How does one agent safely call a small tool set? | Tool schemas, allow-list, bounded loop |
| A3 | Router + specialists | How should a task be routed to a domain expert? | Router with fallback and routing evaluation |
| A4 | Planner–executor–verifier | How do we plan, act and independently check? | Durable plan, executor and verifier |
| A5 | Supervisor + workers | Which subtasks can safely run in parallel? | Supervisor, worker contracts and aggregation |
| A6 | Event-driven agents | How do events start work without duplicate side effects? | Queue consumer, idempotency and dead-letter flow |
| A7 | Human-in-the-loop | Which actions require approval and how is approval audited? | Approval state, reviewer UI/API and audit trail |
| A8 | Embedding retrieval and specialist routing | How can vector similarity rank evidence or suggest a specialist without becoming an authorization decision? | Versioned embedding contract, tenant-scoped retrieval and evaluated router with fallback |
| A9 | Entropy-aware routing with abstention | How can predictive uncertainty trigger a safe fallback without becoming an authorization signal? | Entropy, margin and OOD policy with human escalation and evaluation |
| A10 | State, memory and context | What belongs in working state, episodic memory or retrieval? | Versioned state schema and retrieval policy |
| A11 | Security and tool boundaries | How are identity, permissions and untrusted content handled? | Threat model, tool policy and injection tests |
| A12 | Evaluation and observability | How do we know the agent is reliable and improving? | Golden set, traces, metrics and regression gate |
| A13 | Reliability and cost/latency | How do we bound failures and optimize the whole workflow? | Budgets, retries, checkpointing and performance report |
| A14 | ML experiment governance agent | How can an agent assist ML experiments without controlling data splits, test access or model promotion? | Problem contract, immutable split manifest, validation workflow, test gate and approval record |
| A15 | Objective-aware model selection gate | How can a system select regression/classification workflows and enforce objective-specific evaluation without exposing test labels to development? | Typed prediction contract, deterministic task router, metric contract, validation threshold and frozen-candidate test gate |
| A16 | Tree ensemble experiment gate | How can tree/forest candidates be compared reproducibly while protecting split boundaries and preventing feature-importance claims from becoming causal claims? | Immutable split manifest, bounded training workers, validation metric/importance audit, frozen test gate and human approval |
| A17 | Classical classifier selection gate | How can KNN, Naive Bayes and SVM candidates be compared with correct scaling, calibration checks and a protected final test set? | Typed candidate contracts, leakage-safe pipelines, validation-only selection, frozen test gate and human approval |
| A18 | Unsupervised discovery and anomaly triage gate | How can unlabeled telemetry be grouped and unusual observations triaged without treating clusters/noise as fault truth? | Versioned feature transform, bounded K-Means/hierarchical/DBSCAN candidates, population-aware metrics, stability checks and human interpretation |
| A19 | PCA representation and compression governance gate | How can a low-dimensional representation be fitted and deployed without leakage, scaler mismatch or loss of safety-critical signals? | Versioned scaler/PCA artifact, solver policy, explained-variance/reconstruction evaluation, subspace stability and human approval |
| A20 | Anomaly detection and human triage gate | How can unusual telemetry be surfaced without treating a detector score as a confirmed fault or authorization to act? | Nominal reference contract, versioned transforms, Isolation Forest/LOF/One-Class SVM workers, base-rate-aware threshold policy, idempotent cases and human approval |
| A21 | Time-series forecasting and change-point monitoring gate | How can ordered telemetry be forecast and sustained changes detected without temporal leakage or turning an alert into a diagnosis? | Event-time contract, durable windows, versioned lag features/model/residual baseline, temporal backtests, CUSUM/EWMA policy, idempotent cases and human review |
| A22 | Deep-learning training and release gate | How can an assistant help plan and explain neural-network experiments while deterministic systems own training, data boundaries, evaluation and release? | Typed experiment contract, immutable split manifest, deterministic training runtime, gradient/metric verifier, checkpoint lineage, budget controls and human promotion approval |
| A23 | PyTorch training runtime and GPU-aware execution gate | How can an assistant help configure and inspect PyTorch jobs while deterministic services own autograd training, device quotas, checkpoint recovery and model promotion? | Typed experiment contract, CPU/GPU controller, DataLoader/training worker, full checkpoint manifest, profiler metrics, idempotent job events and human release approval |
| A24 | CNN and signal-representation training gate | How can image, waveform and spectrogram CNN experiments preserve acquisition/preprocessing contracts and remain reproducible from training through release? | Typed modality contract, versioned representation pipeline, 1D/2D CNN workers, robustness evaluation, artifact lineage and human promotion approval |
| A25 | Sequence modeling and stateful inference governance gate | How can recurrent models learn ordered patterns while streaming hidden/cell state remains isolated, versioned, recoverable and safe? | Typed sequence/event contract, RNN/LSTM/GRU workers, BPTT/gradient policy, temporal/device-separated evaluation, per-stream state store, idempotent event processing and human release approval |

## Architecture selection rules

- Prefer a deterministic workflow when the sequence is known.
- Use a single agent before introducing multiple agents.
- Add specialists only when their responsibilities, context or permissions genuinely differ.
- Use planner–executor–verifier when work is multi-step and output correctness can be checked.
- Use parallel workers only for independent subtasks with explicit merge rules.
- Use event-driven execution for asynchronous work; make side effects idempotent.
- Put human approval before irreversible or high-impact operations.
- For unsupervised discovery, never interpret cluster IDs or density noise as semantic labels without independent evidence and human review.
- For PCA, version the scaler, feature order and component basis together; fit only on the declared reference/training window and validate downstream signal retention before production use.
- For anomaly detection, distinguish contaminated-data outlier detection from clean-reference novelty detection; never equate a noise point or negative score with a confirmed fault. Validate thresholds at realistic base rates and route consequential decisions through human review.
- For time-series systems, preserve event-time ordering, construct features only from data available at the forecast origin, validate with rolling-origin splits, and make stream/window state idempotent. A forecast residual or change-point alert is evidence for triage, not a root-cause diagnosis.
- For deep learning, keep tensor computation and gradient updates in deterministic, versioned training code. Persist compatible checkpoints and split manifests, verify gradients/finite values/artifact lineage, and require independent evaluation plus human approval before registry promotion.
- For PyTorch training, keep model/data/device placement explicit, distinguish training/evaluation/autograd modes, checkpoint optimizer and RNG/sampler state for recovery, and benchmark complete pipelines with CUDA synchronization. Never let an agent grant itself GPU quota, load untrusted checkpoints or promote a model.
- For CNNs, version acquisition metadata and preprocessing with the model, preserve tensor-shape contracts, evaluate realistic shifts/noise and channel failures, and benchmark 1D/2D candidates against simpler baselines. Keep the LLM out of the training and release authority path.
- For recurrent models, isolate hidden/cell state by tenant, device, session, stream and model version; define reset, expiry, replay and late-event rules. Record truncated-BPTT boundaries, clip/monitor gradients, test temporal/device leakage, and never interpret recurrent correlation as root-cause proof.

## Companion implementations

- The [budget-aware agent execution controller](patterns/budget-aware-agent-execution.md) extends A13. It places deterministic budget reservations, policy checks, bounded retries, deadline propagation and verification around model/tool calls. It is a runtime resource controller, not a gradient-based neural-network optimizer.
- The [ML Experiment Governance Agent](patterns/ml-experiment-governance-agent.md) extends A14. It uses deterministic split manifests, pipeline isolation, a frozen-candidate test gate, immutable artifact lineage and human approval.
- The [Objective-Aware Model Selection Gate](patterns/objective-aware-model-selection-gate.md) extends A15. It routes validated problem contracts to regression or classification, applies the right metric contract, selects thresholds on validation data and protects the final test-set boundary.
- The [Tree Ensemble Experiment Gate](patterns/tree-ensemble-experiment-gate.md) extends A16. It compares tree/forest candidates with immutable splits, bounded workers, validation-only importance analysis, a frozen-candidate test gate and human approval.
- The [Classical Classifier Selection Gate](patterns/classical-classifier-selection-gate.md) extends A17. It compares KNN, Naive Bayes and SVM using leakage-safe preprocessing, validation-only selection, calibration checks and a protected final test set.
- The [Unsupervised Discovery and Anomaly Triage Gate](patterns/unsupervised-discovery-anomaly-triage.md) extends A18. It compares clustering candidates with versioned feature transforms, population-aware metrics, noise semantics, stability checks and human interpretation.
- The [PCA Representation and Compression Governance Gate](patterns/pca-representation-compression-governance.md) extends A19. It versions preprocessing and principal components together, routes solvers by matrix shape and resource limits, checks reconstruction/downstream utility and requires approval before production schema or alert-policy changes.
- The [Anomaly Detection and Human Triage Gate](patterns/anomaly-detection-human-triage-gate.md) extends A20. It separates novelty scoring from diagnosis, versions the reference population and transforms, enforces base-rate-aware alert policies, creates idempotent cases and requires human review before high-impact actions.
- The [Time-Series Forecasting and Change-Point Monitoring Gate](patterns/time-series-forecasting-change-point-gate.md) extends A21. It combines event-time ordering, durable windows, leakage-safe lag features, temporal backtesting, forecast residuals, CUSUM/EWMA policies, idempotent investigations and human review.
- The [Deep-Learning Training and Release Gate](patterns/deep-learning-training-experiment-gate.md) extends A22. It separates LLM-assisted experiment planning/explanation from deterministic tensor training, checkpointing, evaluation and registry promotion, with typed contracts, immutable data splits, resource budgets and human approval.
- The [PyTorch Training Runtime and GPU-Aware Execution Gate](patterns/pytorch-training-runtime-gpu-execution-gate.md) extends A23. It separates read-only agent assistance from deterministic PyTorch training, CPU/GPU scheduling, DataLoader execution, complete checkpoint recovery, profiling, validation/test boundaries and human registry approval.
- The [CNN and Signal-Representation Training Gate](patterns/cnn-signal-representation-training-gate.md) extends A24. It routes validated image, waveform and spectrogram contracts to approved 1D/2D CNN paths, versions acquisition and preprocessing metadata with the model, evaluates robustness, and keeps training, test access and release under deterministic services and human approval.
- The [Sequence Modeling and Stateful Inference Governance Gate](patterns/sequence-modeling-stateful-inference-governance.md) extends A25. It separates LLM-assisted experiment planning and evidence summaries from deterministic RNN/LSTM/GRU training and inference, versions sequence and acquisition contracts, isolates per-stream hidden/cell state, handles replay and late events idempotently, evaluates with temporal/device splits, and requires independent verification plus human release approval.

## Required engineering contract for every pattern

Every implementation must document:
- Inputs, outputs and typed tool contracts
- Component ownership and state transitions
- Model and prompt versions
- Permission scopes and side-effect classification
- Timeouts, retries, budgets, cancellation and recovery
- Memory and data retention boundaries
- Threat model and prompt-injection tests
- Evaluation cases and measurable acceptance criteria
- Trace fields, metrics, redaction and audit events
- End-to-end latency and cost per successful task

Do not assume a multi-agent design is more capable. Compare it against a simpler baseline on the same evaluation set.
