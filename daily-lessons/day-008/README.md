# Day 008 — Machine Learning Foundations: Problem Formulation, Data Splits, Features, Labels and Baselines

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 3 — Machine learning  
**Previous lesson:** [Day 007 — Optimization Algorithms](../day-007/README.md)

## Learning objectives

By the end of this session, you should be able to:

1. Turn a vague engineering question into a precise supervised-learning problem.
2. Distinguish a sample, feature, label, prediction horizon and decision.
3. Explain training, validation and test datasets and when each is used.
4. Select random, stratified, group-based or time-based splitting appropriately.
5. Identify target leakage, train/test contamination and preprocessing leakage.
6. Build a baseline before trying a more complex model.
7. Choose classification metrics that reflect class imbalance and error costs.
8. Build a reproducible scikit-learn preprocessing and model pipeline.
9. Explain how a bounded agent workflow can support ML experiments without allowing an LLM to bypass data-governance rules.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Connect optimizer knowledge to the ML lifecycle |
| 5–15 min | Formulate a prediction problem |
| 15–25 min | Features, labels and data splits |
| 25–35 min | Leakage, preprocessing and baselines |
| 35–43 min | Metrics and decision thresholds |
| 43–55 min | Python ML pipeline lab |
| 55–60 min | ML experiment agent architecture and quiz |

## 1. From an algorithm to a prediction problem

The previous sessions explained how models are optimized. Before selecting an optimizer or model, first define what the system is supposed to predict.

A useful ML problem statement specifies:

- **Unit of analysis:** what does one row represent? A device, boot, DMA transfer, audio stream or time window?
- **Prediction target:** what exact outcome should be predicted?
- **Prediction time:** when must the prediction be available?
- **Prediction horizon:** what future interval does the label refer to?
- **Available inputs:** which facts are genuinely known at prediction time?
- **Action:** what decision will consume the prediction?
- **Error costs:** what are the consequences of false positives and false negatives?
- **Success metric:** what measurable result would justify using the model?

### Worked example: predict an impending DMA fault

Weak problem statement: “Use AI to detect DMA issues.”

A more precise formulation:

> For each device telemetry window, use counters and measurements available at the end of that window to predict whether a DMA fault will occur during the next ten minutes.

| Contract field | Definition |
|---|---|
| One row | One device telemetry window |
| Features | Temperature, DMA retries, bus errors, link flaps, IP family |
| Label | A DMA fault occurs in the next ten minutes: yes/no |
| Prediction time | End of the telemetry window |
| Horizon | Next ten minutes |
| Decision | Prioritize additional diagnostics or monitoring |
| False positive | Engineer spends time investigating a healthy device |
| False negative | An impending fault is not flagged |

The label must describe a future event relative to the prediction time. If a feature contains information recorded after that time, it is not a valid feature even if it correlates strongly with the label.

This example is a teaching problem, not a validated hardware-fault predictor. Real deployment requires representative silicon data, validated labels, device-family coverage and engineering sign-off.

## 2. Samples, features and labels

Represent a tabular supervised-learning dataset as:

\[
X \in \mathbb{R}^{N \times d}, \qquad y \in \mathcal{Y}^{N}
\]

- \(N\): number of samples.
- \(d\): number of features.
- \(X\): feature matrix.
- \(y\): target or label vector.

A **sample** is one example. A **feature** is an input variable. A **label** is the target answer used during supervised training.

Examples of numerical features:
- Temperature in degrees Celsius
- Retry count
- Error count
- Measured latency
- Queue depth

Examples of categorical features:
- IP family
- Device revision
- Operating mode
- Codec family
- Test scenario

Categorical identifiers should not automatically be treated as continuous measurements. Encoding a family name as 1, 2 or 3 can accidentally imply an ordering or distance that does not exist. One-hot encoding is one common option for nominal categories.

### Feature validity test

For every proposed feature, ask:

1. Is it available at the exact time the prediction is made?
2. Is its meaning consistent across device revisions and firmware versions?
3. Can it be measured in production, not only in the training environment?
4. Does it accidentally reveal the label or a later event?
5. Is it stable enough to be useful, and is its missingness understood?

Feature usefulness is not the same as feature availability at inference time.

## 3. Training, validation and test sets

A common workflow divides labeled data into three disjoint subsets.

~~~text
All labeled data
      |
      v
  Split by the deployment scenario
      |
      +------ Training set
      |       Fit preprocessing and model parameters
      |
      +------ Validation set
      |       Compare models, tune hyperparameters and thresholds
      |
      +------ Test set
              Final unbiased estimate after choices are frozen
~~~

Typical starting proportions might be 70% training, 15% validation and 15% test. These are not universal requirements. Dataset size, class frequency, time dependence and grouping may require a different strategy.

- **Training set:** used to fit model parameters and preprocessing transformations.
- **Validation set:** used for model selection, hyperparameter tuning, threshold selection and other development decisions.
- **Test set:** held back until the development choices are frozen. It estimates performance on unseen data.

Do not repeatedly inspect test results and then modify the model. Once test results guide development, the test set has become part of the development process. Obtain a new untouched holdout for a fresh final estimate.

For small datasets, cross-validation can use data more efficiently for model selection. Keep a final holdout if an independent final estimate is needed.

## 4. Choose the correct split strategy

A random split is not automatically correct.

| Data structure | Candidate strategy | Why |
|---|---|---|
| Independent, similarly distributed rows | Random split | Estimates performance on similar future samples |
| Imbalanced classification labels | Stratified split | Preserves approximate class proportions across subsets |
| Multiple rows per device or board | Group split | Prevents the same entity leaking across subsets |
| Time-ordered telemetry | Time-based split | Trains on the past and evaluates on later periods |
| Repeated experiments per chip | Group by chip or experiment | Measures generalization to unseen groups |

### Example: repeated devices

Suppose device A contributes 10,000 telemetry windows. A random row split may put windows from device A in both training and test sets. The model could learn device-specific signatures and appear to generalize well.

If the deployment question is “Will this work on a previously unseen device?”, split by device identity. If the question is “Will this predict next week on already-known devices?”, a chronological split may be more appropriate.

The split strategy must imitate the intended deployment boundary.

## 5. Data leakage: the silent failure

**Data leakage** occurs when information unavailable at prediction time influences training or model selection.

Common examples:

- Including a post-fault register snapshot when predicting whether the fault will happen.
- Fitting a scaler or imputer on the complete dataset before splitting.
- Selecting features using the test labels.
- Putting windows from the same device into train and test when evaluating unseen-device generalization.
- Randomly splitting time-series windows so future information helps predict the past.
- Using a status field that is assigned only after an engineer has confirmed the fault.

### Correct preprocessing order

~~~text
Split data first
    |
    +--> Training data: fit imputer, scaler, encoder and model
    |
    +--> Validation/test: transform with the already-fitted steps
                            |
                            +--> evaluate; never fit on these sets
~~~

A scikit-learn Pipeline makes this easier to enforce. During cross-validation, each fold fits its preprocessing only on that fold's training portion.

A pipeline does not automatically prevent every form of leakage. You must still define correct labels, split by the correct entity/time boundary and exclude features that reveal future information.

## 6. Build a baseline before a complex model

A baseline answers: “How well can we do without a sophisticated model?”

Useful baselines include:

- **Classification:** majority class, prior probability or a simple rule.
- **Regression:** predict the training-set mean or median.
- **Time series:** last observed value or seasonal naive forecast.
- **Ranking:** a simple popularity or frequency ranking.

A baseline is not a weak formality. It is a reference that exposes whether a model adds meaningful value.

For imbalanced classification, a majority-class classifier may have high accuracy while detecting none of the rare faults. Always inspect class-sensitive metrics.

## 7. Classification metrics and error costs

For binary classification:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | True positive (TP) | False negative (FN) |
| Actual negative | False positive (FP) | True negative (TN) |

Key metrics:

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Recall=\frac{TP}{TP+FN}
\]

\[
F1=2\frac{Precision \cdot Recall}{Precision+Recall}
\]

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** among predicted faults, how many are real?
- **Recall:** among real faults, how many were detected?
- **F1:** harmonic mean of precision and recall.
- **Balanced accuracy:** average recall across classes.
- **ROC-AUC:** ranking quality across classification thresholds.
- **PR-AUC / average precision:** precision-recall performance; often informative when positive events are rare.

No single metric is universally best. In a diagnostic system, a false negative may be much more costly than a false positive, but excessive false alarms can also overload engineers.

### Thresholds are decisions

A classifier may produce a probability estimate \(p\). A threshold converts it into a class:

\[
\hat y =
\begin{cases}
1 & p \geq \tau \\
0 & p < \tau
\end{cases}
\]

The default threshold of 0.5 is not automatically optimal. Select a threshold using validation data and an explicit cost or operating constraint. Never tune it on the final test set.

A score is not automatically a calibrated probability. If a downstream decision relies on probability values, assess calibration separately.

## 8. Practical lab: a synthetic SoC telemetry classifier

The lab creates a reproducible synthetic dataset with:
- Temperature
- DMA retry count
- Bus error count
- Link flap count
- IP family
- A binary teaching label for an impending fault

It deliberately includes a small number of missing numerical and categorical values. It compares a majority-class baseline with a logistic-regression pipeline.

### Install and run

~~~bash
cd daily-lessons/day-008
python -m pip install -r requirements.txt
python lab.py
~~~

The lab uses NumPy, pandas and scikit-learn.

### Pipeline architecture

~~~text
Training rows
    |
    +--> Numerical features
    |       median imputation
    |       standardization
    |
    +--> Categorical features
            most-frequent imputation
            one-hot encoding
    |
    v
ColumnTransformer
    |
    v
Logistic Regression
    |
    v
Validation metrics
    |
    v
Freeze model choice
    |
    v
Refit on train + validation
    |
    v
Evaluate untouched test set once
~~~

### Expected experiment results

With the fixed random seed and dataset generator in the lab, representative results are:

| Metric | Majority baseline — validation | Logistic pipeline — validation |
|---|---:|---:|
| Accuracy | 0.689 | 0.861 |
| Balanced accuracy | 0.500 | 0.826 |
| Precision | 0.000 | 0.804 |
| Recall | 0.000 | 0.732 |
| F1 | 0.000 | 0.766 |
| ROC-AUC | 0.500 | 0.903 |

After selecting the logistic pipeline using validation data, the lab refits it on training plus validation data and evaluates the test set once.

| Metric | Logistic pipeline — held-out test |
|---|---:|
| Accuracy | 0.833 |
| Balanced accuracy | 0.786 |
| Precision | 0.771 |
| Recall | 0.661 |
| F1 | 0.712 |
| ROC-AUC | 0.889 |
| Average precision | 0.805 |

Test confusion matrix, rows actual and columns predicted:

~~~text
                 Predicted 0   Predicted 1
Actual 0              113           11
Actual 1               19           37
~~~

These are deterministic results for this generated teaching dataset, not real silicon diagnostic performance. A production model needs representative data and validation against the intended deployment population.

### Lab checks

The script checks:
- Train, validation and test row IDs are disjoint.
- Class proportions are preserved approximately by stratification.
- The model beats the majority baseline on validation balanced accuracy.
- The imputer statistics equal training-plus-validation statistics after final refitting, not full-dataset statistics.
- The categorical encoder accepts an unseen category without crashing.
- The final held-out metrics meet broad sanity bounds.

## 9. Production ML workflow

A reproducible ML experiment should preserve more than a model file.

~~~mermaid
flowchart TD
  A[Problem contract] --> B[Dataset snapshot and provenance]
  B --> C[Label and feature validation]
  C --> D[Split manifest]
  D --> E[Baseline]
  D --> F[Candidate pipelines]
  E --> G[Validation comparison]
  F --> G
  G --> H[Freeze model and threshold]
  H --> I[Final test evaluation]
  I --> J[Model card and approval]
  J --> K[Versioned deployment]
  K --> L[Monitoring and drift review]
~~~

Record:
- Dataset version and source provenance
- Label definition and prediction horizon
- Feature schema and availability time
- Split strategy, IDs, seed and time boundaries
- Preprocessing and model configuration
- Code, library and model version identifiers
- Validation and test metrics
- Decision threshold and rationale
- Known limitations and approval history

For time-sensitive or safety-relevant applications, include drift monitoring and a rollback plan.

## 10. Reusable agent architecture: ML Experiment Governance Agent

This companion pattern applies agent engineering principles to ML experimentation. The agent can help an engineer inspect data and coordinate experiments, but deterministic services own data splits, permissions, training execution and evaluation.

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake and problem-contract validator]
  I --> D[Dataset steward: schema, provenance and labels]
  D --> S[Deterministic split service]
  S --> B[Baseline runner]
  S --> P[Candidate pipeline runner]
  B --> V[Validation evaluator]
  P --> V
  V --> R[Experiment reviewer]
  R -->|Accept candidate| F[Freeze configuration]
  F --> T[Test-set policy gate]
  T --> E[One-time final evaluator]
  E --> H[Human approval]
  H --> M[Model registry and release]
  V -. metrics and traces .-> O[Observability]
  E -. signed report .-> O
~~~

### Component responsibilities

- **Intake agent:** clarifies target, unit of analysis, horizon, constraints and success criteria.
- **Dataset steward:** checks schema, provenance, missingness, label quality and data access.
- **Split service:** deterministic, versioned split manifests; the LLM cannot choose test rows after seeing outcomes.
- **Experiment runner:** executes allow-listed pipelines in an isolated environment.
- **Evaluator:** computes deterministic metrics from predictions and labels.
- **Reviewer:** explains trade-offs and flags leakage risks.
- **Human approver:** authorizes release or new data access.

### Typed tool contracts

Examples of narrow tools:

- **profile_dataset** — input: dataset version; output: data profile.
- **validate_problem_contract** — input: prediction contract; output: contract report.
- **create_split** — input: dataset version, strategy, seed, group key and time cutoff; output: split manifest.
- **fit_candidate** — input: config ID and train split ID; output: model artifact.
- **evaluate_validation** — input: model ID and validation split ID; output: validation report.
- **freeze_candidate** — input: model ID and config hash; output: frozen candidate.
- **evaluate_final_test** — input: frozen candidate and test split ID; output: signed test report.
- **request_promotion** — input: candidate ID and approval ID; output: promotion request.

Every tool validates its schema and caller permissions. The model can propose parameters, but policy code decides whether the operation is authorized.

### State, reliability and security

Persist experiment state outside the model context:
- Task ID and state-machine status
- Dataset and split manifest IDs
- Artifact hashes and model configuration
- Tool-call idempotency keys
- Validation report and final test report
- Human approval and audit references

Retries must be bounded and idempotent. A failed training job can resume from a checkpoint, but it must not silently change the dataset or split manifest.

Use least-privilege dataset access, tenant isolation, secret redaction and sandboxed execution. Treat CSV cells, notebooks, logs and retrieved documents as untrusted data; they can contain prompt-injection text.

### Cost, latency and architecture trade-offs

- Run deterministic schema checks and metrics without an LLM.
- Cache immutable dataset profiles by dataset version and schema hash.
- Use one orchestrator until genuine independent work justifies specialists.
- Parallelize independent candidate experiments only with resource quotas.
- Avoid expensive models before a baseline and data-quality checks.
- Measure cost per accepted model, not only token cost.

A deterministic workflow is usually best for the core ML lifecycle. LLM agents add value for requirement clarification, documentation, hypothesis generation and explaining results—not for bypassing data controls.

## 11. Key terminology

| Term | Meaning |
|---|---|
| Sample | One observation or row |
| Feature | Input variable used for prediction |
| Label / target | Desired prediction output |
| Prediction horizon | Future interval described by the target |
| Training set | Data used to fit model and preprocessing parameters |
| Validation set | Data used for development choices |
| Test set | Held-out data for final evaluation |
| Stratification | Preserving approximate class proportions across splits |
| Group split | Keeping related entities in the same partition |
| Data leakage | Use of information unavailable at prediction time |
| Baseline | Simple reference model or rule |
| Pipeline | Chained preprocessing and estimator steps |
| Imputation | Replacing missing values using a defined method |
| Feature engineering | Transforming raw data into useful model inputs |
| Generalization | Performance on unseen data from the intended population |
| Data drift | Change in input data distribution over time |
| Concept drift | Change in the relationship between inputs and target |

## 12. Five review questions

1. Why is “predict a DMA fault” incomplete without a unit of analysis, prediction time and horizon?
2. When should you use a group split instead of a random row split?
3. Why must a scaler or imputer be fitted only on training data?
4. How can a majority-class baseline achieve high accuracy while having zero recall for faults?
5. Why should the test-set gate be controlled by deterministic policy rather than by an LLM planner?

## 13. Completion checklist

- [ ] Write a prediction contract for one engineering problem.
- [ ] Identify features that would leak future information.
- [ ] Explain when random, stratified, group and time splits are appropriate.
- [ ] Run the Python lab and inspect the confusion matrix.
- [ ] Compare the majority baseline with the logistic pipeline.
- [ ] Explain why validation and test data have different roles.
- [ ] Review the ML Experiment Governance Agent pattern.
- [ ] Answer the five review questions.

## 14. References

1. [Google Machine Learning Crash Course — Datasets, generalization and overfitting](https://developers.google.com/machine-learning/crash-course/overfitting): data quality, dataset splitting and generalization.
2. [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html): random and stratified splitting.
3. [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html): leakage prevention and preprocessing pipelines.
4. [scikit-learn — Dummy estimators](https://scikit-learn.org/stable/api/sklearn.dummy.html): simple baselines.
5. [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html): classification metrics and evaluation tools.
6. [Google Machine Learning Crash Course — Exercises](https://developers.google.com/machine-learning/crash-course/exercises): practical exercises on classification, metrics and data preparation.

## Next lesson

**Day 009 — Supervised Learning: Linear Regression, Logistic Regression and the Learning Objective.**

We will derive the first predictive models, understand their loss functions, interpret coefficients and compare regression with classification on practical datasets.
