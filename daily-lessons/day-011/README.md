# Session 011 — K-Nearest Neighbors, Naive Bayes and Support Vector Machines

**Phase:** 3 — Machine Learning  
**Duration:** 60 minutes  
**Prerequisites:** Sessions 005, 008, 009 and 010: probability, data splits, logistic regression, decision trees and model evaluation.

## Learning objectives

By the end of this session, you will be able to:

1. Explain instance-based, probabilistic and margin-based classification.
2. Calculate Euclidean and Manhattan distances and explain why scaling matters for KNN.
3. Derive the Naive Bayes decision rule from Bayes' theorem and state its conditional-independence assumption.
4. Explain the SVM maximum-margin objective, support vectors, soft-margin parameter C and kernel trick.
5. Build leakage-safe scikit-learn pipelines for KNN, Gaussian Naive Bayes and SVM.
6. Compare models using validation data and evaluate a frozen candidate once on a held-out test set.
7. Select a model with deployment constraints such as memory, latency, interpretability and calibration in mind.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Recap: classification, splits and metrics |
| 5–18 min | KNN: distances, voting, scaling and k |
| 18–29 min | Naive Bayes: Bayes rule and conditional independence |
| 29–43 min | SVM: margin, C, kernels and support vectors |
| 43–55 min | Python experiment: common split, validation selection and frozen test |
| 55–60 min | Production model-selection architecture and review |

---

## 1. Three different ways to classify

These algorithms approach the same task from different assumptions:

| Algorithm | Learning idea | What is retained? |
|---|---|---|
| K-nearest neighbors (KNN) | Similar examples should have similar labels | Training examples or an index |
| Naive Bayes (NB) | Apply Bayes' theorem with a simplifying feature-independence assumption | Class priors and feature likelihood statistics |
| Support vector machine (SVM) | Find a boundary with a large margin between classes | A separating function and, for kernel SVMs, support vectors |

A model is not good merely because its mathematics is elegant. It must work on unseen data and fit the operational constraints.

## 2. K-nearest neighbors

KNN is an instance-based, non-parametric method. It does little explicit model fitting: it stores the training observations and looks up nearby examples when a query arrives.

### 2.1 Distance

For two feature vectors x and z, Euclidean distance is:

\[
d_2(x,z)=\sqrt{\sum_{j=1}^{d}(x_j-z_j)^2}
\]

Manhattan distance is:

\[
d_1(x,z)=\sum_{j=1}^{d}|x_j-z_j|
\]

Minkowski distance generalizes both:

\[
d_p(x,z)=\left(\sum_j|x_j-z_j|^p\right)^{1/p}
\]

- p=1: Manhattan
- p=2: Euclidean

### 2.2 Worked example

Suppose a telemetry query has standardized coordinates q=(1.0,1.0). Three stored examples are:

| Example | Coordinates | Label | Euclidean distance |
|---|---|---|---:|
| A | (0.8, 0.9) | Fault | 0.224 |
| B | (1.1, 1.2) | Fault | 0.224 |
| C | (0.2, -0.1) | Normal | 1.204 |

With k=3, two of the three neighbors are Fault, so uniform voting predicts Fault.

With distance weighting, closer neighbors contribute more. scikit-learn supports both uniform and inverse-distance voting.

### 2.3 Why feature scaling matters

Imagine temperature is measured in degrees from 35 to 105, while retry count is between 0 and 4. Raw Euclidean distance can be dominated by temperature simply because its numeric range is larger.

Standardization transforms each feature using training-set statistics:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}
\]

Fit the scaler on training data only, then apply that same fitted scaler to validation and test data. Put the scaler and estimator in one scikit-learn Pipeline to avoid leakage.

### 2.4 Choosing k

- Small k: flexible boundary, sensitive to noise; often lower bias and higher variance.
- Large k: smoother boundary; often higher bias and lower variance.
- Tune k and vote weighting using validation data or cross-validation—not the final test set.

**Limitations:** prediction can be expensive because many stored examples may need to be searched. Distance becomes less informative in very high-dimensional spaces (the curse of dimensionality). KNN is often a useful baseline for modest, well-scaled datasets, not a universal choice.

## 3. Naive Bayes

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem.

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

The naive assumption is that features are conditionally independent given the class:

\[
P(x_1,\ldots,x_d\mid C)\approx\prod_{j=1}^{d}P(x_j\mid C)
\]

Therefore:

\[
\hat C=\arg\max_C P(C)\prod_jP(x_j\mid C)
\]

In practice, implementations use log probabilities to avoid numerical underflow:

\[
\log P(C\mid X)=\text{constant}+\log P(C)+\sum_j\log P(x_j\mid C)
\]

### 3.1 Gaussian Naive Bayes

For continuous features, GaussianNB models each feature's class-conditional distribution as a Gaussian:

\[
P(x_j\mid C=c)=\frac{1}{\sqrt{2\pi\sigma_{jc}^{2}}}
\exp\left(-\frac{(x_j-\mu_{jc})^2}{2\sigma_{jc}^{2}}\right)
\]

It estimates a mean and variance for each feature and class.

Other variants are designed for different data:
- MultinomialNB: non-negative counts, commonly word counts.
- BernoulliNB: binary features.
- CategoricalNB: categorical feature values.

### 3.2 Strengths and caveats

**Strengths**
- Very fast training and inference.
- Small model footprint.
- Useful baseline, especially for sparse text/count data.
- Naturally produces class-probability estimates.

**Caveats**
- Correlated telemetry features violate the naive conditional-independence assumption.
- A useful classifier is not necessarily a well-calibrated probability estimator.
- GaussianNB's distribution assumptions may not fit discrete counts or multimodal sensor distributions.

If probabilities drive a threshold or downstream risk decision, evaluate calibration separately.

## 4. Support vector machines

An SVM seeks a decision boundary with a large margin between classes.

For a linear classifier:

\[
f(x)=w^Tx+b
\]

The predicted class is determined by the sign of f(x). The closest training examples that constrain the boundary are called **support vectors**.

### 4.1 Maximum margin and soft margin

For linearly separable data, the canonical hard-margin objective is:

\[
\min_{w,b}\frac{1}{2}\lVert w\rVert^2
\quad\text{subject to}\quad
y_i(w^Tx_i+b)\geq 1
\]

Real data contains overlap and noise. A soft-margin SVM introduces slack variables:

\[
\min_{w,b,\xi}\frac{1}{2}\lVert w\rVert^2+C\sum_i\xi_i
\]

subject to:

\[
y_i(w^Tx_i+b)\geq 1-\xi_i,\quad \xi_i\geq0
\]

The parameter **C** controls the trade-off:
- Smaller C: stronger regularization, wider tolerance for training violations.
- Larger C: stronger pressure to classify training examples correctly; may overfit noisy data.

### 4.2 Kernel trick

Some data cannot be separated by a straight line in the original feature space. A kernel computes similarity as if features had been mapped into a richer space, without explicitly constructing that mapping.

Common kernels:

| Kernel | Intuition |
|---|---|
| Linear | Straight decision boundary |
| Polynomial | Interactions up to a selected degree |
| RBF | Flexible local similarity boundary |
| Sigmoid | Neural-network-like similarity form; less commonly the first choice |

RBF kernel:

\[
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)
\]

- Larger gamma: each example has more local influence; the boundary can become very intricate.
- Smaller gamma: broader influence and smoother boundaries.

C and gamma must be selected together using validation or cross-validation.

### 4.3 SVM engineering caveats

- SVMs are sensitive to feature scale; use a fitted scaler in a Pipeline.
- Kernel SVC training can become impractical as sample count grows; scikit-learn notes that its fit time scales at least quadratically with the number of samples.
- For large datasets, consider LinearSVC or SGDClassifier.
- SVC's decision_function is a margin score, not a calibrated probability.
- If probabilities are required, use a separate calibration procedure with properly isolated validation/cross-validation data.

## 5. Compare the algorithms

| Dimension | KNN | GaussianNB | SVM |
|---|---|---|---|
| Training | Mostly stores examples | Estimates simple statistics | Solves an optimization problem |
| Prediction | Distance search and vote | Evaluate class likelihoods | Evaluate decision function |
| Scaling | Usually important | Not usually required for GaussianNB | Important |
| Memory | Can retain full dataset | Usually compact | Support vectors/model parameters |
| Nonlinear boundary | Local neighborhoods | Limited by likelihood assumptions | RBF and other kernels |
| Probability output | Neighbor vote proportions | Native posterior estimates | Requires calibration for probabilities |
| High dimensions | Distance concentration can hurt | Often a useful simple baseline | Linear SVM can work well |

There is no universally best algorithm. Compare models on the same split, with the same feature contract and an evaluation metric that matches the engineering objective.

## 6. Hands-on lab — synthetic SoC telemetry classification

The lab creates 3,000 reproducible synthetic samples with:
- Temperature
- DMA retry count
- Bus error count
- Queue depth
- Link flap count
- A deliberately unhelpful noise sensor

The target is generated from a noisy nonlinear rule with a temperature/link-flap interaction. This is a teaching dataset, not measured silicon behavior.

### Experiment protocol

1. Create stratified train, validation and test partitions (60/20/20).
2. Fit KNN candidates inside a StandardScaler Pipeline.
3. Fit GaussianNB candidates.
4. Fit scaled linear and RBF SVC candidates.
5. Select hyperparameters and model family using validation balanced accuracy.
6. Freeze the selected candidate.
7. Evaluate that candidate and a predeclared majority baseline once on the held-out test set.

### Reproduced results

Positive class prevalence: 23.17%.

**Best validation setting in each family**

| Candidate | Validation balanced accuracy | Selected setting |
|---|---:|---|
| KNN | 0.7474 | k=21, uniform weights |
| GaussianNB | 0.7484 | var_smoothing=1e-9 |
| SVC | 0.8092 | RBF, C=0.5, gamma=scale |
| Majority baseline | 0.5000 | Most frequent class |

SVC was selected on validation data. Only after freezing it was the held-out test set evaluated.

**Frozen SVC — held-out test**

| Metric | Result |
|---|---:|
| Accuracy | 0.8517 |
| Balanced accuracy | 0.8557 |
| Precision | 0.6316 |
| Recall | 0.8633 |
| F1 | 0.7295 |
| ROC-AUC | 0.9281 |

**Majority baseline — held-out test**

| Metric | Result |
|---|---:|
| Accuracy | 0.7683 |
| Balanced accuracy | 0.5000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5000 |

The baseline's accuracy looks respectable because most samples are normal, yet it detects no faults. Balanced accuracy and recall expose that failure.

The values above are reproduced with the fixed random seed and lab implementation. They are not a claim about real hardware performance.

### Run locally

~~~bash
cd daily-lessons/day-011
python -m pip install -r requirements.txt
python lab.py
~~~

Expected final line:

~~~text
PASS: all Day 011 lab assertions passed.
~~~

## 7. Production model-selection architecture — A17

### Classical Classifier Selection Gate

This companion design extends the experiment-governance patterns with three distinct candidate families: distance-based, probabilistic and margin-based classifiers.

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake and typed problem contract]
  I --> D[Dataset and schema validation]
  D --> S[Immutable split manifest]
  S --> R[Deterministic candidate runner]
  R --> K[KNN pipeline]
  R --> N[Naive Bayes pipeline]
  R --> V[Linear / RBF SVM pipeline]
  K --> E[Validation evaluator]
  N --> E
  V --> E
  E --> G[Metric and latency gate]
  G --> F[Frozen candidate]
  F --> T[One-time held-out test gate]
  T --> H[Human review and model registry]
  H --> O[Approved deployment artifact]
  E -.-> M[Traces, metrics, cost and audit]
  R -.-> M
~~~

### Component boundaries

- **Intake/router:** classifies the task and produces a typed prediction contract. It may recommend candidate families but cannot change data permissions.
- **Dataset service:** validates schema, lineage, label definition and split membership.
- **Preprocessing pipeline:** owns fitted scaling/imputation; fit only on training folds.
- **Candidate runners:** execute bounded, versioned KNN, NB and SVM configurations.
- **Evaluator:** computes predeclared metrics, calibration checks and resource measurements.
- **Policy gate:** applies deterministic acceptance thresholds.
- **Test gate:** evaluates only a frozen candidate against the untouched test partition.
- **Human reviewer:** approves promotion and records the decision.
- **Artifact registry:** stores model, preprocessing, data/split identifiers, configuration and evaluation report.

### Typed candidate contract

A candidate configuration should include:
- candidate_id
- algorithm (knn, gaussian_nb, linear_svm, rbf_svm)
- feature_schema_version
- preprocessing_version
- hyperparameters
- training_data_version
- split_manifest_id
- random_seed
- code_version

Reject unknown algorithms, invalid parameter ranges and schema mismatches before execution.

### Reliability, security and operations

- Keep test labels inaccessible to candidate workers and LLM tools.
- Treat uploaded logs, dataset text and retrieved documentation as untrusted content, not instructions.
- Enforce permissions in the tool service, not in prompts.
- Use a single deterministic workflow for a small candidate set. Add supervisor/worker parallelism only when candidate runs are independent and compute budgets are reserved atomically.
- For event-driven training, use job IDs, idempotency keys, checkpoints and a dead-letter path.
- Persist run state and artifact references; do not rely on chat history as the experiment database.
- Bound CPU, RAM, wall-clock time, candidate count and retries.
- Record model/config/data versions, tool calls, validation metrics, latency, peak memory and cost.
- Require approval before deployment or other consequential writes.
- Evaluate the entire workflow against a fixed benchmark; an LLM-generated explanation is not evidence that a model passed.

### Architecture trade-offs

- **Deterministic workflow:** simplest and easiest to audit for a fixed experiment protocol.
- **Single agent + tools:** useful when natural-language requests need flexible experiment setup, while execution stays constrained.
- **Router + specialists:** helpful when separate teams own data, training and review; introduces routing and handoff failure modes.
- **Planner–executor–verifier:** useful for open-ended research tasks, but the verifier must be independent and criteria must be explicit.
- **Supervisor–worker:** speeds up independent candidate runs but adds scheduling, resource contention and result aggregation.
- **Event-driven:** appropriate for long-running jobs and queues; requires idempotency and durable state.
- **Human-in-the-loop:** required for model promotion in high-impact workflows.

The LLM may explain the experiment and suggest follow-up candidates. It must not grant itself data access, change the test manifest or approve its own deployment.

## 8. Key terminology

- **Instance-based learning:** prediction uses stored examples rather than only a compact fitted parameter vector.
- **Non-parametric:** model complexity is not fixed by a small, predetermined parameter count.
- **Distance metric:** rule for measuring dissimilarity between feature vectors.
- **Conditional independence:** features are assumed independent once the class is known.
- **Prior:** probability assigned to a class before observing the features.
- **Likelihood:** probability of observed features under a class model.
- **Margin:** distance-like separation between a decision boundary and its closest constrained samples.
- **Support vector:** training sample that helps define the SVM boundary.
- **Kernel:** function that computes similarity in an implicit feature space.
- **Calibration:** agreement between predicted probabilities and observed event frequencies.
- **Balanced accuracy:** average recall across classes.
- **Data leakage:** use of information during training/model selection that would not legitimately be available at prediction time.

## 9. Five review questions

1. Why can unscaled temperature dominate retry count in Euclidean KNN distance?
2. What conditional-independence assumption makes Naive Bayes computationally simple, and why can it be unrealistic for correlated telemetry?
3. What is a support vector, and how does C affect the soft-margin SVM?
4. Why does an SVC decision score not automatically represent a calibrated probability?
5. Why must hyperparameters be selected on validation data before the final test set is used?

## 10. Further reading

1. [scikit-learn — Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
2. [scikit-learn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
3. [scikit-learn — Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
4. [scikit-learn — KNeighborsClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
5. [scikit-learn — SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
6. [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
7. Cortes and Vapnik, [Support-Vector Networks (1995)](https://doi.org/10.1007/BF00994018)

## Next session — Session 012

**Unsupervised Learning: Clustering, K-Means, Hierarchical Clustering and DBSCAN**

We will learn how to discover groups and anomalous observations when labels are unavailable, including how to evaluate clusters without pretending that a cluster ID is a ground-truth class.
