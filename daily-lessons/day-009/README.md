# Day 009 — Supervised Learning: Linear Regression, Logistic Regression and the Learning Objective

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 3 — Machine learning  
**Previous lesson:** [Day 008 — ML Foundations](../day-008/README.md)

## Learning objectives

By the end of this session, you should be able to:

1. Distinguish regression from classification based on the target and decision.
2. Derive the linear-regression prediction equation and squared-error objective.
3. Explain logistic regression using logits, sigmoid and binary cross-entropy.
4. Interpret coefficients, intercepts, odds and decision thresholds.
5. Explain L1/L2 regularization and the meaning of scikit-learn's C parameter.
6. Choose metrics that match the task and operational cost.
7. Build and evaluate linear and logistic models with a reproducible pipeline.
8. Explain how an AI agent can assist model selection without controlling evaluation policy.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Recap problem formulation and data splits |
| 5–15 min | Regression versus classification |
| 15–25 min | Linear regression and least squares |
| 25–35 min | Logistic regression, sigmoid and cross-entropy |
| 35–42 min | Regularization, scaling and thresholds |
| 42–55 min | Python hands-on experiment |
| 55–60 min | Production model-selection architecture and review |

## 1. Start with the target: regression or classification?

The target determines the learning problem.

- **Regression:** predict a numeric quantity, such as transfer latency, temperature, throughput or remaining time.
- **Classification:** predict a category, such as fault/no fault, codec family or pass/fail.
- **Probability estimation:** estimate how likely a class is, rather than only returning a hard label.

A model name does not define the problem. Logistic regression is called “regression” because it models log-odds as a linear function, but it is commonly used for classification.

### SoC example

| Engineering question | Problem type | Possible output |
|---|---|---|
| Predict DMA transfer latency | Regression | 18.4 ms |
| Predict whether a link fault occurs | Binary classification | Fault probability 0.72 |
| Identify which of five IP blocks is failing | Multiclass classification | One class or a probability vector |
| Estimate remaining time before thermal throttling | Regression / time-to-event | 42 seconds |

Before fitting a model, write down the target, unit of analysis, prediction time, horizon and action. See [Day 008](../day-008/README.md) for prediction-contract and data-splitting discipline.

## 2. Linear regression: a numeric prediction

For one feature:

\[
\hat y = wx+b
\]

For multiple features:

\[
\hat y = w_1x_1+w_2x_2+\cdots+w_dx_d+b
\]

In vector notation:

\[
\hat y = Xw+b
\]

- \(X\): feature matrix.
- \(w\): learned coefficient vector.
- \(b\): intercept.
- \(\hat y\): predicted numeric target.

### Worked example: approximate transfer latency

Suppose an illustrative model is:

\[
\widehat{latency}=5
+0.12(T-65)
+0.65Q
+0.020P
+1.8R
\]

where T is temperature in °C, Q is queue depth, P is payload in KB and R is DMA retry count.

For T=70, Q=4, P=256 and R=1:

\[
5+0.12(5)+0.65(4)+0.020(256)+1.8=15.12\text{ ms}
\]

This equation is illustrative, not fitted hardware characterization. Real systems may require nonlinear terms, interactions, separate models by operating regime, or a different target definition.

### How does training choose the coefficients?

Ordinary least squares (OLS) chooses coefficients that minimize residual sum of squares:

\[
J(w,b)=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

This is the mean squared error (MSE) objective, up to an optional constant factor. A residual is the observed target minus the prediction.

MSE penalizes large errors more strongly because errors are squared. It is sensitive to outliers. Mean absolute error (MAE) is less sensitive to a few large residuals and is often easier to interpret in the target's units.

The fitted model is not automatically causal. A coefficient describes an association under the model and data; it does not prove that changing a hardware variable will cause the predicted change.

### Assumptions and diagnostics

A linear predictor is a useful baseline, but inspect:
- Residuals versus fitted values for nonlinearity or changing variance.
- Residual distribution when statistical inference is important.
- Correlated features and coefficient instability.
- Outliers and high-leverage samples.
- Whether train and deployment populations differ.

“Linear” means linear in the model parameters. You can add transformations and interactions, such as x², log(x), or x₁x₂, while still fitting a linear model in the expanded features.

## 3. Logistic regression: a probability for a class

A linear equation can produce any real number. A probability must lie between zero and one. Logistic regression first computes a linear score, called the **logit**:

\[
z=w^Tx+b
\]

Then it applies the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The predicted positive-class probability is:

\[
p(y=1\mid x)=\sigma(w^Tx+b)
\]

### Worked example

Suppose:

\[
z=-2+0.8(\text{retries})+0.9(\text{bus errors})
\]

For two retries and one bus error:

\[
z=-2+1.6+0.9=0.5
\]

\[
p=\sigma(0.5)\approx0.622
\]

The model estimates about 62.2% probability for the positive class. This is a model estimate, not a guarantee that the fault will occur.

### Why is it still called a linear model?

The log-odds are linear:

\[
\log\left(\frac{p}{1-p}\right)=w^Tx+b
\]

The sigmoid transforms that linear score into a probability. With the original features, the decision boundary at a 0.5 threshold is \(w^Tx+b=0\), a linear boundary.

## 4. Binary cross-entropy: what logistic regression minimizes

For one binary example with label \(y\in\{0,1\}\) and predicted probability \(p\):

\[
L(y,p)=-[y\log(p)+(1-y)\log(1-p)]
\]

Across N examples:

\[
J(w,b)=-\frac{1}{N}\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
\]

If the true label is 1, assigning probability 0.99 produces a small loss, while assigning 0.01 produces a large loss. The reverse applies when the true label is 0.

This connects directly to Session 006 on cross-entropy. The loss is used for fitting; accuracy, precision, recall, F1, ROC-AUC and calibration metrics answer different evaluation questions.

## 5. A probability is not yet a decision

A classifier often returns a probability. An application chooses a threshold τ:

\[
\hat y =
\begin{cases}
1 & p\geq\tau\\
0 & p<\tau
\end{cases}
\]

A 0.5 threshold is a convention, not a law. If missing a fault is expensive, a lower threshold may be appropriate, but it usually increases false alarms.

Select thresholds on validation data using an explicit operating objective, such as:
- Maximize F1.
- Achieve at least a required recall.
- Keep false alarms below a capacity limit.
- Minimize an explicit expected cost of false positives and false negatives.

Never tune a threshold on the final test set. If probabilities drive operational decisions, also check calibration: among cases assigned probability near 0.7, roughly 70% should be positive over a sufficiently large representative sample.

## 6. Regularization: control model complexity

When many features are correlated or the dataset is small, coefficients can become unstable. Regularization adds a penalty to the training objective.

### Ridge / L2

\[
J_{Ridge}=MSE+\lambda\sum_j w_j^2
\]

L2 discourages large coefficients and usually shrinks them toward zero without making most exactly zero.

### Lasso / L1

\[
J_{Lasso}=MSE+\lambda\sum_j |w_j|
\]

L1 can drive some coefficients exactly to zero, creating a sparse model. With correlated features, the selected feature can be unstable.

### Elastic Net

Combines L1 and L2 penalties. It can be useful when there are groups of correlated features.

In scikit-learn LogisticRegression, C is the inverse regularization strength: smaller C means stronger regularization. The exact supported penalty/solver combinations depend on the selected solver and library version.

### Why scale features?

Temperature, payload size and retry count use very different numeric ranges. Scaling helps regularized models compare coefficient magnitudes fairly and improves numerical optimization. Fit the scaler only on training data, then reuse its learned transformation on validation/test data. A Pipeline helps enforce this.

For unregularized OLS, scaling is not required for the mathematical least-squares solution, but it can improve numerical conditioning and make coefficient comparisons easier.

## 7. Practical lab: latency regression and fault classification

The lab creates a fixed-seed synthetic SoC telemetry dataset with temperature, queue depth, payload size, DMA retries, bus errors and link flaps.

It trains two separate models:
1. A regression model to estimate transfer latency in milliseconds.
2. A binary classifier to estimate a teaching fault label.

### Run it

~~~bash
cd daily-lessons/day-009
python -m pip install -r requirements.txt
python lab.py
~~~

### Regression pipeline

~~~text
Training partition
      |
      +--> StandardScaler
      |
      +--> LinearRegression / Ridge
      |
      v
Held-out regression test
      |
      +--> MAE, RMSE, R²
~~~

### Classification pipeline

~~~text
Stratified training partition
      |
      +--> StandardScaler
      |
      +--> LogisticRegression
      |
      +--> Validation probabilities
      |        |
      |        +--> choose threshold using validation F1
      |
      v
Freeze model + threshold
      |
      v
Held-out test evaluation once
      |
      +--> confusion matrix, precision, recall, F1
      +--> ROC-AUC, log loss, Brier score
~~~

### Reproducible results

With the fixed seed and synthetic generator in the lab:

**Regression — held-out test**

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 5.908 ms | 7.175 ms | -0.000 |
| Linear regression | 2.944 ms | 3.682 ms | 0.737 |
| Ridge (alpha=5) | 2.943 ms | 3.681 ms | 0.737 |

The model substantially improves over predicting the training-set mean. Ridge is almost identical here because the synthetic problem is close to linear and the regularization is mild.

**Classification — held-out test**

Positive-class prevalence is approximately 16.4%.

| Model / threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.837 | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic, threshold 0.50 | 0.839 | 0.538 | 0.545 | 0.102 | 0.171 |
| Logistic, validation-selected 0.20 | 0.761 | 0.694 | 0.361 | 0.593 | 0.449 |

Probability metrics for the logistic model on the held-out test set:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.745 |
| Average precision | 0.423 |
| Log loss | 0.388 |
| Brier score | 0.118 |

At the selected threshold, the confusion matrix is:

~~~text
                 Predicted 0   Predicted 1
Actual 0              239           62
Actual 1               24           35
~~~

The threshold was selected using validation F1, not test labels. The lower threshold improves recall at the cost of precision and accuracy. This is a decision trade-off, not a free improvement.

These are deterministic teaching results from synthetic data. They do not represent real silicon fault-prediction performance. A production model needs representative hardware data, validated labels, device-family and time-based holdouts, calibration analysis and engineering approval.

## 8. Production considerations

Before shipping a linear or logistic model:
- Verify that each feature exists at prediction time.
- Check for distribution shift across board revisions, firmware and workload.
- Evaluate performance by device family and important operating segments.
- Inspect false positives and false negatives with domain engineers.
- Calibrate probabilities if they drive risk decisions.
- Choose thresholds using validation data and explicit operational costs.
- Version the dataset, feature schema, preprocessing, model and threshold together.
- Monitor input drift, prediction distribution, error rates and delayed labels.
- Define rollback and retraining criteria.

A high ROC-AUC does not prove that a threshold is operationally acceptable. A low loss does not guarantee a useful decision. Metrics must match the actual engineering action.

## 9. Reusable agent architecture: Objective-Aware Model Selection Gate

This companion pattern extends the ML Experiment Governance Agent from Day 008. Its purpose is to route a well-defined prediction task to an appropriate objective and evaluation contract—not to let an LLM pick a metric after seeing test results.

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake agent: clarify target and decision]
  I --> C[Typed problem contract validator]
  C --> R{Deterministic task router}
  R -->|Numeric target| RG[Regression workflow]
  R -->|Categorical target| CL[Classification workflow]
  RG --> RB[Baseline and candidate runner]
  CL --> CB[Baseline and candidate runner]
  RB --> EV[Objective-specific evaluator]
  CB --> EV
  EV --> VG[Validation policy gate]
  VG -->|Accept candidate| F[Freeze model, features and threshold]
  F --> TG[Untouched test-set access gate]
  TG --> TE[Final evaluator]
  TE --> H[Human approval]
  H --> MR[Model registry / release]
  EV -. traces .-> O[Metrics, audit and observability]
  TE -. signed report .-> O
~~~

### Component boundaries

- **Intake agent:** helps translate an engineering question into a prediction contract.
- **Contract validator:** deterministic validation of target type, units, horizon, feature availability and metric requirements.
- **Task router:** routes numeric targets to regression and categorical targets to classification.
- **Candidate runner:** executes allow-listed pipelines in an isolated environment.
- **Evaluator:** computes task-specific metrics from versioned predictions and labels.
- **Validation gate:** compares candidates against predeclared acceptance criteria.
- **Test gate:** permits final evaluation only for a frozen candidate and immutable split manifest.
- **Human approver:** authorizes promotion and records rationale.

### Typed tool contracts

| Tool | Input | Output |
|---|---|---|
| validate_problem_contract | Target type, unit, horizon, features | Contract report |
| run_regression_candidate | Dataset/split IDs, model config | Model artifact |
| run_classifier_candidate | Dataset/split IDs, model config | Model artifact |
| evaluate_candidate | Artifact ID, validation split ID, metric contract | Validation report |
| select_threshold | Validation probabilities, labels, operating objective | Threshold proposal |
| freeze_candidate | Artifact, feature schema, threshold, config hash | Immutable candidate ID |
| evaluate_final_test | Frozen candidate ID, test split ID | Final test report |
| request_promotion | Candidate ID, approval record | Promotion request |

### Production safeguards

- The LLM may propose a task formulation, but deterministic code validates the target and schema.
- The LLM cannot change the test split, inspect test labels during development or promote a model.
- All tools enforce caller permissions and validate typed arguments.
- Training jobs have bounded CPU, memory, wall-clock and retry budgets.
- Long-running training uses an event-driven job ID, idempotency key and durable status.
- Parallel workers can compare independent candidates, but must share a fixed split manifest and merge results deterministically.
- A verifier checks metric calculations and artifact hashes; a human approves consequential releases.
- Traces record task ID, dataset/model versions, split ID, tool latency, resource use and outcome, with sensitive data redacted.

### Choosing the architecture

Use a deterministic workflow for known training and evaluation steps. Start with one agent plus a small tool set for experiment assistance. Add router/specialist agents only when model families or data domains require genuinely different expertise. Use planner–executor–verifier for exploratory multi-step analysis, supervisor–worker only for independent experiments, event-driven execution for long training jobs, and human approval for model promotion.

Measure cost per accepted model and time to a trustworthy result—not just tokens or number of agents.

## 10. Key terminology

| Term | Meaning |
|---|---|
| Regression | Predicting a numeric target |
| Classification | Predicting a discrete class |
| Linear predictor | Weighted sum of features plus intercept |
| Coefficient | Learned weight for a feature |
| Intercept | Model offset when feature values are zero in the chosen representation |
| Residual | Actual target minus predicted target |
| OLS | Ordinary least squares coefficient fitting |
| MSE / RMSE | Mean squared error / its square root |
| MAE | Mean absolute error |
| Logit | Log-odds; linear score before sigmoid |
| Sigmoid | Function mapping a real score into (0, 1) |
| Binary cross-entropy | Probabilistic loss for binary labels |
| Regularization | Penalty discouraging overly complex or large coefficients |
| Ridge / L2 | Squared-coefficient penalty |
| Lasso / L1 | Absolute-coefficient penalty |
| Threshold | Cutoff converting a probability into a class |
| Calibration | Agreement between predicted probabilities and observed frequencies |

## 11. Five review questions

1. Why is predicting latency a regression problem, while predicting fault/no-fault is a classification problem?
2. What does the sigmoid function do to the linear score in logistic regression?
3. Why can a majority classifier achieve high accuracy but zero recall for a rare fault class?
4. What happens to regularization strength in scikit-learn LogisticRegression when C is made smaller?
5. Why must a classification threshold be selected on validation data rather than the final test set?

## 12. Completion checklist

- [ ] Explain the difference between regression and classification.
- [ ] Derive the linear prediction equation and MSE objective.
- [ ] Calculate a sigmoid probability from a logit.
- [ ] Explain binary cross-entropy.
- [ ] Describe L1, L2 and feature scaling.
- [ ] Run the lab and compare the mean baseline with linear regression.
- [ ] Compare the majority classifier with logistic regression.
- [ ] Explain the precision/recall trade-off after threshold selection.
- [ ] Review the Objective-Aware Model Selection Gate.
- [ ] Answer the five review questions.

## 13. References

1. [scikit-learn — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html): OLS, linear prediction and regularized linear models.
2. [scikit-learn — LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html): ordinary least squares estimator.
3. [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): logistic classification, regularization and solver behavior.
4. [scikit-learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html): regression and classification metrics.
5. [scikit-learn — Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): preprocessing, pipelines and leakage prevention.
6. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course): practical supervised-learning lessons.

## Next lesson

**Day 010 — Decision Trees, Entropy, Information Gain and Random Forests.**

We will move from linear decision boundaries to nonlinear, rule-like partitions, understand how trees select splits and build an ensemble of trees.
