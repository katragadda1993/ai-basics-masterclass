# Day 005 — Probability, Statistics and Bayes' Theorem

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 2 — Mathematics for AI  
**Previous lesson:** [Day 004 — Functions, Derivatives, Gradients and the Chain Rule](../day-004/README.md)

## Learning objectives

By the end of this session, you should be able to:

1. Define an event, sample space and random variable.
2. Distinguish probability mass, probability density and cumulative probability.
3. Calculate expectation, variance and standard deviation.
4. Explain conditional probability, independence and Bayes' theorem.
5. Apply Bayes' theorem to a rare-event diagnostic example and explain the base-rate effect.
6. Explain a Gaussian distribution and estimate its parameters using maximum likelihood.
7. Distinguish a model score from a calibrated probability.
8. Explain how uncertainty estimates can guide an agent's routing and escalation without replacing authorization.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Recall gradients and connect statistics to model learning |
| 5–15 min | Events, random variables and probability distributions |
| 15–25 min | Expectation, variance and standard deviation |
| 25–37 min | Conditional probability and Bayes' theorem |
| 37–45 min | Gaussian distribution and maximum likelihood |
| 45–55 min | Python lab: Bayes, simulation and Gaussian MLE |
| 55–58 min | Probability-aware agent routing and safety |
| 58–60 min | Five review questions |

## 1. Why probability and statistics matter in AI

Day 004 showed how derivatives tell us how a loss changes as model parameters change. But real data is variable, incomplete and noisy. A model must learn from samples and often needs to express uncertainty about its predictions.

Probability provides a mathematical language for uncertainty. Statistics helps us summarize samples, estimate unknown parameters and evaluate whether patterns are likely to generalize.

Examples:
- A classifier estimates the probability that an audio frame contains a click or pop.
- A diagnostic model estimates the probability of a fault given telemetry.
- A language model assigns probabilities to possible next tokens.
- A monitoring system estimates the expected rate of errors in a time window.

A probability is meaningful only relative to a defined event, population, data-generation process and set of assumptions.

## 2. Events, sample spaces and random variables

A **sample space** is the set of possible outcomes of an experiment. An **event** is a subset of those outcomes.

For a fair six-sided die:

\[
\Omega=\{1,2,3,4,5,6\}
\]

The event “roll an even number” is \(A=\{2,4,6\}\), so:

\[
P(A)=\frac{3}{6}=0.5
\]

A **random variable** maps outcomes to numerical values. For example, \(X\) can represent the number of bus errors observed in a fixed one-second window.

For a discrete random variable, the probability mass function (PMF) assigns probability to each value:

\[
p(x)=P(X=x), \qquad \sum_x p(x)=1
\]

For a continuous random variable, a probability density function (PDF) describes density. Probability is an area over an interval, not the density at a single point:

\[
P(a\le X\le b)=\int_a^b f(x)\,dx
\]

The cumulative distribution function (CDF) is:

\[
F(x)=P(X\le x)
\]

A continuous variable can have \(P(X=x)=0\) at every individual point while still having nonzero probability over intervals.

## 3. Expectation, variance and standard deviation

### 3.1 Expected value

The expected value is the probability-weighted long-run average:

\[
E[X]=\sum_x xP(X=x)
\]

Example: suppose the number of retries in a transaction has this distribution:

| Retries (x) | Probability |
|---:|---:|
| 0 | 0.2 |
| 1 | 0.5 |
| 2 | 0.3 |

Then:

\[
E[X]=0(0.2)+1(0.5)+2(0.3)=1.1
\]

The expected number of retries is 1.1 per transaction over many comparable transactions. It does not mean that an individual transaction can have 1.1 retries.

### 3.2 Variance and standard deviation

Variance measures the expected squared distance from the mean:

\[
\operatorname{Var}(X)=E[(X-\mu)^2]
=E[X^2]-E[X]^2
\]

For the retry example:

\[
E[X^2]=0^2(0.2)+1^2(0.5)+2^2(0.3)=1.7
\]

\[
\operatorname{Var}(X)=1.7-1.1^2=0.49
\]

Standard deviation is the square root of variance:

\[
\sigma=\sqrt{0.49}=0.7
\]

Variance is expressed in squared units; standard deviation is in the original units.

### 3.3 Population versus sample variance

For a complete population of \(N\) values:

\[
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
\]

For a sample of \(n\) observations, the commonly used unbiased sample-variance estimator is:

\[
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

Do not mix these formulas. Maximum-likelihood estimation of a Gaussian's variance uses denominator \(n\), whereas the unbiased sample-variance estimator uses \(n-1\).

## 4. Conditional probability and Bayes' theorem

### 4.1 Conditional probability

Conditional probability asks: what is the probability of event A when we already know event B occurred?

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
\]

provided \(P(B)>0\).

The notation \(P(A\mid B)\) is not the same as \(P(B\mid A)\). Reversing the condition generally changes the answer.

### 4.2 Bayes' theorem

Bayes' theorem reverses a conditional probability when the prior and likelihood are known:

\[
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
\]

For two possible states, “fault” and “healthy”:

\[
P(\text{fault}\mid\text{alarm})=
\frac{P(\text{alarm}\mid\text{fault})P(\text{fault})}
{P(\text{alarm})}
\]

The denominator accounts for alarms from both faulty and healthy systems:

\[
P(\text{alarm})=
P(\text{alarm}\mid\text{fault})P(\text{fault})
+
P(\text{alarm}\mid\text{healthy})P(\text{healthy})
\]

### 4.3 Worked example: rare silicon fault

Assume an illustrative diagnostic detector has:
- Fault prevalence (prior): 1%
- Sensitivity: 95% (it alarms on 95% of actual faults)
- Specificity: 90% (it stays quiet on 90% of healthy cases)
- False-positive rate: 10%

We want the probability of a real fault after an alarm.

\[
P(F)=0.01,\quad P(A\mid F)=0.95
\]

\[
P(A\mid\neg F)=1-0.90=0.10
\]

\[
P(F\mid A)=
\frac{0.95\times0.01}
{0.95\times0.01+0.10\times0.99}
\]

\[
P(F\mid A)=\frac{0.0095}{0.1085}\approx0.0876
\]

<box>
**Result: approximately 8.76%**

Even with 95% sensitivity, fewer than one in ten alarms indicate a real fault under these assumed base rates and detector characteristics.
</box>

Why? Faults are rare, so the healthy population can produce many false alarms. This is the **base-rate effect**. Sensitivity alone does not tell us the probability that an alarm is correct.

The numbers are a teaching example, not measured silicon-validation performance. In a real project, estimate prevalence, sensitivity and specificity from representative, independently labelled data. Account for process, workload, temperature, firmware and population shifts.

### 4.4 Independence is an assumption, not a convenience

Two observations are conditionally independent given a state only if:

\[
P(A_1,A_2\mid F)=P(A_1\mid F)P(A_2\mid F)
\]

Repeated samples from the same failing device, same clock domain or same firmware run may be correlated. Multiplying their likelihoods as if independent can create unjustified confidence.

## 5. Gaussian distribution and maximum likelihood

### 5.1 Normal distribution

A Gaussian (normal) distribution is described by mean \(\mu\) and standard deviation \(\sigma\):

\[
f(x)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

The mean locates the center; standard deviation controls spread.

The standard score is:

\[
z=\frac{x-\mu}{\sigma}
\]

A z-score says how many standard deviations an observation is from the mean. The familiar “68–95–99.7” rule is an approximation for normally distributed data: about 68% lies within one standard deviation, 95% within two, and 99.7% within three.

Do not assume every telemetry metric is Gaussian. Error counts may be discrete; latency may be skewed or heavy-tailed; data can be multimodal or autocorrelated. Inspect the data and validate distributional assumptions.

### 5.2 Maximum likelihood estimation (MLE)

MLE chooses parameters that make the observed data most likely under a model.

For independent observations \(x_1,\dots,x_n\) from a Gaussian:

\[
\hat{\mu}_{MLE}=\frac{1}{n}\sum_i x_i
\]

\[
\hat{\sigma}^2_{MLE}=\frac{1}{n}\sum_i(x_i-\hat{\mu})^2
\]

Example observations: 9, 10, 11, 12, 13.

\[
\hat{\mu}=11,\qquad
\hat{\sigma}^2=2,\qquad
\hat{\sigma}=\sqrt{2}\approx1.414
\]

This is a fitted model of the observed sample, not proof that the real process is Gaussian. MLE can be sensitive to outliers and model misspecification.

### 5.3 Likelihood versus probability

- Probability: assume parameters are known; ask how likely an observation or event is.
- Likelihood: observations are known; compare which parameter values make those observations more plausible.

For independent observations, likelihood multiplies individual densities or masses. In computation, log-likelihood is often used because adding logarithms is more numerically stable than multiplying many tiny values.

## 6. Probability estimates are not automatically calibrated

A classifier might output a score of 0.9. That number is not necessarily a trustworthy 90% probability.

A model is **calibrated** when, among cases assigned a probability near 0.8, approximately 80% are positive over a sufficiently large representative set.

Useful evaluation methods include:
- Reliability diagrams / calibration curves
- Brier score
- Log loss
- Precision and recall at operational thresholds
- Calibration on a held-out dataset representative of deployment

Discrimination and calibration are different. A model can rank faulty cases above healthy cases yet still produce poor probability estimates.

## 7. Practical lab: Bayes, simulation and Gaussian MLE

The standard-library Python lab in [lab.py](lab.py) implements:
1. Discrete expectation and population variance.
2. Bayes' theorem for a rare fault detector.
3. A seeded Monte Carlo experiment to compare simulated and analytical posterior probability.
4. Gaussian maximum-likelihood estimates.
5. A log-likelihood calculation.
6. Input validation and deterministic assertions.

Run:

\`\`\`bash
cd daily-lessons/day-005
python lab.py
\`\`\`

No third-party packages are required.

Expected key values:
- Expected retries: 1.1
- Retry variance: 0.49
- Fault probability after alarm: about 0.08756
- Gaussian MLE mean for [9, 10, 11, 12, 13]: 11
- Gaussian MLE variance: 2

Monte Carlo results fluctuate. The script fixes the random seed for reproducibility and checks a tolerance rather than expecting an exact match.

## 8. Agent architecture connection: probability-aware routing

A specialist router may use a classifier to estimate whether a request concerns a DMA, PCIe, audio or memory issue. A probability estimate can help prioritize a specialist, but it must not grant access to tools or authorize actions.

\`\`\`mermaid
flowchart TD
  E[New diagnostic event] --> V[Validate schema and provenance]
  V --> F[Extract features]
  F --> M[Versioned model + calibrated probabilities]
  M --> R[Routing policy with thresholds and abstention]
  R -->|Clear low-risk route| S[Read-only specialist]
  R -->|Ambiguous / out-of-distribution| H[Human triage]
  R -->|High impact action| H
  S --> Q[Independent evidence checks]
  Q --> A[Audit record and response]
  H --> A
  P[Authorization policy] --> S
  P --> H
  A --> O[Observability and evaluation]
\`\`\`

### Component boundaries

- **Classifier:** estimates probabilities; it does not execute actions.
- **Router policy:** maps estimates to a route, including an abstain/fallback path.
- **Authorization service:** independently checks identity, scope and permissions.
- **Specialist:** receives a typed task and least-privilege tools.
- **Verifier:** checks evidence and acceptance criteria.
- **Audit store:** records model version, feature/schema version, event ID, decision, evidence references and approvals.

### Example routing policy

Use thresholds selected on validation data, not arbitrary numbers copied from a tutorial:
- If the top class probability exceeds a validated high threshold and the input is in-distribution, route to a read-only specialist.
- If scores are close, calibration is poor, or the input is out-of-distribution, abstain and request human triage.
- If an operation is consequential, require explicit authorization and approval regardless of model confidence.

### Typed task contract

\`\`\`json
{
  "event_id": "unique-event-id",
  "case_id": "diagnostic-case-id",
  "feature_schema_version": "v1",
  "model_version": "router-2026-01",
  "candidate": "dma",
  "estimated_probability": 0.82,
  "evidence_refs": ["trace-123", "log-456"],
  "requested_mode": "read_only"
}
\`\`\`

Validate the schema and event identity. Do not accept a model-generated candidate as proof of authorization. Avoid placing raw confidential telemetry into prompts or logs.

### Production trade-offs

- **Latency and cost:** a cheap deterministic filter can handle obvious cases before invoking a larger model.
- **Reliability:** use bounded retries, timeouts, idempotency keys and durable state for event-driven execution.
- **Memory:** store case state and evidence references outside the prompt; retrieve only what is needed.
- **Evaluation:** measure routing accuracy, abstention rate, false-negative rate, calibration, latency and cost.
- **Security:** treat telemetry and retrieved documents as untrusted data; enforce permissions in code.
- **Human-in-the-loop:** send uncertain or high-impact cases to an engineer instead of forcing a prediction.

A probability is evidence for a decision, not a substitute for engineering judgment or an access-control decision.

## 9. Key terminology

| Term | Meaning |
|---|---|
| Sample space | Set of possible outcomes |
| Event | A subset of possible outcomes |
| Random variable | Numerical mapping from outcomes |
| PMF | Probability mass function for discrete variables |
| PDF | Probability density function for continuous variables |
| CDF | Probability that a variable is at most a value |
| Expected value | Probability-weighted average |
| Variance | Expected squared deviation from the mean |
| Standard deviation | Square root of variance |
| Conditional probability | Probability given another event |
| Prior | Probability before observing new evidence |
| Likelihood | Probability of evidence under a hypothesis |
| Posterior | Updated probability after evidence |
| Sensitivity | True-positive rate |
| Specificity | True-negative rate |
| Base rate | Prevalence of an event in the population |
| MLE | Parameter estimate maximizing likelihood |
| Calibration | Agreement between predicted probabilities and observed frequencies |
| Abstention | Declining to make a confident automated choice |

## 10. Five review questions

1. A fault detector has 95% sensitivity and 90% specificity. If fault prevalence is only 1%, why is the probability of a real fault after an alarm not 95%?
2. For a discrete random variable with values 0, 1 and 2 and probabilities 0.2, 0.5 and 0.3, calculate its expected value and variance.
3. What is the difference between a PDF value and the probability that a continuous random variable lies in an interval?
4. Why does Gaussian MLE variance divide by \(n\), while the usual unbiased sample variance divides by \(n-1\)?
5. Why must an AI agent's authorization policy remain independent of its classifier's confidence score?

## Takeaways

- Probability describes uncertainty; statistics learns from observed samples.
- Expected value summarizes a long-run average; variance and standard deviation describe spread.
- Bayes' theorem combines prior prevalence and evidence likelihood.
- Rare-event base rates can make many alarms false even when sensitivity is high.
- A model score is not automatically a calibrated probability.
- Probabilistic routing can help select specialists, but permissions and approval gates must remain deterministic and independently enforced.

## Further reading

- [OpenStax — Mean or Expected Value and Standard Deviation](https://openstax.org/books/introductory-statistics-2e/pages/4-2-mean-or-expected-value-and-standard-deviation)
- [SciPy — Probability distributions](https://docs.scipy.org/doc/scipy/tutorial/stats/probability_distributions.html)
- [SciPy — Continuous distributions and fitting data](https://docs.scipy.org/doc/scipy/tutorial/stats/continuous.html)
- [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/)

## Completion checklist

- [ ] Read the lesson and reproduce the Bayes calculation by hand.
- [ ] Run lab.py and inspect all assertions.
- [ ] Explain the base-rate effect in your own words.
- [ ] Distinguish MLE variance from unbiased sample variance.
- [ ] Explain why an agent must abstain when uncertain and why confidence never grants permissions.
- [ ] Answer all five review questions.
