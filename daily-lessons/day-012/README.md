# Session 012 — Unsupervised Learning: Clustering, K-Means, Hierarchical Clustering and DBSCAN

**Phase:** 3 — Machine Learning  
**Duration:** 60 minutes  
**Prerequisites:** Sessions 003, 005, 008 and 010–011: vectors, probability, data preparation, trees, distances and model evaluation.

## Learning objectives

By the end of this session, you will be able to:

1. Distinguish supervised learning, clustering and anomaly detection.
2. Explain K-Means assignments, centroid updates, inertia and initialization.
3. Use the elbow method, silhouette score and stability checks without treating any one metric as ground truth.
4. Explain agglomerative hierarchical clustering, linkage and dendrograms.
5. Explain DBSCAN core, border and noise points, and tune eps and min_samples.
6. Compare algorithms by geometry, assumptions, scalability and ability to handle new observations.
7. Build a reproducible clustering experiment for unlabeled SoC telemetry.
8. Design an agent-assisted discovery workflow with deterministic evaluation, typed tools, safe state, observability and human approval.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Supervised vs. unsupervised learning |
| 5–18 min | K-Means and choosing K |
| 18–28 min | Hierarchical clustering and dendrograms |
| 28–40 min | DBSCAN and density-based anomaly triage |
| 40–53 min | Python experiment |
| 53–57 min | Production architecture and operational controls |
| 57–60 min | Five review questions |

---

## 1. What changes when labels disappear?

In supervised learning, every training example contains a target label. A classifier learns a mapping from features to known classes.

In clustering, the input contains features but no target labels. The algorithm groups observations according to a chosen notion of similarity or density.

Example: a silicon telemetry stream contains temperature, DMA retry rate, bus error rate and queue pressure. We may not know which operating modes exist. Clustering can help reveal recurring operating regimes and unusual observations.

**Important distinction:** a cluster is not automatically a fault class. A density outlier is not automatically a hardware defect. The algorithm discovers structure under its chosen representation and distance metric; an engineer must interpret that structure.

~~~mermaid
flowchart LR
  A[Unlabeled telemetry] --> B[Schema and quality checks]
  B --> C[Feature selection and scaling]
  C --> D[Clustering algorithm]
  D --> E[Cluster profiles]
  D --> F[Noise / unusual observations]
  E --> G[Engineer interpretation]
  F --> G
  G --> H[Validated labels or follow-up tests]
~~~

## 2. K-Means

K-Means partitions observations into K clusters. Each cluster is represented by its centroid (the mean feature vector of assigned observations).

For observations x_i and centroids mu_j, the objective is to minimize within-cluster sum of squares (inertia):

\[
J=\sum_{i=1}^{N}\min_{j\in\{1,\ldots,K\}}\|x_i-\mu_j\|_2^2
\]

### 2.1 Lloyd's algorithm

1. Choose K initial centroids (scikit-learn defaults to K-Means++).
2. Assign each observation to its nearest centroid.
3. Recompute each centroid as the mean of its assigned observations.
4. Repeat assignment and update until convergence or the iteration limit.

The method converges to a local optimum, not necessarily the global optimum. Different initializations can lead to different solutions; use multiple starts (n_init) and a fixed seed for reproducible experiments.

### 2.2 Small worked example

One-dimensional observations:

\[
X=\{1,2,8,9\},\quad K=2
\]

Start with centroids 1 and 9.

- Assignments: {1,2} and {8,9}
- Updated centroids: mu_1=1.5, mu_2=8.5
- Inertia: (1-1.5)^2+(2-1.5)^2+(8-8.5)^2+(9-8.5)^2=1

The centroids summarize the two groups. They need not be actual observations.

### 2.3 What inertia tells you—and what it does not

Inertia is non-increasing as K increases. If every point gets its own cluster, inertia can approach zero. Therefore, “lowest inertia” is not a useful standalone rule for choosing K.

The **elbow method** plots inertia against K and looks for a point where additional clusters produce diminishing improvement. The elbow can be ambiguous.

### 2.4 Silhouette score

For each sample:

- a: mean distance to points in its own cluster.
- b: mean distance to points in the nearest other cluster.

\[
s=\frac{b-a}{\max(a,b)}
\]

The score is between -1 and 1:
- Near +1: the point is much closer to its own cluster.
- Near 0: clusters overlap around the point.
- Below 0: another cluster may be a better fit.

Silhouette is defined only when there are at least two clusters and fewer clusters than samples. It often favors compact, convex groups and should not be used as the sole criterion for density-based clusters.

### 2.5 Scaling and geometry

K-Means uses Euclidean distance. If temperature is measured in degrees and retry rate is a small fraction, raw scale can make temperature dominate.

Standardization:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}
\]

Fit the scaler on the designated reference/training window and persist it. Apply the same scaler to later observations. Do not refit on each live batch: that changes the coordinate system and makes centroids incomparable.

K-Means works best for roughly compact, similarly scaled, convex/isotropic clusters. It is not ideal for crescent-shaped clusters, very different densities or large numbers of outliers.

## 3. Hierarchical clustering

Hierarchical clustering builds a nested grouping structure rather than requiring every merge decision to be summarized by a single centroid.

**Agglomerative clustering** is bottom-up:

1. Begin with each observation as its own cluster.
2. Find the two closest clusters under a linkage rule.
3. Merge them.
4. Repeat until the requested number of clusters or distance threshold is reached.

A **dendrogram** visualizes the merge sequence. Cutting the dendrogram at a selected height yields a flat clustering.

### 3.1 Linkage methods

| Linkage | Distance between two clusters | Typical behavior |
|---|---|---|
| Single | Minimum pairwise distance | Can form long chains |
| Complete | Maximum pairwise distance | Favors compact groups |
| Average | Average pairwise distance | Compromise between single and complete |
| Ward | Increase in within-cluster variance | Compact, variance-oriented groups; Euclidean geometry |

Ward linkage chooses merges that minimize the increase in within-cluster sum of squares. It is not interchangeable with arbitrary metrics.

### 3.2 Trade-offs

- Useful when the hierarchy itself is meaningful or K is not known initially.
- Does not require random centroid initialization.
- Standard agglomerative clustering is transductive: the fitted hierarchy describes the supplied observations; the common scikit-learn estimator does not provide a general predict method for arbitrary new samples.
- Naive pairwise merging can be expensive in time and memory as N grows. Connectivity constraints can help when justified by the domain.

## 4. DBSCAN — density-based clustering

DBSCAN means **Density-Based Spatial Clustering of Applications with Noise**.

Instead of asking for K centroids, it finds regions with enough nearby observations.

Two main parameters:

- eps: neighborhood radius.
- min_samples: number of observations in the neighborhood needed for a core point. In scikit-learn, the point itself counts toward this number.

### 4.1 Three point types

- **Core point:** has at least min_samples observations within eps.
- **Border point:** is within the neighborhood of a core point but does not itself meet the core density requirement.
- **Noise point:** is not density-reachable from a core point; scikit-learn labels it -1.

~~~mermaid
flowchart TD
  A[Observation] --> B{At least min_samples\nwithin eps?}
  B -->|Yes| C[Core point]
  B -->|No| D{Within eps of a core point?}
  D -->|Yes| E[Border point]
  D -->|No| F[Noise: label -1]
  C --> G[Expand density-connected cluster]
  E --> G
~~~

### 4.2 Parameter behavior

- Smaller eps: stricter neighborhoods; more observations may become noise and a real cluster may fragment.
- Larger eps: more observations connect; separate clusters can merge.
- Larger min_samples: denser regions are required to form a cluster.

eps is expressed in the chosen feature space and distance metric. After standardization, it is in standardized-distance units—not degrees Celsius or raw retry counts.

### 4.3 Why DBSCAN is useful—and dangerous to over-interpret

DBSCAN can find non-convex shapes and mark low-density observations as noise without specifying K in advance. But:

- It struggles when real clusters have very different densities.
- Its results depend strongly on scaling, distance metric and eps.
- A noise point is not necessarily faulty; it may be a rare but valid operating condition.
- A standard DBSCAN fit is transductive and does not offer a general out-of-sample predict method.
- The scikit-learn implementation can use substantial memory in unfavorable parameter/data regimes; profile realistic workloads and consider OPTICS or other approaches when appropriate.

Use DBSCAN noise as a **triage queue**, not as an automatic fault verdict.

## 5. Compare the algorithms

| Property | K-Means | Agglomerative | DBSCAN |
|---|---|---|---|
| Must choose K? | Yes | Usually choose cluster count or distance threshold | No |
| Main idea | Centroid assignment | Repeated cluster merging | Density connectivity |
| Cluster shape | Compact/convex tendency | Depends on linkage | Arbitrary shapes |
| Outliers/noise | Every point is assigned | Every point is assigned | Explicit noise label -1 |
| New samples | predict with fitted K-Means | No general standard predict | No general standard predict |
| Main tuning | K, initialization | Linkage, K/threshold | eps, min_samples, metric |
| Scaling | Usually essential | Distance/linkage dependent | Usually essential for distance-based use |

No algorithm discovers the one true grouping independent of assumptions. Select the representation, metric and algorithm that fit the operational question.

## 6. Clustering evaluation without labels

When labels are absent, use several forms of evidence:

1. **Internal metrics:** silhouette, Davies–Bouldin, Calinski–Harabasz, inertia (for K-Means).
2. **Cluster size/profile:** are groups large enough and interpretable?
3. **Stability:** do similar clusters appear across seeds, time windows or reasonable feature perturbations?
4. **Operational utility:** do clusters help explain behavior, reduce triage time or guide a controlled follow-up test?
5. **External metrics:** if trusted labels become available later, compare with ARI or other external measures—but keep those labels out of fitting and model selection.

A high silhouette score does not prove that the groups correspond to meaningful hardware modes. For DBSCAN, do not blindly compare a silhouette computed after removing noise with a K-Means score computed on every sample; the populations differ.

## 7. Hands-on lab — synthetic SoC operating modes and anomaly triage

The lab generates 930 synthetic observations with four telemetry features:
- Temperature
- DMA retry rate
- Bus error rate
- Queue pressure

There are three compact generated operating modes and 30 injected unusual points. The clustering algorithms receive only feature vectors. Generator labels are retained solely for a post-hoc teaching metric (Adjusted Rand Index, ARI) and to check how well DBSCAN's noise flag catches the injected points.

### Reproduced experiment

**K-Means sweep**

| K | Inertia | Silhouette | Smallest cluster |
|---:|---:|---:|---:|
| 2 | 2061.629 | 0.4729 | 314 |
| 3 | 1014.561 | 0.5882 | 305 |
| 4 | 924.139 | 0.5946 | 8 |
| 5 | 830.611 | 0.4865 | 7 |
| 6 | 780.216 | 0.3552 | 7 |

The K=4 silhouette is slightly higher, but it creates an eight-point micro-cluster. The lab demonstrates a predeclared teaching policy requiring at least 3% of the batch in every cluster; under that constraint K=3 is selected. In production, minimum cluster size must be justified by the use case and reviewed alongside stability and operational utility.

**K-Means and Ward hierarchical clustering**

| Algorithm | Clusters | Silhouette | ARI on generated inliers |
|---|---:|---:|---:|
| K-Means, K=3 | 3 | 0.5882 | 1.0000 |
| Agglomerative, Ward, K=3 | 3 | 0.5857 | 1.0000 |

ARI is computed only after fitting, against the synthetic generator labels, for teaching. Real unlabeled telemetry does not provide this ground truth.

**DBSCAN**

Configuration: eps=0.60, min_samples=8.

| Measure | Result |
|---|---:|
| Clusters (excluding noise) | 3 |
| Noise observations | 40 |
| Noise fraction | 4.30% |
| Silhouette on non-noise observations only | 0.6330 |
| Injected-anomaly precision (synthetic truth) | 75.00% |
| Injected-anomaly recall (synthetic truth) | 100.00% |

DBSCAN flags all 30 injected points but also flags 10 legitimate generated points. This is a useful reminder that density outliers and actual faults are different concepts.

**DBSCAN eps sensitivity**

| eps | Clusters | Noise fraction | Synthetic precision | Synthetic recall |
|---:|---:|---:|---:|---:|
| 0.50 | 3 | 8.39% | 38.46% | 100% |
| 0.55 | 3 | 4.95% | 65.22% | 100% |
| 0.60 | 3 | 4.30% | 75.00% | 100% |
| 0.65 | 3 | 3.76% | 85.71% | 100% |

These are reproducible synthetic teaching results, not real silicon diagnostic performance or a universal parameter recommendation.

### Run locally

~~~bash
cd daily-lessons/day-012
python -m pip install -r requirements.txt
python lab.py
~~~

Expected final line:

~~~text
PASS: all Day 012 lab assertions passed.
~~~

The lab checks finite inputs, deterministic cluster structure, a minimum-size guard against the tiny K=4 cluster, agreement with synthetic inlier modes after fitting, DBSCAN noise counts and anomaly-triage precision/recall.

## 8. Production agent architecture — A18

### Unsupervised Discovery and Anomaly Triage Gate

The agent assists with experiment setup and interpretation. Deterministic services own data access, feature transformations, clustering, metrics and permissions.

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake agent: typed discovery contract]
  I --> V[Schema, tenant and data-quality validator]
  V --> S[Versioned reference window and scaler]
  S --> R[Deterministic experiment workflow]
  R --> K[K-Means worker]
  R --> H[Agglomerative worker]
  R --> D[DBSCAN worker]
  K --> E[Metric, size and stability evaluator]
  H --> E
  D --> E
  E --> P[Cluster profiles and noise candidates]
  P --> X[Verifier: evidence and policy checks]
  X --> A[Human engineering review]
  A -->|Approved interpretation| M[Versioned discovery report]
  A -->|Needs investigation| Q[Follow-up test / triage queue]
~~~

### Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Intake agent | Translate request into typed experiment contract | Data permissions or direct writes |
| Data validator | Schema, provenance, tenant and quality checks | Model interpretation |
| Feature service | Versioned feature selection/scaling | Re-fitting silently per live batch |
| Candidate runners | Bounded K-Means, hierarchical and DBSCAN jobs | Cross-tenant data access |
| Evaluator | Metrics, cluster sizes, stability, noise rate and resource use | Semantic fault labels |
| Verifier | Check evidence, constraints and report consistency | Inventing causal explanations |
| Human reviewer | Interpret modes and authorize follow-up | Unlogged manual changes |
| Registry/report store | Persist experiment configuration, artifacts and approvals | Unversioned mutable state |

### Reusable orchestration patterns and trade-offs

- **Deterministic workflow:** default for a fixed clustering experiment. It is easier to reproduce, test and audit.
- **Single agent + typed tools:** useful for natural-language requests such as “cluster last week's DMA telemetry.” The agent calls allow-listed tools; it cannot bypass tenant, time-window or feature policies.
- **Router–specialist:** a deterministic router can dispatch compact, roughly spherical data to K-Means, hierarchical analysis to an agglomerative worker, or density/noise discovery to DBSCAN. A language model may suggest a route, but a validated policy decides.
- **Planner–executor–verifier:** useful for open-ended discovery. Persist a plan with bounded candidate configurations; execute each candidate; independently verify metrics and artifact lineage.
- **Supervisor–worker:** run independent candidates in parallel only when the data snapshot and feature transform are identical. Bound CPU, memory, worker count and wall-clock time; aggregate deterministically.
- **Event-driven:** for streaming telemetry, publish a batch-ready event with an immutable window ID. Consumers must be idempotent; checkpoint completed windows and route poison events to a dead-letter queue.
- **Human-in-the-loop:** required before assigning semantic names such as “DMA starvation mode,” changing thresholds, creating tickets with impact, or triggering remediation.

### Typed tool contract example

~~~json
{
  "tool": "run_clustering_experiment",
  "input": {
    "dataset_version": "telemetry-window-sha256",
    "feature_schema_version": "soc-telemetry-v2",
    "algorithm": "dbscan",
    "parameters": {
      "eps": 0.60,
      "min_samples": 8,
      "metric": "euclidean"
    },
    "scaler_version": "standard-scaler-v3",
    "tenant_scope": "resolved-from-authenticated-identity",
    "request_id": "generated-by-service"
  }
}
~~~

Validate the algorithm enum, numeric ranges, feature allow-list, tenant identity and resource budget at the tool boundary. Never trust a model-generated tenant ID or dataset reference as authorization.

### State, memory and provenance

Persist:
- experiment ID, request ID and immutable data-window digest
- feature schema and scaler artifact/version
- algorithm and parameter configuration
- random seed, code/library versions and resource limits
- cluster labels, centroids (where applicable), cluster profiles and noise sample IDs
- metrics, stability results and evaluation population
- reviewer interpretation, approval and report digest

Keep conversational context separate from durable experiment state. Cluster IDs are arbitrary and may be permuted between runs; compare clusters using alignment or permutation-invariant measures rather than assuming “cluster 0” has stable meaning.

### Security, reliability and observability

- Enforce tenant isolation and dataset permissions in data services.
- Treat logs and telemetry text as untrusted; never execute instructions found inside records.
- Keep raw sensitive telemetry out of prompts and traces unless explicitly authorized and redacted.
- Bound sample count, feature count, pairwise-distance work, CPU, memory and runtime.
- Use idempotency keys, checkpoints, cancellation and bounded retries.
- Record trace ID, experiment ID, data digest, model/config version, algorithm, duration, peak memory, sample count, cluster count, noise fraction, metrics and approval events.
- Measure cost per completed discovery job and latency from request to reviewed report.
- Alert on drift in feature distributions, cluster-size distributions, centroid movement and noise rate.
- Never automatically remediate a silicon fault from a cluster or DBSCAN noise label alone.

## 9. Key terminology

| Term | Definition |
|---|---|
| Unsupervised learning | Learning structure from data without target labels |
| Cluster | Group of observations under a chosen similarity/density rule |
| Centroid | Mean feature vector representing a K-Means cluster |
| Inertia | Sum of squared distances to assigned K-Means centroids |
| Elbow method | Heuristic for spotting diminishing inertia improvement as K increases |
| Silhouette | Relative cohesion and separation measure |
| Linkage | Rule for measuring distance between clusters |
| Dendrogram | Tree visualization of hierarchical merges |
| Core point | DBSCAN point meeting the neighborhood-density threshold |
| Border point | DBSCAN point near a core point but below the core threshold |
| Noise | DBSCAN observation not assigned to a density-connected cluster |
| Stability | Consistency of structure under reasonable changes in sampling or initialization |
| ARI | Permutation-invariant agreement metric when external labels exist |

## 10. Five review questions

1. Why does K-Means inertia always tend to decrease as K increases, and why is that not enough to choose K?
2. How do Ward and single linkage differ in hierarchical clustering?
3. What do eps and min_samples control in DBSCAN?
4. Why is a DBSCAN noise point not automatically a silicon fault?
5. What must an agent-assisted clustering workflow persist to make a result reproducible and auditable?

## 11. References

1. [scikit-learn — Clustering user guide](https://scikit-learn.org/stable/modules/clustering.html): K-Means, hierarchical clustering, DBSCAN, assumptions and limitations.
2. [scikit-learn — K-Means API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html): initialization, n_init, inertia and prediction.
3. [scikit-learn — AgglomerativeClustering API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html): linkage and hierarchy parameters.
4. [scikit-learn — DBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html): core samples, eps, min_samples, noise label and memory notes.
5. [scikit-learn — silhouette_score API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html): definition, valid label counts and interpretation.
6. Ester et al. (1996), [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf).
7. Rousseeuw (1987), [Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis](https://doi.org/10.1016/0377-0427(87)90125-7).

---

## Next session — Session 013

**Dimensionality Reduction: PCA, Eigenvectors, SVD and Feature Compression**

We will learn how to reduce high-dimensional telemetry into compact representations, preserve variance, visualize clusters and understand the role of PCA before clustering and anomaly detection.
