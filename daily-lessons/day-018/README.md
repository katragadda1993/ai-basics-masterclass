# Session 018 — Convolutional Neural Networks: Convolution, Filters, Stride, Padding, Pooling and Feature Maps

**Phase:** 4 — Deep Learning  
**Duration:** 60 minutes  
**Prerequisite:** Session 017 — PyTorch tensors, autograd, mini-batch training, CPU/GPU placement and checkpointing.

## Why this lesson follows Session 017

Session 016 built an MLP from first principles; Session 017 moved that network into PyTorch and covered the training runtime. This lesson changes the model architecture. A fully connected layer treats every input position as a separate feature. A convolutional layer instead uses local connectivity and shared filters to detect the same pattern at many spatial positions.

The practical lab uses small synthetic images so every input shape, feature map and parameter count can be inspected. The engineering connection is broader: 1D CNNs can process sampled waveforms, while 2D CNNs can process images or time-frequency representations such as audio spectrograms.

## Learning objectives

By the end of this session, you will be able to:

1. Explain why convolution is useful for spatial and time-frequency data.
2. Distinguish channels, filters, kernels, feature maps and receptive fields.
3. Calculate convolution output dimensions from kernel, stride, padding and dilation.
4. Count trainable parameters in Conv2d and dense layers.
5. Explain weight sharing, local connectivity and translation equivariance.
6. Understand max pooling, average pooling and strided convolution trade-offs.
7. Trace NCHW tensor shapes through a small CNN.
8. Implement and train a CNN with PyTorch.
9. Connect CNN design to 1D waveforms and 2D audio spectrograms.
10. Design a governed CNN training and release workflow.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Why an MLP is not always the right inductive bias |
| 5–15 min | Convolution, filters, channels and feature maps |
| 15–25 min | Output shapes, stride, padding, dilation and parameter count |
| 25–33 min | Pooling, receptive fields and CNN design trade-offs |
| 33–42 min | PyTorch Conv2d and a complete shape trace |
| 42–54 min | Run the synthetic image classification lab |
| 54–60 min | Production architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Convolutional layer | A layer that applies learned spatial filters with local connectivity and shared weights |
| Kernel / filter | A small bank of learned weights applied over an input region |
| Input channel | One input plane, such as grayscale intensity or an RGB channel |
| Output channel | One learned feature map produced by a filter |
| Feature map | Spatial responses produced by one learned filter |
| Stride | Number of positions the filter moves between applications |
| Padding | Values added around the input boundary |
| Dilation | Spacing between kernel elements |
| Receptive field | The input region that can influence one output activation |
| Weight sharing | Reusing the same filter parameters at every spatial position |
| Pooling | A local aggregation/downsampling operation |
| NCHW | Batch, channels, height, width tensor layout |
| Depthwise convolution | A grouped convolution that applies spatial filters independently per input channel |
| Pointwise convolution | A 1×1 convolution used to mix information across channels |

## 1. Why convolution?

A dense layer over a flattened image has no built-in notion that adjacent pixels are neighbors. It can learn spatial patterns, but it must learn separate weights for many positions.

A CNN introduces two useful assumptions:

- **Local connectivity:** a filter initially observes a small neighborhood.
- **Weight sharing:** the same filter is reused at every location.

This creates a useful inductive bias for images, spectrograms and other grid-like data. A local edge or texture can be detected regardless of where it appears.

A convolutional network often builds a hierarchy:

1. Early layers respond to local edges, orientations or short patterns.
2. Intermediate layers combine these into textures and shapes.
3. Deeper layers combine patterns into more task-specific representations.

This hierarchy is learned from data; a filter is not automatically an edge detector just because it is in the first layer.

### Convolution versus mathematical convolution

Most deep-learning libraries implement the cross-correlation operation in their Conv2d modules: the kernel is applied without flipping it first. The operation is conventionally called convolution in deep learning. Training learns the kernel weights either way, but the distinction matters when manually checking a numerical example.

## 2. A worked 2D cross-correlation example

Input:

\[
X=
\begin{bmatrix}
1&2&0\\
0&1&3\\
2&1&1
\end{bmatrix}
\]

Kernel:

\[
K=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix}
\]

Use stride 1 and no padding. Apply the 2×2 kernel at each valid location, multiply element-wise and sum:

\[
Y=
\begin{bmatrix}
0&-1\\
-1&0
\end{bmatrix}
\]

The output is 2×2 because a 2×2 kernel fits in four positions over a 3×3 input.

The kernel is a learned detector. During training, gradient descent changes its weights to reduce the task loss.

## 3. Channels, filters and feature maps

A PyTorch Conv2d input normally has shape:

\[
[N,C_{in},H,W]
\]

Where:

- N is batch size.
- Cin is input channel count.
- H and W are spatial dimensions.

A Conv2d layer configured with 1 input channel and 8 output channels learns 8 filters. Each filter spans the full input-channel depth and produces one output feature map.

For an RGB image, Cin is 3. A filter with a 3×3 spatial kernel therefore has 3×3×3 weights, plus an optional bias.

The output has shape:

\[
[N,C_{out},H_{out},W_{out}]
\]

A feature map is not a class label. It is an intermediate learned representation.

## 4. Output-size calculation

For one spatial dimension:

\[
L_{out}=
\left\lfloor
\frac{L_{in}+2P-D(K-1)-1}{S}+1
\right\rfloor
\]

Where:

- Lin = input length.
- K = kernel size.
- S = stride.
- P = padding on each side.
- D = dilation.

Apply the same calculation independently to height and width.

### Worked shape trace

Input batch:

\[
[64,1,16,16]
\]

Layer 1: Conv2d(1, 8, kernel=3, stride=1, padding=1)

\[
[64,8,16,16]
\]

ReLU preserves shape. MaxPool2d(2) halves each spatial dimension:

\[
[64,8,8,8]
\]

Layer 2: Conv2d(8, 16, kernel=3, stride=1, padding=1)

\[
[64,16,8,8]
\]

ReLU preserves shape. MaxPool2d(2):

\[
[64,16,4,4]
\]

Flattening gives 16×4×4 = 256 features per example.

A linear layer maps 256 features to 32 hidden units, then the final layer maps 32 units to 3 class logits.

## 5. Parameter counting

For a standard Conv2d layer with bias:

\[
Params=C_{out}\left(C_{in}K_hK_w+1\right)
\]

For the first layer, Conv2d(1, 8, 3×3):

\[
8(1\cdot3\cdot3+1)=80
\]

For the second layer, Conv2d(8, 16, 3×3):

\[
16(8\cdot3\cdot3+1)=1168
\]

The dense classifier has:

- Linear(256, 32): 256×32 + 32 = 8,224 parameters.
- Linear(32, 3): 32×3 + 3 = 99 parameters.

Total:

\[
80+1168+8224+99=9571
\]

Pooling and ReLU have no trainable parameters.

The parameter count is small, but the number of multiply-accumulate operations also depends on output spatial dimensions. Parameter count alone is not a complete latency or memory estimate.

## 6. Stride, padding and dilation

| Setting | Effect | Typical reason |
|---|---|---|
| Stride 1 | Evaluates nearby positions densely | Preserve spatial detail |
| Stride 2 | Moves two positions at a time | Downsample while convolving |
| Padding 0 / valid | Does not add border values | Reduce spatial dimensions |
| Padding 1 for 3×3, stride 1 | Preserves height and width | Keep spatial resolution |
| Dilation greater than 1 | Spreads kernel taps apart | Increase receptive field without a larger dense kernel |

Padding affects how border locations are treated. Zero padding is common, but reflect, replicate and circular modes may be more suitable for some signal boundaries. Choose based on the data-generating process rather than habit.

Dilation expands the effective receptive field. With kernel size K and dilation D, the effective kernel extent is D(K−1)+1.

## 7. Pooling and downsampling

Max pooling keeps the largest activation in a local region. Average pooling keeps the mean.

For a 2×2 max-pooling window:

\[
\begin{bmatrix}
1&4\\
2&3
\end{bmatrix}
\rightarrow 4
\]

Pooling reduces spatial resolution and activation memory. It can provide tolerance to small shifts, but it also discards precise location information.

Pooling is not mandatory in every CNN. Alternatives include strided convolutions, adaptive pooling and global average pooling. For localization tasks, aggressive downsampling may destroy important spatial detail.

## 8. Receptive field and hierarchical features

The receptive field is the part of the original input that can influence a particular activation.

Stacking convolutional layers increases the receptive field. Downsampling layers make it grow faster in input coordinates.

A deeper CNN can combine larger-context information, but more depth is not automatically better. The model must have enough data, suitable optimization and appropriate regularization.

## 9. PyTorch Conv2d

A compact CNN:

~~~python
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
~~~

Input contract: NCHW float tensor with shape [N, 1, 16, 16]. Output: [N, 3] logits.

For multiclass classification, use CrossEntropyLoss with integer class labels. Do not apply softmax before CrossEntropyLoss; it expects raw logits and applies a numerically stable log-softmax internally.

## 10. Practical lab — synthetic shape classification

The lab creates three image classes:

- Horizontal strokes.
- Vertical strokes.
- Diagonal strokes.

Each image is 16×16 grayscale with randomized position, thickness, intensity and additive noise. The data is generated from independent seeds for training, validation and test splits.

### Dataset

| Split | Samples |
|---|---:|
| Training | 1,200 |
| Validation | 300 |
| Test | 300 |
| Classes | 3 |
| Image tensor | [N, 1, 16, 16] |

The CNN is:

\[
Conv(1,8,3)\rightarrow ReLU\rightarrow MaxPool(2)
\]

\[
Conv(8,16,3)\rightarrow ReLU\rightarrow MaxPool(2)
\]

\[
Flatten\rightarrow Linear(256,32)\rightarrow ReLU\rightarrow Linear(32,3)
\]

The model has 9,571 trainable parameters.

### Reproduced reference run

Environment: PyTorch 2.10.0 CPU build, deterministic algorithms enabled, one CPU thread.

| Measurement | Result |
|---|---:|
| Best validation accuracy | 100.00% |
| Held-out test accuracy | 100.00% |
| Held-out test loss | 0.000239 |
| Trainable parameters | 9,571 |
| Test examples | 300 |

Confusion matrix (rows = actual class; columns = predicted class):

| Actual \ Predicted | Horizontal | Vertical | Diagonal |
|---|---:|---:|---:|
| Horizontal | 100 | 0 | 0 |
| Vertical | 0 | 100 | 0 |
| Diagonal | 0 | 0 | 100 |

**Interpretation:** this synthetic task has clean, deliberately separated patterns. Perfect accuracy demonstrates that the implementation can learn this controlled task; it does not predict performance on real images, spectrograms or silicon telemetry. Real deployments need device- or session-separated test sets, realistic noise and operating modes, and independent data.

### Run the lab

~~~bash
cd daily-lessons/day-018
python -m pip install -r requirements.txt
python lab.py
~~~

The script selects CUDA when available, otherwise CPU. Use --device cpu or --device cuda to override. The reference run used CPU; CUDA performance is not claimed or hardware-validated here.

### Exercises

1. Change the first convolution from 8 output channels to 4 and 16. Record parameter count, validation accuracy and runtime.
2. Replace MaxPool2d with stride-2 convolution. Compare output shapes and results.
3. Change padding from 1 to 0. Update the dense input size and explain the spatial shrinkage.
4. Use a 5×5 kernel in the first layer. Calculate the output size before running the model.
5. Add a fourth class, such as a cross, and update the final classifier and dataset generator.
6. Inspect the first-layer filters and intermediate feature maps. Do not assume a filter has a human-readable meaning without testing it.
7. Create a 1D CNN for a sampled waveform or a 2D CNN for a spectrogram. Keep the sampling rate, FFT/window/hop settings and normalization version in the input contract.

## 11. Connecting CNNs to audio and silicon diagnostics

A CNN is not limited to camera images.

### 1D CNN for waveforms

Input shape:

\[
[N,C,L]
\]

A 1D convolution slides across time. It can learn local patterns in sampled audio, sensor streams or protocol traces.

### 2D CNN for spectrograms

A waveform can be transformed into a time-frequency representation such as an STFT magnitude or log-mel spectrogram.

Input shape:

\[
[N,C,F,T]
\]

Where F is frequency bins and T is time frames.

A 2D CNN can learn local time-frequency patterns. For audio silicon diagnostics, candidate signals include codec output, microphone captures, loopback recordings and known-good versus failing acoustic signatures.

The representation pipeline is part of the model contract: sample rate, channel order, window length, hop length, FFT size, mel filter version, log scaling, normalization and crop policy must be versioned.

Do not arbitrarily reshape tabular counters into an image just to use Conv2d. Use a 1D CNN for ordered sequences, a 2D CNN for meaningful time-frequency or spatial structure, and an MLP/tree model for ordinary tabular features unless there is evidence for a structured representation.

## 12. Production pattern A24 — CNN and Signal-Representation Training Gate

### Architecture

~~~mermaid
flowchart TD
    A[Engineer objective] --> B[Read-only intake agent]
    B --> C[Typed modality and dataset contract]
    C --> D[Identity, schema and split validator]
    D --> E[Versioned representation pipeline]
    E --> F{Input modality router}
    F -->|Image| G[2D CNN worker]
    F -->|Waveform| H[1D CNN worker]
    F -->|Spectrogram| I[2D time-frequency CNN worker]
    G --> J[Deterministic trainer]
    H --> J
    I --> J
    J --> K[Metrics and model artifact]
    K --> L[Independent verifier]
    L --> M[Validation and robustness gate]
    M --> N{Candidate acceptable?}
    N -->|No| O[Bounded retry or reject]
    N -->|Yes| P[Frozen candidate]
    P --> Q[Protected test gate]
    Q --> R[Human approval]
    R --> S[Model registry]
~~~

### Component boundaries

| Component | Responsibility | Authority boundary |
|---|---|---|
| Intake agent | Clarify task and propose modality | No dataset, code execution or hardware authority |
| Contract validator | Validate shape, channels, sample rate, labels and budget | Reject unknown or out-of-policy fields |
| Representation pipeline | Resize/crop/normalize or compute waveform/spectrogram features | Versioned deterministic transform |
| Modality router | Select an approved 1D, 2D or tabular path | Routing is a proposal; policy validates it |
| CNN worker | Execute approved architecture and training code | Access only authorized training data and compute |
| Evaluator | Compute declared validation/test metrics | Test access only after candidate freeze |
| Verifier | Check shape, finite values, artifact lineage and robustness | Deterministic independent checks |
| Approval/registry gate | Record approval and register artifact | Human owner and release policy required |

### Seven agent patterns

| Pattern | A24 application | Trade-off and guardrail |
|---|---|---|
| Deterministic workflow | Validate → transform → train → evaluate → package | Default for all numerical operations |
| Single agent + typed tools | Read metrics, inspect feature maps and summarize experiments | Start read-only; no arbitrary code execution |
| Router-specialist | Route waveform, spectrogram, image or tabular tasks | Use modality contracts and deterministic fallback |
| Planner–executor–verifier | Plan bounded architecture/augmentation experiments | Persist plan; cap trials, GPU time and wall clock |
| Supervisor–worker | Parallel independent seeds or architecture candidates | Control accelerator memory and concurrency |
| Event-driven | Queue training jobs and publish checkpoint/evaluation events | Idempotent job IDs, durable state and bounded retries |
| Human-in-the-loop | Approve data changes, thresholds and model promotion | Auditable reviewer, rationale and expiry |

### Typed input contract example

~~~json
{
  "run_id": "audio-cnn-018",
  "task": "three_class_shape_demo",
  "modality": "image",
  "tensor_layout": "NCHW",
  "input_shape": [1, 16, 16],
  "dtype": "float32",
  "normalization_version": "unit-range-v1",
  "label_schema": "orientation-v1",
  "dataset_version": "synthetic-shapes-v1",
  "split_manifest_id": "seeded-split-018",
  "architecture_id": "small-cnn-1d8-8d16",
  "budget": {
    "max_wall_seconds": 600,
    "max_gpu_hours": 0.1,
    "max_concurrent_runs": 1
  }
}
~~~

A production audio contract would also include sample rate, channel mapping, FFT/window/hop configuration, frequency scale and feature-extraction version. Validate all dimensions and resource limits before scheduling.

### State, memory and artifacts

Persist:
- Dataset fingerprint and split manifest.
- Label schema and class mapping.
- Input modality, layout and preprocessing version.
- Model architecture/configuration hash.
- Training seed, optimizer and scheduler configuration.
- Checkpoint hash, code commit and environment digest.
- Validation metrics, confusion matrix and robustness results.
- Test evaluation record after candidate freeze.
- Reviewer decision and registry version.

Conversation memory is not model state. Do not use an LLM transcript as a substitute for a reproducible experiment manifest.

### Security and reliability

- Treat filenames, images, logs and dataset metadata as untrusted content.
- Keep dataset ACLs and accelerator quotas in deterministic services.
- Use typed allow-listed tools; never expose arbitrary shell/Python execution to the model.
- Verify artifact hashes and load only trusted checkpoints.
- Isolate tenants and redact private audio/images and identifiers from prompts and traces.
- Make job submission and event consumption idempotent.
- Use atomic checkpoint writes and bounded retries.
- Separate training, validation and protected test permissions.
- Require independent verification and human approval for registry promotion.

### Evaluation and observability

Evaluate more than aggregate accuracy:
- Per-class precision, recall and confusion matrix.
- Device/session-separated generalization.
- Robustness to noise, amplitude changes, time shifts and missing channels.
- Calibration and abstention behavior where predictions drive triage.
- Inference latency, throughput, peak memory and preprocessing cost.
- Model, dataset, representation and code lineage.

For audio diagnostics, measure false alarms per device-hour and missed-event rate, not just image-style accuracy.

### Cost and latency

- Use a small CPU smoke test before allocating GPU time.
- Profile preprocessing, host-to-device copies, convolution, dense layers and data loading separately.
- Bound batch size, image/spectrogram resolution, worker count and concurrent jobs.
- Benchmark channels-last memory format and mixed precision only on supported hardware and with numerical checks.
- Use an LLM at experiment boundaries, not per training batch or per input frame.
- Compare a CNN against a simpler baseline on the same splits before accepting the additional compute and operational complexity.

## 13. Five review questions

1. Why does weight sharing reduce the number of parameters compared with an independently parameterized dense layer over every spatial location?
2. For a 16×16 input, 3×3 kernel, stride 1 and padding 1, what is the output spatial size?
3. What is the difference between an input channel and an output feature map?
4. Why can aggressive pooling be harmful for localization or short transient detection?
5. Why must sample rate, spectrogram settings and normalization be versioned together with an audio CNN?

## 14. Recommended references

1. [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
2. [PyTorch MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
3. [Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
4. [Deep Learning, Goodfellow, Bengio and Courville — Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
5. [PyTorch Channels Last Memory Format](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

## Takeaways

- CNNs use local connectivity and shared filters to learn spatially reusable patterns.
- Stride, padding, dilation and pooling determine output shapes and information retention.
- Feature maps are learned representations, not labels or explanations by themselves.
- 1D CNNs suit ordered waveforms; 2D CNNs suit images and meaningful time-frequency representations.
- For silicon diagnostics, the preprocessing and acquisition contract is as important as the model architecture.
- Agents can help plan and explain experiments, but deterministic code owns training, data permissions, evaluation and release.
