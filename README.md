# Number Guesser

> **A from-scratch ML systems project: PyTorch training, C inference, Raylib UI, and numerical verification between the two.**

Number Guesser is a handwritten-digit recognition system built as a long-term **career keystone project** for learning machine learning, mathematics, C, systems programming, and ML deployment.

The central idea is simple:

```
Draw a digit
    ↓
Preprocess it to the model input contract
    ↓
Run the trained CNN in C
    ↓
Inspect activations / logits
    ↓
Predict 0–9
```

The project deliberately goes below the usual `model(x)` abstraction.

---

## Current architecture

The current model is a small CNN:

```
1 × 28 × 28
    ↓
Conv2D  1  → 32   kernel=3, stride=1, padding=1
ReLU
Conv2D 32  → 32   kernel=3, stride=1, padding=1
ReLU
MaxPool           kernel=2, stride=2
    ↓
32 × 14 × 14
    ↓
Conv2D 32  → 32   kernel=3, stride=1, padding=1
ReLU
Conv2D 32  → 32   kernel=3, stride=1, padding=1
ReLU
MaxPool           kernel=2, stride=2
    ↓
32 × 7 × 7
    ↓
Flatten → 1568
    ↓
Linear → 10 logits
    ↓
argmax → digit
```

PyTorch is the **training/reference implementation**.

C is the intended **deployment/inference implementation**.

The long-term target is:

```
Raylib/C UI
    ↓
C preprocessing
    ↓
C CNN
    ↓
prediction
```

Python should eventually be needed for training, evaluation, export, and development tooling—not normal application inference.

---

## Why C?

This project is intentionally not just a PyTorch demo.

The C runtime exposes the important operations directly:

- tensor storage
- contiguous memory
- pointers
- allocation and ownership
- convolution
- ReLU
- max pooling
- matrix multiplication
- flattening
- weight loading
- numerical comparison
- inference API design

The goal is not to claim that C is inherently better than PyTorch.

The goal is to understand **what PyTorch is doing underneath the abstraction** and then build a small deployment runtime ourselves.

---

## Repository structure

```
Number-Guesser/
├── python/                 # Training, evaluation, export and reference tools
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   ├── gui.py              # Development/reference GUI
│   └── c_backend.py        # Optional ctypes bridge
│
├── c/                      # Native inference/runtime
│   ├── include/
│   ├── src/
│   └── tools/
│
├── benchmark/              # PyTorch ↔ C numerical verification
│   ├── run_pytorch.py
│   ├── c_benchmark.c
│   ├── compare.py
│   └── README.md
│
├── tests/                  # Unit/integration tests
├── data/                   # Local datasets; not committed
├── models/                 # Generated model artifacts
│
├── AGENTS.md               # AI-agent engineering guide + full roadmap
├── requirements.txt
├── LICENSE
└── README.md
```

---

# The most important engineering contract

## PyTorch is the reference. C is the deployment implementation.

A correct prediction is not enough to claim that the implementations match.

The benchmark compares:

```
input
 ↓
conv1
 ↓
relu1
 ↓
conv2
 ↓
relu2
 ↓
pool1
 ↓
conv3
 ↓
relu3
 ↓
conv4
 ↓
relu4
 ↓
pool2
 ↓
logits
```

For every stage we want:

- matching shape
- max absolute difference
- mean absolute difference
- first stage exceeding tolerance

If the first mismatch is `conv1`, debug `conv1` before touching later layers.

The benchmark is a core part of the project, not an optional debugging script.

---

# Input contract

The CNN expects:

```
shape:  1 × 28 × 28
dtype:  float32
range:  normalized grayscale values
```

Hand-drawn input is a separate engineering problem because a user's drawing does not naturally look exactly like MNIST.

The preprocessing pipeline therefore needs to be treated as a tested component:

```
canvas
 ↓
foreground detection
 ↓
bounding box
 ↓
square crop
 ↓
centering
 ↓
resize
 ↓
28 × 28
 ↓
normalization
 ↓
CNN
```

Known concern: **domain shift** between MNIST and arbitrary mouse drawings.

Do not fix this by randomly changing the CNN. First verify the preprocessing and compare the final 28×28 tensor.

---

# Weight serialization

The Python exporter currently writes the trained tensors in a fixed order:

1. Conv1 weights
2. Conv1 bias
3. Conv2 weights
4. Conv2 bias
5. Conv3 weights
6. Conv3 bias
7. Conv4 weights
8. Conv4 bias
9. Linear weights
10. Linear bias

The current binary model is expected to be **175016 bytes**.

This is a compatibility contract between Python and C.

Future work should move from a raw implicit binary to a small versioned model format with:

- magic number
- format version
- architecture identifier
- dtype
- tensor count
- tensor metadata
- payload
- optional checksum

---

# Roadmap

## Phase 0 — Baseline

- [x] Train CNN
- [x] Save trained weights
- [x] Export C-friendly weights
- [x] Implement C CNN inference
- [x] Build benchmark tooling
- [x] Build initial GUI

Next:

- [ ] Record reproducible PyTorch test accuracy
- [ ] Run parity on representative MNIST samples
- [ ] Record current hand-drawing accuracy separately
- [ ] Make benchmark failures return non-zero status
- [ ] Document numerical tolerance

## Phase 1 — Prove C/PyTorch parity

- [ ] Compare every intermediate tensor
- [ ] Test all ten classes
- [ ] Test multiple MNIST samples
- [ ] Validate tensor shapes
- [ ] Validate weight ordering
- [ ] Validate flatten ordering
- [ ] Validate Conv2D semantics
- [ ] Validate MaxPool semantics
- [ ] Add regression tests

**Gate:** C and PyTorch agree within a documented tolerance.

## Phase 2 — Make preprocessing rigorous

- [ ] Define the preprocessing contract
- [ ] Implement the same preprocessing in C
- [ ] Compare Python/C preprocessing outputs
- [ ] Test blank images
- [ ] Test shifted digits
- [ ] Test small/large digits
- [ ] Test stroke-width variation
- [ ] Visualize the exact 28×28 tensor
- [ ] Measure accuracy on controlled handwritten examples

**Gate:** preprocessing is deterministic and tested.

## Phase 3 — Make C the real application runtime

- [ ] Raylib drawing canvas
- [ ] C preprocessing
- [ ] C inference
- [ ] prediction + probability display
- [ ] clear/reset
- [ ] prediction history
- [ ] runtime error handling
- [ ] no Python dependency for normal inference

## Phase 4 — Make the CNN inspectable

- [ ] Show 28×28 model input
- [ ] Show Conv1 feature maps
- [ ] Show later feature maps
- [ ] Show pooling outputs
- [ ] Show logits
- [ ] Show softmax probabilities
- [ ] Show per-layer timing
- [ ] Add debug/inspection mode

This is where the project becomes more than a digit classifier: users can see the computation happen.

## Phase 5 — Testing and systems quality

- [ ] Tensor indexing tests
- [ ] ReLU tests
- [ ] Linear-layer tests
- [ ] Conv2D tests
- [ ] MaxPool tests
- [ ] preprocessing tests
- [ ] model-loader tests
- [ ] end-to-end inference tests
- [ ] C/PyTorch parity tests
- [ ] AddressSanitizer
- [ ] UndefinedBehaviorSanitizer
- [ ] strict compiler warnings
- [ ] static analysis
- [ ] CI
- [ ] reproducible build

## Phase 6 — Profile, then optimize

Measure first:

- preprocessing latency
- each CNN layer latency
- total inference latency
- allocations per prediction
- model load time
- peak memory

Then investigate:

- reusable buffers
- fewer heap allocations
- cache-friendly loop order
- contiguous memory
- compiler optimization
- SIMD/vectorization
- optional parallelism

No optimization should be accepted without a measured reason.

## Phase 7 — Model/deployment engineering

- [ ] version model files
- [ ] validate architecture compatibility
- [ ] validate tensor shapes
- [ ] validate dtype
- [ ] validate tensor count
- [ ] checksum model artifacts
- [ ] reproducible export command
- [ ] clear errors for incompatible models

## Phase 8 — ML experiments

Only after the deployment/runtime is trustworthy:

- [ ] architecture experiments
- [ ] augmentation experiments
- [ ] normalization experiments
- [ ] optimizer comparisons
- [ ] learning-rate experiments
- [ ] ablation studies
- [ ] quantization
- [ ] inference optimization

Every experiment should record:

```
Hypothesis
Dataset
Change
Training configuration
Metric
Result
Conclusion
```

---

# Definition of done

A feature is not done because it compiles.

It is done when:

- the behavior works,
- the relevant tests exist,
- the relevant benchmark passes,
- documentation matches reality,
- failures are handled,
- generated artifacts are controlled,
- the change is reproducible.

For inference changes, "done" additionally requires numerical evidence against PyTorch.

---

# Career-project north star

The final project should demonstrate one coherent story:

> **I trained a neural network in PyTorch, understood the mathematics behind it, exported the learned parameters, implemented the inference engine in C, proved the C runtime against the reference implementation, built the preprocessing and native runtime, tested it, profiled it, and exposed the internal computation so other people can understand what the network is doing.**

That is more valuable than simply having a high MNIST accuracy number.

---

# Documentation

- **[AGENTS.md](AGENTS.md)** — detailed instructions for AI agents, engineering rules, milestone gates, and the complete implementation roadmap.
- **[benchmark/README.md](benchmark/README.md)** — numerical parity methodology.

If you are an AI agent working in this repository, read **AGENTS.md first**.

---

# License

MIT
