# Number Guesser

A from-scratch handwritten digit recognizer:

```text
Mouse drawing
     ↓
28 × 28 grayscale image
     ↓
C CNN inference (hand-written, no ML libraries)
     ↓
10 digit classes (0–9)
     ↓
Raylib UI: prediction + confidence
```

Trained in PyTorch, re-implemented from scratch in C — no ONNX Runtime, no
libtorch, no NN libraries of any kind. If it runs, we wrote the math.

## Project Goals

- Learn PyTorch by building the training pipeline.
- Train a CNN on MNIST handwritten digits.
- Understand preprocessing, batching, loss, optimization, and evaluation.
- Export the trained network weights to a transparent binary format.
- Implement CNN inference in C by hand: conv2d, relu, maxpool, linear, argmax.
- Build a Raylib interface for drawing and predicting digits.
- Visualize the network and its activations as part of the final application.

## Structure

```text
number-guesser/
├── data/          # Local datasets (not committed)
├── models/        # Generated model weights (not committed)
├── python/        # PyTorch training and export code
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py           ← empty placeholder, this is Part 4
│   └── helper_functions.py
├── c/             # C inference and Raylib application
│   ├── include/
│   │   ├── nn.h
│   │   └── ui.h
│   └── src/
│       ├── main.c
│       ├── nn.c
│       └── ui.c
├── tests/         # Tests for math, preprocessing, and inference
├── main.py        # Temporary project entry point (currently empty)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Status (verified against actual code, not assumed)

### Step 0 — Project setup
- [x] Repository structure
- [x] Python/C directories
- [x] Dataset and model directories
- [x] Test directory
- [x] Initial dependency file

### Step 1 — MNIST pipeline
- [x] Download/load MNIST (`dataset.py`, via `torchvision.datasets.MNIST`)
- [x] Build Dataset/DataLoader pipeline (batch size 64, shuffled train set)
- [ ] Inspect image and label batches
- [x] Preprocessing — see note below: it's simpler than "normalize/flatten"

### Step 2 — PyTorch model
- [x] Implement CNN — **not** the MLP originally planned, see §Architecture
- [x] Implement training loop functions (`train_step`, `test_step`)
- [ ] **Real** training loop — `evaluate.py` currently runs `train_step`
      and `test_step` exactly **once** (one epoch). Decide if that's
      intentional (smoke test) before treating any exported weights as final.
- [x] Save `.pth` weights — `evaluate.py` does `torch.save(model.state_dict(), ...)`
      → `models/number_guesser_model.pth` (not yet present — needs a run)

### Step 3 — C inference
- [ ] Export weights into a C-friendly format (`export.py` — spec below, not implemented)
- [ ] Implement matrix multiplication / Linear
- [ ] Implement ReLU
- [ ] Implement Conv2D
- [ ] Implement MaxPool
- [ ] Implement forward pass
- [ ] Implement argmax prediction

### Step 4 — Raylib application
- [ ] Drawing canvas
- [ ] Convert drawing to 28 × 28 input
- [ ] Predict button
- [ ] Prediction/confidence display
- [ ] Network visualization

**Correction to the original plan:** the README used to describe a
`784 → 128 → 64 → 10` MLP. `model.py` has since become a CNN. This document
now reflects the real architecture.

---

## Architecture (inspected from `model.py`, not assumed)

`_MainModel(input_shape=1, hidden_units=32, output_shape=10)`:

```text
Input: 1×28×28  (grayscale, single channel)

block_1:
  Conv2d(1  → 32, k=3, s=1, p=1)   → 32×28×28
  ReLU
  Conv2d(32 → 32, k=3, s=1, p=1)   → 32×28×28
  ReLU
  MaxPool2d(k=2, s=2)              → 32×14×14

block_2:
  Conv2d(32 → 32, k=3, s=1, p=1)   → 32×14×14
  ReLU
  Conv2d(32 → 32, k=3, s=1, p=1)   → 32×14×14
  ReLU
  MaxPool2d(k=2, s=2)              → 32×7×7

classifier:
  Flatten                          → 1568
  Linear(1568 → 10)                → 10 logits
```

Padding=1 with a 3×3 stride-1 kernel preserves H×W, so only the two
`MaxPool2d(2)` layers shrink spatial size (28 → 14 → 7). That answers the
`# Where did this in_features shape come from?` comment in `model.py`:
`hidden_units*7*7 = 32*7*7 = 1568`.

### Parameters to export

| Layer | `state_dict` key | Weight shape | Weight count | Bias count |
|---|---|---|---|---|
| Conv1 | `block_1.0` | [32, 1, 3, 3] | 288 | 32 |
| Conv2 | `block_1.2` | [32, 32, 3, 3] | 9,216 | 32 |
| Conv3 | `block_2.0` | [32, 32, 3, 3] | 9,216 | 32 |
| Conv4 | `block_2.2` | [32, 32, 3, 3] | 9,216 | 32 |
| Linear | `classifier.1` | [10, 1568] | 15,680 | 10 |

(`nn.Sequential` indices `.0`/`.2` — ReLU and MaxPool have no parameters but
still occupy an index slot.)

**Total: 43,754 float32 parameters → 175,016 bytes (~171 KB).**

---

## Preprocessing (inspected from `dataset.py`)

```python
transform = transforms.ToTensor()
```

That's the entire transform — no `Normalize`, no mean/std subtraction, no
inversion, no resizing. `ToTensor()` does exactly two things the C side must
replicate:

1. Single-channel float tensor, shape `(1, 28, 28)`.
2. Pixel values scaled to **`[0.0, 1.0]`** by dividing by 255.

MNIST digits are white-stroke-on-black-background. C preprocessing is
therefore just: grayscale pixel (0–255) → `pixel / 255.0f`. If the Raylib
canvas is drawn white-on-black (matching MNIST), no inversion is needed —
pinned down for real in the Raylib preprocessing step.

---

## Weight export format (target — `export.py` not implemented yet)

A single `weights.bin`, no header: the 5 parameter tensors written
back-to-back as raw little-endian float32, in this fixed order:

```
conv1_w (288)   conv1_b (32)
conv2_w (9216)  conv2_b (32)
conv3_w (9216)  conv3_b (32)
conv4_w (9216)  conv4_b (32)
fc_w    (15680) fc_b (10)
```

Total file size: exactly **175,016 bytes**. Since the architecture is fixed
and known on both ends, the C loader reads floats in this order into
pre-sized arrays — no parsing needed. A file size that isn't exactly
175,016 bytes is an instant sanity check that something's wrong.

`Conv2d.weight` in PyTorch is `[out_channels, in_channels, kH, kW]`,
row-major/C-contiguous — the same layout the C convolution will assume, so
export should be a straight `tensor.numpy().tobytes()` per parameter with no
reordering. This gets verified for real by comparing intermediate outputs
layer-by-layer, C vs PyTorch (Step 3 verification, not yet built).

---

## Toolchain

| Tool | Status |
|---|---|
| `gcc` | available |
| `raylib` | not installed yet — needed before the Raylib UI step, not before |
| PyTorch / training | happens on your machine, outside this chat's sandbox |

## Philosophy

No black boxes where understanding matters. The project is intentionally
built in stages so the mathematics, PyTorch implementation, and final C
inference remain understandable — build a piece, compile it, test it against
a hand-verifiable example, then move on.
