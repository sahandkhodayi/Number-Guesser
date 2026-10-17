# Architecture

## Purpose

Number Guesser trains a small convolutional neural network (CNN) using PyTorch, exports its parameters to a flat binary file, and performs inference in native C. A Raylib desktop interface captures a handwritten digit and displays the model's prediction.

The project intentionally separates model training from native inference: the C program does not need PyTorch at runtime.

## Component map

| Component | Responsibility |
|---|---|
| `python/dataset.py` | Downloads/loads MNIST and creates PyTorch data loaders |
| `python/model.py` | Defines the CNN architecture used by training, evaluation, and export |
| `python/train.py` | Trains the model and saves a checkpoint |
| `python/evaluate.py` | Loads a checkpoint and measures test-set loss and accuracy |
| `python/export.py` | Serializes parameters in a specific order for the C loader |
| `python/digit_collector.py` | Collects processed user-drawn digit samples |
| `c/include/nn.h`, `c/src/nn.c` | Tensor storage, layer operations, model loading, and forward inference |
| `c/src/ui.c` | Drawing canvas and conversion from drawn pixels to model input |
| `c/src/main.c` | Raylib event loop, model inference, and prediction visualization |
| `benchmark/run_pytorch.py` | Produces reference intermediate tensors from PyTorch |
| `benchmark/c_benchmark.c` | Runs the C model on the matching benchmark input and emits intermediate tensors |
| `tests/test_nn.c` | Unit tests for selected C operations |
| `tests/test_parity.py` | Runs the reference and C benchmark, then compares intermediate outputs |

## Data flow

```text
MNIST
  │
  ▼
python/dataset.py ──► python/model.py ──► python/train.py
                                             │
                                             ▼
                              models/number_guesser_model.pth
                                             │
                                             ▼
                                     python/export.py
                                             │
                                             ▼
                                    models/weights.bin
                                             │
                                             ▼
User drawing ──► C preprocessing ──► C CNN forward pass ──► logits
                                                               │
                                                               ▼
                                                   prediction + display
```

The checkpoint and exported binary are different artifacts. The checkpoint is a PyTorch state dictionary; the binary is a flat sequence of float parameters in the order expected by the C model loader.

## Model shape contract

The current model uses one input channel and 28×28 input images. Its main shape flow is:

```text
[B, 1, 28, 28]
  → convolution/ReLU blocks
  → max pool: [B, 32, 14, 14]
  → convolution/ReLU blocks
  → max pool: [B, 32, 7, 7]
  → flatten: [B, 1568]
  → linear classifier: [B, 10]
```

Here `B` is the batch size. The C interactive application processes one drawing at a time, while PyTorch training usually processes batches.

## Critical cross-language contracts

Changes to any of these require coordinated updates and tests:

1. **Architecture:** convolution count, channel counts, kernel size, padding, pooling, flatten order, and classifier dimensions.
2. **Weight order:** `python/export.py` writes tensors in an explicit order; `CnnModel` and `model_load()` must read the same order.
3. **Tensor layout:** C uses contiguous channel-first storage with index `((c * height) + y) * width + x`.
4. **Numerical behavior:** convolution and pooling definitions, padding, stride, activation placement, and float precision must match PyTorch.
5. **Input preprocessing:** the drawing UI must produce 28×28 float input with the expected pixel orientation, range, background, and centering.
6. **Output interpretation:** the C application uses logits for argmax and applies softmax for the displayed probabilities.

## Runtime paths

Some existing code uses paths relative to the current working directory. For example, the C GUI currently expects `../models/weights.bin` when launched with the build directory as its working directory. Keep this in mind when launching the executable manually or from an IDE.

A future reliability improvement should make the model path explicit or derive it from a well-defined application/data directory instead of relying on the process working directory.

## What this architecture does not guarantee

- Numerical parity on a benchmark does not prove every operation is correct for every input.
- Softmax values are not automatically calibrated probabilities of correctness.
- MNIST test accuracy does not guarantee good performance on drawings from every person or input device.
- A successful build does not establish memory safety; sanitizer and edge-case testing are separate tasks.
