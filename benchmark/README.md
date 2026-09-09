# C ↔ PyTorch Benchmark

This folder exists to answer one question before we tune the Raylib UI:

> **Does the C inference implementation produce the same result as the trained PyTorch CNN?**

## Benchmark procedure

1. Select one real MNIST test image.
2. Convert it to the exact `float32` `[1, 28, 28]` input used by PyTorch.
3. Save those 784 values to `benchmark/input.bin`.
4. Run the PyTorch benchmark and dump intermediate tensors.
5. Run the C benchmark with the same `input.bin` and `models/weights.bin`.
6. Compare every stage and report the maximum absolute difference.

## What we are checking

```text
input
  ↓
Conv1 → ReLU
  ↓
Conv2 → ReLU → MaxPool
  ↓
Conv3 → ReLU
  ↓
Conv4 → ReLU → MaxPool
  ↓
Flatten → Linear
  ↓
logits → argmax
```

A small floating-point difference is expected. A large difference means the first stage where it appears is where we should debug.

## Important

This is **not** a training benchmark. PyTorch remains responsible for training. C is being tested as the inference/deployment implementation.

Generated binary files and benchmark output should remain untracked; only the scripts and documentation belong in Git.
