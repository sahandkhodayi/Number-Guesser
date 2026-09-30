# Number Guesser

A from-scratch learning project connecting machine learning, C systems programming, computer vision, and deployment.

The project trains a CNN on MNIST with PyTorch, exports the learned weights into a binary format, and runs the same network in native C. A Raylib application lets you draw a digit and send it through the C inference pipeline.

## Architecture

    MNIST
      ↓
    PyTorch CNN
      ↓
    training
      ↓
    weights.bin
      ↓
    C inference engine
      ↓
    Raylib UI
      ↓
    draw → preprocess → predict

## Current model

    1×28×28
    → Conv 1→32, 3×3, padding 1
    → ReLU
    → Conv 32→32, 3×3, padding 1
    → ReLU
    → MaxPool 2×2
    → Conv 32→32, 3×3, padding 1
    → ReLU
    → Conv 32→32, 3×3, padding 1
    → ReLU
    → MaxPool 2×2
    → Flatten 1568
    → Linear 1568→10

## Repository

    python/                 PyTorch training/evaluation/export
    c/                      native C inference + Raylib application
    benchmark/              Python↔C comparison/benchmark tooling
    models/                 exported model artifacts
    tests/                  automated tests to be expanded
    fullguide.md            next-stage project guide
    foundations-reference.md completed implementation reference

## Important: the project is already built

The current project already has the core CNN and C inference implementation:

- Tensor storage and indexing
- Linear
- ReLU
- Conv2D
- MaxPool2D
- model loading
- full C forward pass
- Raylib drawing UI
- drawing preprocessing
- PyTorch training
- model export
- evaluation
- Python/C intermediate-tensor benchmark

These are not fake future checklist items. They are existing foundations.

## Project guide

**Start with [fullguide.md](fullguide.md).**

The main guide now assumes the completed foundations above and focuses on the next engineering stages:

1. automatic Python↔C numerical parity
2. C unit tests
3. sanitizers and memory correctness
4. robust model serialization
5. preprocessing tests
6. personal handwriting evaluation
7. error analysis
8. controlled experiments
9. C performance profiling
10. architecture improvements
11. multi-digit recognition and OCR

If you need the old beginner-oriented code explanations, use [foundations-reference.md](foundations-reference.md). You do not need to read that document from the beginning again.

## Current next milestone

The next task is automatic Python↔C numerical parity:

1. feed exactly the same input to PyTorch and C
2. dump corresponding intermediate tensors
3. compare them automatically
4. find the first divergent layer
5. fail the test when the difference exceeds tolerance

Then move to the C test suite and sanitizers.

## Philosophy

> Do not claim something is correct because it runs. Measure it, test it, and understand why.

This repository is a laboratory for learning, not a collection of code to copy.
