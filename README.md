
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

    python/      PyTorch training/evaluation/export
    c/           native C inference + Raylib application
    benchmark/   Python↔C comparison/benchmark tooling
    models/      exported model artifacts
    tests/       automated tests to be expanded
    fullguide.md living project textbook and roadmap

## The important part

This is intentionally not just a model-training project.

The long-term goal is to understand:

- neural-network mathematics
- tensor shapes and memory layout
- C pointers and ownership
- convolution implementation
- binary model serialization
- numerical parity between two implementations
- testing and sanitizers
- image preprocessing
- ML evaluation and error analysis
- performance profiling
- multi-digit recognition
- OCR and sequence modeling

## Project guide

Read [fullguide.md](fullguide.md) as the main learning material.

It is kept synchronized with the actual repository state. Completed implementation work is treated as material to understand and verify rather than as a fake future checklist.

## Current next milestone

The next major engineering task is automatic Python↔C numerical parity:

1. feed exactly the same input to PyTorch and C
2. dump corresponding intermediate tensors
3. compare them automatically
4. find the first divergent layer
5. fail the test when the difference exceeds tolerance

After that, the project moves into a real C test suite, sanitizers, personal handwriting evaluation, preprocessing experiments, and eventually multi-digit OCR.

## Philosophy

> Do not claim something is correct because it runs. Measure it, test it, and understand why.

This repository is a laboratory for learning, not a collection of code to copy.


## Full project guide

**Start with [fullguide.md](fullguide.md).** It is the complete coding textbook for this repository.

For each major source file it is organized as:

1. what the file is for
2. the actual project code
3. line-by-line explanation
4. how it connects to the rest of the system
5. what to test and what to build next

The guide intentionally keeps completed code. Completed parts are there to be understood, tested, and used as the foundation for the next features.
