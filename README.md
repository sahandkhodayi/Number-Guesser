# Number Guesser

A from-scratch learning project connecting machine learning, C systems programming, computer vision, and deployment.

The project trains a CNN on MNIST with PyTorch, exports its weights, and runs the same network in native C. A Raylib application lets you draw a digit and send it through the C inference pipeline.

## Architecture

    MNIST → PyTorch CNN → exported weights → C inference → Raylib UI

## Repository

- `python/` — training, evaluation, export, and personal digit collection
- `c/` — native C inference and Raylib application
- `benchmark/` — Python/C numerical comparison
- `models/` — model artifacts
- `tests/` — automated tests
- `main.py` — interactive project menu (follow the workflow guide to add it)
- `Makefile` — common development commands (follow the workflow guide to add it)
- `fullguide.md` — code-first project textbook for testing, reliability, and new capabilities
- `workflow-guide.md` — code-first guide for a user-friendly entry point and Makefile
- `foundations-reference.md` — reference for existing implementations

## Current state

The core CNN, C inference engine, drawing UI, preprocessing, model export, and Python/C tensor benchmark already exist. The next stage is to make the system reliable and measurable, not to repeat a beginner CNN tutorial.

## Guides

- Start with [fullguide.md](fullguide.md) for automated parity, C unit tests, sanitizers, model-file validation, preprocessing, handwriting evaluation, error analysis, profiling, and future features.
- Follow [workflow-guide.md](workflow-guide.md) to add a root-level menu and Makefile for training, evaluation, exporting, tests, and running the GUI.
- Use [foundations-reference.md](foundations-reference.md) only when you need to revisit code that is already implemented.

## Important workflow distinction

Training updates `models/number_guesser_model.pth`. Evaluation loads that checkpoint without training. Export writes `models/weights.bin` for the C application. After retraining, export again before running the C GUI if you want it to use the new weights.

> Do not claim something is correct because it runs. Measure it, test it, and understand why.
