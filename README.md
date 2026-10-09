# Number Guesser

A from-scratch learning project connecting machine learning, C systems programming, computer vision, and deployment.

The project trains a CNN on MNIST with PyTorch, exports its weights, and runs the same network in native C. A Raylib application lets you draw a digit and send it through the C inference pipeline.

## Architecture

    MNIST → PyTorch CNN → exported weights → C inference → Raylib UI

## Repository

- `python/` — training, evaluation, export
- `c/` — native C inference and Raylib application
- `benchmark/` — Python/C numerical comparison
- `models/` — model artifacts
- `tests/` — automated tests
- `fullguide.md` — code-first project textbook
- `foundations-reference.md` — reference for existing implementations

## Current state

The core CNN, C inference engine, drawing UI, preprocessing, model export, and Python/C tensor benchmark already exist. The next stage is to make the system reliable and measurable, not to repeat a beginner CNN tutorial.

## Project guide

Start with [fullguide.md](fullguide.md). Each chapter gives the goal, why it matters, relevant files, implementation code, explanation, run commands, debugging guidance, and completion criteria. The roadmap covers automated numerical parity, C unit tests, sanitizers, model-file validation, preprocessing inspection, personal-handwriting evaluation, error analysis, controlled experiments, profiling, architecture cleanup, and multi-digit recognition.

Use [foundations-reference.md](foundations-reference.md) only when you need to revisit code that is already implemented.

## Next milestone

The current Python/C benchmark shows close numerical agreement for the test input. Next, automate that parity check so future changes cannot silently break it.

> Do not claim something is correct because it runs. Measure it, test it, and understand why.
