# C / Raylib Runtime

This is the main application runtime for Number Guesser.

The C program owns:
- drawing
- preprocessing
- CNN inference
- prediction
- probability display

Python is used for training, evaluation, export, and reference/benchmark tooling.

## Prerequisites

- C compiler
- CMake >= 3.20
- raylib
- exported model at `models/weights.bin`

The repository contains raylib headers for development, but the raylib library itself is a system dependency. The official raylib site provides a Windows installer and current reference material.

## Ubuntu / WSL

Ubuntu provides a `libraylib-dev` development package in its repositories. citeturn1search0turn1search8

Install:

```bash
sudo apt update
sudo apt install build-essential cmake libraylib-dev
```

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j
./build/number_guesser
```

The model path is resolved relative to the repository root, so run the executable from the repository root.

## Windows + MinGW

Install:

1. MinGW-w64/GCC
2. CMake
3. raylib

The official raylib website provides a Windows installer.

Then configure CMake using a generator/toolchain that can find your raylib installation.

Example:

```powershell
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
.\build\number_guesser.exe
```

If CMake cannot find raylib, configure its installation prefix/toolchain according to your raylib installation.

## Model export

If `models/weights.bin` does not exist, first train/evaluate/export from Python.

From the repository root:

```bash
python python/train.py
python python/evaluate.py
python python/export.py
```

Do not retrain merely to run the C application if a known-good exported model already exists locally.

## Controls

- **Left mouse button:** draw
- **C:** clear
- **Enter:** predict
- **Esc / window close:** quit

The application preprocesses the drawing before passing a `1 × 28 × 28` float32 tensor into the C CNN.

## Important

The C UI is the intended final runtime. The Python GUI is a development/reference tool.

If C and Python disagree, use the benchmark/parity pipeline before changing the model architecture.
