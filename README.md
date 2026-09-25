# Number Guesser

> A from-scratch ML systems project: PyTorch training, C inference, and a Raylib C application.

Number Guesser recognizes handwritten digits with a small CNN trained in PyTorch and executed by a native C inference runtime.

```text
Mouse drawing
     ↓
C preprocessing
     ↓
1 × 28 × 28 float32 tensor
     ↓
C CNN
     ↓
10 logits
     ↓
prediction + probabilities
```

## Current architecture

```text
1×28×28
  ↓ Conv 1→32
  ↓ ReLU
  ↓ Conv 32→32
  ↓ ReLU
  ↓ MaxPool 2×2
  ↓ Conv 32→32
  ↓ ReLU
  ↓ Conv 32→32
  ↓ ReLU
  ↓ MaxPool 2×2
  ↓ Flatten 1568
  ↓ Linear 1568→10
  ↓ logits
```

PyTorch is the training/reference implementation. C is the intended application/deployment runtime.

## Quick start — C application

Run commands from the **repository root**.

### Ubuntu / WSL

Ubuntu provides a `libraylib-dev` development package. citeturn1search0turn1search8

```bash
sudo apt update
sudo apt install build-essential cmake libraylib-dev
cmake -S . -B build
cmake --build build -j
./build/number_guesser
```

### Windows + MinGW

Install GCC/MinGW, CMake, and raylib. The official raylib site provides a Windows installer. citeturn0search0

Then from the repository root:

```powershell
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
.\build\number_guesser.exe
```

If CMake cannot find your raylib installation, configure its install prefix/toolchain for your local setup.

### Model file

The C application expects:

```text
models/weights.bin
```

This is generated from the trained PyTorch model:

```bash
python python/export.py
```

If you do not already have a trained model, train it first:

```bash
python python/train.py
python python/evaluate.py
python python/export.py
```

Generated model files are intentionally ignored by Git.

## Controls

- **Left mouse button:** draw
- **C:** clear
- **Enter:** predict
- **Esc / window close:** quit

## Repository structure

```text
Number-Guesser/
├── python/          # PyTorch training, evaluation and export
├── c/               # Native CNN + Raylib application
├── benchmark/       # PyTorch ↔ C numerical verification
├── tests/            # Automated tests
├── models/           # Local generated model files
├── data/             # Local MNIST data
├── CMakeLists.txt
├── PROJECT_GUIDE_PROMPT.md
└── README.md
```

## Numerical correctness

The project is not considered complete merely because C and PyTorch predict the same digit.

The benchmark should compare:

```text
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

At every stage we want shape equality and numerical error measurements.

## Long-term roadmap

1. Prove C/PyTorch numerical parity.
2. Make preprocessing deterministic and tested.
3. Make C/Raylib the complete runtime.
4. Visualize intermediate CNN activations.
5. Add unit/integration tests, sanitizers and CI.
6. Profile before optimizing.
7. Improve the model format and reproducibility.
8. Run controlled ML experiments.

## Full guidebook prompt

**[PROJECT_GUIDE_PROMPT.md](PROJECT_GUIDE_PROMPT.md)** contains the prompt for another AI to write the full step-by-step project textbook/roadmap.

That guide is intentionally separate from agent instructions: it asks another AI to produce educational documentation covering the code, mathematics, debugging, testing, benchmarking, and future roadmap.

## Philosophy

> Correctness before optimization.
>
> Measured truth before claims.
>
> Understanding before abstraction.
>
> C is the application runtime; Python is the training/reference environment.

The goal is not just to make a digit classifier. The goal is to understand and build the complete path from mathematical model to trained parameters to native inference running on the machine.