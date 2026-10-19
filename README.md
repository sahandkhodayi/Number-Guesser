# Number Guesser

A handwritten-digit recognition app built with **PyTorch, C, and Raylib**. Draw a digit, run it through a convolutional neural network, and see the prediction.

<!-- IMAGE 1 — PROJECT BANNER
What to add: a wide, clean image of the actual app window with a handwritten digit on the canvas and the predicted digit visible.
Suggested file: assets/banner.png
After adding it, replace this comment with:
<p align="center"><img src="assets/banner.png" alt="Number Guesser application" width="800"></p>
-->

## Demo


<p align="center">
  <video src="assets/demo.mp4" width="800" controls muted playsinline
         aria-label="Demo: draw a digit and see the model prediction"></video>
</p>

*Add a short recording here showing the complete experience: draw → predict → result.*

## How it works

<!-- IMAGE 2 — PIPELINE DIAGRAM (OPTIONAL)
What to add: a simple diagram with four boxes and arrows:
MNIST → PyTorch training → exported weights → native C inference → Raylib app.
This is optional; a real demo and screenshot are more important.
Suggested file: assets/pipeline.png
-->

1. **Train:** Python and PyTorch train a CNN on the MNIST handwritten-digit dataset.
2. **Export:** the trained model's weights are converted into a binary file that the C program can load.
3. **Predict:** the desktop app passes the drawn digit through the native C inference engine and displays the result.
4. **Check:** C unit tests and a Python/C numerical comparison help verify the implementations.

The project combines machine learning with lower-level systems programming: Python handles training, while C runs the model in the desktop app.

## Features

- Draw handwritten digits in a desktop application.
- Recognize digits using a CNN trained on MNIST.
- Run model inference in native C instead of using PyTorch for predictions in the app.
- Train, evaluate, and export the model using Python scripts.
- Compare selected Python and C intermediate values with a numerical parity test.
- Collect handwritten digit samples using the project's digit-collection tool.

## Tech stack

- **Python / PyTorch** — dataset, model training, evaluation, and weight export.
- **C** — neural-network inference and model loading.
- **Raylib** — desktop drawing interface.
- **CMake / Make** — build and development workflows.
- **CTest / GitHub Actions** — automated checks.

## Getting started

> **Note:** this repository currently provides the source code and developer workflow. It does not yet provide a verified, ready-to-run desktop download with all required files bundled. Building the GUI from source requires the development tools below.

### Requirements

- Python 3.10–3.12 recommended.
- A working PyTorch installation.
- CMake and a C compiler.
- Raylib installed and discoverable by CMake for the desktop GUI.

### 1. Get the source

```bash
git clone https://github.com/sahandkhodayi/Number-Guesser.git
cd Number-Guesser
```

Or download the repository ZIP from GitHub.

### 2. Install Python dependencies

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails, choose the appropriate command for your OS and hardware from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

### 3. Train and export the model

A fresh clone does not contain generated model weights. Run:

```bash
python python/train.py
python python/export.py
```

Training uses the MNIST dataset and may take a while. If you retrain the model later, export the weights again before running the C app.

### 4. Build the desktop app

Install Raylib first, then configure and build:

```bash
cmake -S . -B build -DBUILD_GUI=ON
cmake --build build --config Release
```

Run the generated `number_guesser` executable with `build/` as its working directory, because the app currently loads `../models/weights.bin` relative to the working directory. On Windows, the executable is normally under `build/Release/`.

You can also launch the interactive project menu:

```bash
python main.py
```

## Development commands

| Command | Purpose |
|---|---|
| `make train` | Train the PyTorch model |
| `make evaluate` | Evaluate the saved checkpoint |
| `make export` | Export weights for C inference |
| `make build` | Build the desktop app |
| `make run` | Build and open the project menu |
| `make test` | Run C unit tests and Python/C parity checks |
| `make clean` | Remove generated build artifacts |

GNU Make may not be installed on Windows by default. You can use WSL, install Make, or run the Python/CMake commands directly.

## Project structure

| Path | Purpose |
|---|---|
| `python/` | Model, training, evaluation, data helpers, and export |
| `c/include/`, `c/src/` | C inference engine and Raylib UI |
| `tests/` | C unit tests and Python/C parity test |
| `benchmark/` | Inputs and outputs for numerical comparison |
| `models/` | Generated model checkpoints and exported weights |
| `main.py` | Interactive project menu |
| `CMakeLists.txt`, `Makefile` | Build and development commands |

## Tests and documentation

The repository includes C unit tests, a Python/C numerical parity test, and a GitHub Actions workflow. These checks cover configured test cases; they do not guarantee correct predictions for every handwriting style or prove that the GUI works on every machine.

- [Architecture](docs/ARCHITECTURE.md)
- [Development setup](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Model and data](docs/MODEL_AND_DATA.md)
- [Docker and CI](docs/DOCKER_AND_CI.md)
- [Full project guide](fullguide.md)
- [Workflow guide](workflow-guide.md)

## Docker

Docker is an optional tool for developers and automated testing. The current image is designed for headless C tests; it does **not** run the Raylib desktop GUI. End users should not need Docker to use a future packaged desktop release.

## Roadmap

- [ ] Add a real demo recording and application screenshot.
- [ ] Package and test a ready-to-run Windows build.
- [ ] Package and test a ready-to-run Linux build.
- [ ] Publish a release with clear download and launch instructions.

---

Made as a learning project exploring neural networks, computer vision, C, and running a trained model outside its original Python framework.
