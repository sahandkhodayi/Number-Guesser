# Number Guesser

**Draw a handwritten digit. Watch a neural network recognize it.**  
Number Guesser is a small computer-vision project that connects a PyTorch model to a native C inference engine and a desktop drawing app.

> **Version 1.0.0 — release preparation**
> The source code and development workflow are in the repository. A genuinely easy-to-install release still needs tested, downloadable desktop builds and a bundled model file; see [the release checklist](#what-is-needed-for-an-easy-download).

<!--
SHOWCASE MEDIA — add these files when ready:
1. Record a short 10–20 second GIF showing: draw a digit -> prediction appears.
   Save as assets/demo.gif, then uncomment the line below.
2. Add a clean screenshot of the running app as assets/screenshot.png, then uncomment the line below.
![Number Guesser demo](assets/demo.gif)
![Number Guesser desktop app](assets/screenshot.png)
-->

## See it in action

**Demo video / GIF:** *Reserved for a short recording of drawing a digit and seeing the prediction.*

**Application screenshot:** *Reserved for a clear screenshot of the running desktop app.*

These are intentional placeholders so the README can be completed with real captures of the actual program—not mockups.

## What it does

- Trains a convolutional neural network (CNN) on the MNIST handwritten-digit dataset using PyTorch.
- Exports trained weights to a compact binary file.
- Runs the model's forward pass in native C.
- Provides a Raylib desktop UI where you can draw a digit and get a prediction.
- Includes a numerical parity test that compares selected intermediate values from the Python and C implementations.

### How the pieces fit together

```text
MNIST dataset
    |
    v
PyTorch model (Python) ---> checkpoint (.pth)
                                  |
                                  v
                           weight exporter
                                  |
                                  v
                           weights.bin
                                  |
                                  v
Draw a digit ---> Raylib UI ---> native C CNN ---> predicted digit
                                  ^
                                  |
                         C unit tests / parity checks
```

Python handles training, evaluation, and exporting. C handles inference in the desktop app.

## Getting started

### The easiest option

When downloadable desktop builds are published, you should be able to download the archive for your operating system, extract it, and run the application without installing Python, PyTorch, CMake, or a C compiler.

**Those ready-to-run Windows/Linux packages are not attached yet.** Until then, use the source setup below. I don't want to label a source-code checkout as a one-click release.

### Run from source

You'll need:
- Python 3.10–3.12 recommended (the CI workflow currently uses Python 3.12).
- Git, CMake, and a working C compiler.
- Raylib installed and discoverable by CMake to build the desktop GUI.
- A supported PyTorch installation. If installing PyTorch fails, choose the matching command at [pytorch.org](https://pytorch.org/get-started/locally/).

#### 1. Download the project

```bash
git clone https://github.com/sahandkhodayi/Number-Guesser.git
cd Number-Guesser
```

Or use GitHub's **Code → Download ZIP** option.

#### 2. Create an environment and install Python packages

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 3. Create the model files

A fresh checkout does not include generated model weights. Train the model and export the weights used by the C app:

```bash
python python/train.py
python python/export.py
```

Training downloads/uses the MNIST dataset and may take a while depending on your hardware. You only need to train again if you want to create a new checkpoint; the export step must be repeated after retraining.

#### 4. Build and launch the desktop app

Make sure Raylib is installed and CMake can find it, then run:

```bash
cmake -S . -B build -DBUILD_GUI=ON
cmake --build build --config Release
```

Launch the generated `number_guesser` executable from the `build` directory (on Windows it will normally be under `build/Release`). Running it with `build` as the working directory is important because the app currently loads `../models/weights.bin` relative to its working directory.

The repository's `main.py` menu can also run common tasks:

```bash
python main.py
```

## Useful development commands

| Command | What it does |
|---|---|
| `make train` | Train the PyTorch model |
| `make evaluate` | Evaluate the saved checkpoint |
| `make export` | Export weights for C inference |
| `make build` | Build the desktop app (requires Raylib) |
| `make run` | Build and open the project menu |
| `make test` | Run C unit tests and Python/C parity checks |
| `make clean` | Remove generated build artifacts |

On Windows, GNU Make may not be installed by default. You can use WSL, install Make, or run the Python/CMake commands directly.

## Testing and engineering

The repository includes C unit tests, a Python/C numerical parity test, and a GitHub Actions workflow. These checks cover the configured tests—not every possible image, all memory-safety issues, or successful GUI launch on every desktop.

- [Architecture](docs/ARCHITECTURE.md)
- [Development setup](docs/DEVELOPMENT.md)
- [Testing](docs/TESTING.md)
- [Model and data](docs/MODEL_AND_DATA.md)
- [Docker and CI](docs/DOCKER_AND_CI.md)
- [Full project guide](fullguide.md)
- [Workflow guide](workflow-guide.md)

## What is needed for an easy download?

Before calling this a polished v1.0.0 desktop release, the release should include:

- [ ] A short real demo GIF and a screenshot.
- [ ] A tested Windows build archive with the executable, required runtime files, and `models/weights.bin`.
- [ ] A tested Linux build archive, or a clearly stated list of supported Linux distributions.
- [ ] A simple first-run test on a clean machine that does not have the development environment installed.
- [ ] A release page with clear download names and short run instructions.
- [ ] Passing CI checks, plus a manual GUI smoke test on each platform advertised.

Until those are done, the source project is usable for developers, but the end-user install experience is not yet one-click.

## Why Docker is here

Docker is currently for **developers and CI**, not for someone who just wants to draw a digit. It gives the headless C tests a consistent environment and is useful for debugging/build automation. The Raylib drawing window is graphical, so the current Docker image does not package or launch the desktop UI.

**For an ordinary user, Docker should not be required.** The right end-user experience is a downloadable app bundle with the trained weights included. Docker can remain an optional developer/testing tool.

## Project structure

| Path | Purpose |
|---|---|
| `python/` | Model, training, evaluation, data helpers, and weight export |
| `c/include/`, `c/src/` | C inference engine and Raylib desktop UI |
| `tests/` | C unit tests and Python/C parity test |
| `benchmark/` | Numerical comparison inputs and outputs |
| `models/` | Generated checkpoints and exported weights |
| `main.py` | Interactive project menu |
| `CMakeLists.txt`, `Makefile` | Build and development commands |

---

Built as a learning project exploring neural networks, computer vision, C, and the path from a trained model to a native application.
