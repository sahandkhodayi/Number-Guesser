# Number Guesser

A from-scratch learning project connecting machine learning, C systems programming, computer vision, and deployment.

The project trains a convolutional neural network (CNN) on MNIST with PyTorch, exports its weights, and runs the network in native C. A Raylib desktop application lets you draw a digit and send it through the C inference pipeline.

## Project at a glance

```text
MNIST data
   │
   ▼
PyTorch training ──► checkpoint (.pth)
                           │
                           ▼
                     weight exporter
                           │
                           ▼
                    binary weights (.bin)
                           │
                           ▼
                 native C inference engine
                           ▲
                           │
                 Raylib drawing application
```

Python is responsible for training, evaluation, and exporting. C is responsible for running inference in the desktop application. The Python/C parity benchmark compares intermediate numerical results so that the two implementations can be checked against each other.

## Repository layout

| Path | Purpose |
|---|---|
| `python/` | Dataset loading, training, evaluation, model definition, weight export, and handwritten-digit collection |
| `c/include/`, `c/src/` | C inference API, CNN operations, model loading, preprocessing, and Raylib UI |
| `benchmark/` | Python reference outputs and C outputs used by the numerical parity benchmark |
| `tests/` | C unit tests and Python/C parity test driver |
| `models/` | Generated model checkpoints and exported weights; these artifacts are not committed by default |
| `main.py` | Interactive project menu |
| `Makefile` | Short commands for common development tasks |
| `CMakeLists.txt` | CMake build configuration for the desktop app |
| `fullguide.md` | Detailed code-first project guide |
| `workflow-guide.md` | Explanation of the menu and Makefile workflow |
| `foundations-reference.md` | Reference material for existing implementations |
| `docs/` | Architecture, setup, testing, and model/data documentation |

## Quick start

The desktop GUI requires Python dependencies, a C compiler, CMake, and a Raylib package that CMake can find. The exact Raylib installation steps depend on your platform and toolchain.

### 1. Create a Python environment

Windows PowerShell:

```powershell
py -m venv .venv
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

If the PyTorch installation command fails, use the installation selector at [pytorch.org](https://pytorch.org/get-started/locally/) for your operating system and hardware, then install the remaining dependencies. The desktop app additionally needs Raylib configured for CMake.

### 2. Inspect available commands

```bash
python main.py
```

On Windows, GNU Make is not included by default. Use a Make installation, WSL, or invoke the corresponding Python/CMake commands directly.

Common commands from the repository root:

| Command | Purpose |
|---|---|
| `make help` | List Makefile targets |
| `make menu` | Open the interactive menu |
| `make train` | Train and save a PyTorch checkpoint |
| `make evaluate` | Evaluate the saved checkpoint without training |
| `make export` | Export the checkpoint for C inference |
| `make build` | Configure and build the Raylib app |
| `make run` | Build and launch the app |
| `make test` | Run the configured parity and C unit tests |
| `make clean` | Remove generated build artifacts |

**Platform status:** the headless C unit tests and parity benchmark now use CMake to select the active compiler and executable format. The desktop GUI still requires Raylib configured for the chosen compiler, and Windows/Linux execution must be verified on those platforms before claiming full support. See [Development setup](docs/DEVELOPMENT.md) and [Testing](docs/TESTING.md).

## Correct model workflow

1. Train when you intentionally want to update the checkpoint: `make train`.
2. Evaluate the saved checkpoint: `make evaluate`.
3. Export the checkpoint to the binary format consumed by C: `make export`.
4. Build and run the desktop application: `make run`.

Training and export are separate operations. After retraining, export again before expecting the C application to use the new weights. Evaluation should not train or update the checkpoint.

Generated files such as datasets, checkpoints, exported weights, and benchmark binaries are ignored by Git. A fresh clone therefore does not necessarily contain a trained model; train and export one before launching the GUI.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — components and data flow.
- [Development setup](docs/DEVELOPMENT.md) — environments, tools, commands, and current platform caveats.
- [Testing](docs/TESTING.md) — what the existing tests check and what still needs coverage.
- [Model and data](docs/MODEL_AND_DATA.md) — tensor shapes, preprocessing, checkpoint/export contract, and artifact lifecycle.
- [Full project guide](fullguide.md) — deeper explanations, implementation work, and future improvements.
- [Workflow guide](workflow-guide.md) — menu and Makefile design.
- [Foundations reference](foundations-reference.md) — reminders for code that is already implemented.

## Current status and limitations

The core CNN, C inference engine, drawing UI, preprocessing, model export, C unit tests, and Python/C numerical benchmark exist. The next engineering work is to make builds and tests reliable across Windows and Linux/WSL, expand automated coverage, validate model artifacts strictly, and add CI.

The current parity test checks intermediate tensors for its configured input and tolerance. Passing it is evidence of agreement for those tested stages and inputs; it is not proof of general correctness, memory safety, or equivalent accuracy on all handwritten digits.

## Engineering principle

Do not claim a component is correct merely because it runs. Measure it, test it, and document the conditions under which the result was verified.
