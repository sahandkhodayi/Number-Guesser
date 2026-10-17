# Development setup

This document describes the current setup and distinguishes existing behavior from work still needed. Windows and Linux/WSL are target platforms, but the current scripts are not yet fully cross-platform.

## Required tools

- Python compatible with the installed PyTorch and torchvision versions.
- pip and Python virtual environments.
- A C11 compiler.
- CMake 3.20 or newer, as required by `CMakeLists.txt`.
- Raylib available through a CMake package configuration.
- GNU Make for the provided Makefile.
- Git.

The Raylib package and compiler setup differ between Windows and Linux. Use the installation instructions for the package/toolchain you choose, and verify that CMake can locate Raylib before treating the desktop build as configured.

## Python environment

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation due to local execution policy, either follow Microsoft's guidance for your environment or invoke the environment's Python executable directly. Do not change machine-wide security policy just to run this project.

The current requirements file lists `torch`, `torchvision`, `numpy`, and `matplotlib`. The digit collector also imports Pillow. Add Pillow to the declared project dependencies as a separate reliability change rather than assuming it is present transitively.

For platform-specific PyTorch wheels, consult [the official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Current commands

Run these from the repository root.

| Command | Current purpose |
|---|---|
| `python main.py` | Interactive menu |
| `python python/train.py` | Train and save checkpoint |
| `python python/evaluate.py` | Evaluate saved checkpoint |
| `python python/export.py` | Export parameters to `models/weights.bin` |
| `cmake -S . -B build` | Configure CMake build |
| `cmake --build build` | Build the desktop application |
| `python tests/test_parity.py` | Run Python/C parity driver |
| `make help` | List Makefile targets, when GNU Make is installed |

The Makefile currently uses Unix-style commands such as `mkdir -p` and launches `./number_guesser`, so it should be treated as a Linux/WSL workflow for now. The parity test also invokes GCC and uses a `.exe` filename unconditionally. These are known platform gaps to fix; do not interpret the existence of a Makefile as proof that native Windows works.

## Recommended verification order

1. Confirm the active interpreter: `python --version` and `python -m pip --version`.
2. Confirm dependencies import: `python -c "import torch, torchvision, numpy; print(torch.__version__)"`.
3. Confirm tools are available: `cmake --version`, `gcc --version` or the chosen compiler, and `make --version`.
4. Configure CMake and resolve any Raylib package discovery error.
5. Run the existing C unit tests.
6. Train or obtain a compatible checkpoint, then export weights.
7. Run parity tests.
8. Launch the GUI from the working directory expected by the current model path.

## Known setup pitfalls

- **CMake cannot find Raylib:** install Raylib for the chosen compiler and make its CMake package config discoverable. A library installed for a different compiler/toolchain may not be usable.
- **Checkpoint missing:** evaluation and export require a saved checkpoint.
- **C weights missing:** the GUI needs `models/weights.bin`; generated artifacts are ignored by Git.
- **Wrong model path:** the current GUI loads a path relative to its working directory.
- **Python package mismatch:** install PyTorch and torchvision versions compatible with each other and your Python version.
- **No Make on Windows:** use WSL or install a suitable Make implementation; alternatively run the underlying Python/CMake commands directly.
- **Parity test cannot launch compiler:** currently it assumes GCC. Portability work should choose the compiler through CMake or a configurable environment variable.

## Safety when cleaning

The existing `make clean` target removes the CMake build directory. Before adding more cleanup targets, ensure they do not delete personal handwritten samples, datasets, or trained checkpoints without an explicit opt-in.
