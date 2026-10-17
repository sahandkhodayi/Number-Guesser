# Number-Guesser — One-Command Workflow Guide

This is the implementation guide for the root-level menu and Makefile, which are now present in the repository. It is retained as a code-first explanation of their design; it is not a list of files that still need to be created. It does **not** rewrite the CNN, training loop, evaluation loop, C inference, or tests.

> **Maintenance note:** this guide preserves the original implementation walkthrough, so some code examples may predate the newer CMake/CTest portability work. Treat the current `main.py`, `Makefile`, `CMakeLists.txt`, and `tests/test_parity.py` as the source of truth when an example differs.

The goal is to make common tasks discoverable and repeatable:

- launch the interactive application;
- train a new PyTorch checkpoint;
- evaluate an existing checkpoint without training;
- export the trained checkpoint for C inference;
- run the existing tests;
- build the C/Raylib application.

## Important distinction: train is not evaluate

- `train` changes model weights and saves `models/number_guesser_model.pth`.
- `evaluate` loads the saved checkpoint and measures it on the test set. It must not update weights.
- `export` converts the checkpoint into `models/weights.bin`, which the C application loads.
- `run` launches the C GUI. It does not train the network.

This separation avoids accidentally training when you only want to check a model.

---

# Chapter 1 — Root-level `main.py`

## Goal

Give the user one command and a small menu instead of remembering which file to launch.

## Why

The project already has separate Python scripts. That separation is good: training, evaluation, and export have different responsibilities. The root entry point should coordinate them, not duplicate their implementation.

We will call the existing scripts as subprocesses. That keeps their current import behavior and makes errors visible in the terminal.

## Reference implementation — `main.py`

```python
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_command(command, *, cwd=ROOT):
    """Run a command and propagate its exit status."""
    print("\\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_python_script(relative_path):
    script = ROOT / relative_path
    if not script.is_file():
        raise SystemExit(f"Required script not found: {script}")
    run_command([sys.executable, str(script)])


def train():
    run_python_script("python/train.py")


def evaluate():
    run_python_script("python/evaluate.py")


def export_model():
    run_python_script("python/export.py")


def collect_digits():
    run_python_script("python/digit_collector.py")


def build_app():
    run_command(["cmake", "-S", str(ROOT), "-B", str(ROOT / "build")])
    run_command(["cmake", "--build", str(ROOT / "build")])


def run_app():
    executable_names = (
        ["number_guesser.exe", "number_guesser"]
        if sys.platform == "win32"
        else ["number_guesser", "number_guesser.exe"]
    )

    build_dir = ROOT / "build"
    executable = next(
        (build_dir / name for name in executable_names
         if (build_dir / name).is_file()),
        None,
    )

    if executable is None:
        print("Application is not built yet. Building it first...")
        build_app()
        executable = next(
            (build_dir / name for name in executable_names
             if (build_dir / name).is_file()),
            None,
        )

    if executable is None:
        raise SystemExit(
            "Could not find the built application. Check CMake and Raylib."
        )

    # The C app currently loads ../models/weights.bin relative to its
    # working directory, so run it from build/.
    run_command([str(executable)], cwd=build_dir)


def run_tests():
    # Run the Python/C numerical parity regression test.
    run_python_script("tests/test_parity.py")

    # Compile and run the C unit tests.
    test_executable = ROOT / "build" / (
        "test_nn.exe" if sys.platform == "win32" else "test_nn"
    )
    test_executable.parent.mkdir(parents=True, exist_ok=True)

    run_command([
        "gcc", "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
        str(ROOT / "tests/test_nn.c"),
        str(ROOT / "c/src/nn.c"),
        "-o", str(test_executable),
        "-lm",
    ])
    run_command([str(test_executable)])


ACTIONS = {
    "1": ("Run the drawing app", run_app),
    "2": ("Train the PyTorch model", train),
    "3": ("Evaluate the saved model", evaluate),
    "4": ("Export weights for C inference", export_model),
    "5": ("Collect your own handwritten digits", collect_digits),
    "6": ("Run tests", run_tests),
    "7": ("Build the C app", build_app),
}


def main():
    while True:
        print("\\n=== Number Guesser ===")
        for key, (description, _) in ACTIONS.items():
            print(f"{key}. {description}")
        print("0. Exit")

        choice = input("Choose an action: ").strip()

        if choice == "0":
            print("Goodbye.")
            return

        action = ACTIONS.get(choice)
        if action is None:
            print("Invalid choice. Enter a number from 0 to 7.")
            continue

        description, function = action
        print(f"\\n--- {description} ---")

        try:
            function()
        except subprocess.CalledProcessError as error:
            print(f"\\nCommand failed with exit code {error.returncode}.")
            print("Read the error above, fix the cause, then try again.")
        except FileNotFoundError as error:
            print(f"\\nRequired program was not found: {error.filename}")
            print("Check that Python, GCC, and/or CMake are installed and on PATH.")

        input("\\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
```

## Explanation

- `ROOT` is based on the location of `main.py`, not the terminal's current folder. Paths therefore work even when the user launches the script from another directory.
- `sys.executable` reuses the Python interpreter running the menu. This is safer than hardcoding `python`, especially when using a virtual environment.
- `subprocess.run(..., check=True)` reports failed commands rather than pretending they succeeded.
- The menu calls the existing scripts instead of copying their logic. There is still only one training implementation and one evaluation implementation.
- The GUI is run with `cwd=build` because the current C code loads `../models/weights.bin` relative to its working directory. If you later change model paths to be based on the executable or an explicit command-line argument, this constraint can be removed.
- The test action runs the existing parity script and compiles the existing C unit test. It assumes the parity script and GCC are configured for your platform; see the platform note below.

## Run

From the repository root:

```bash
python main.py
```

Try menu option 3 first if a trained checkpoint already exists. Try option 2 only when you intentionally want to train and replace the checkpoint.

## Debugging

- **Checkpoint not found:** run the training action first.
- **Python imports fail:** check that the scripts remain under `python/`; this entry point executes each script by path.
- **CMake cannot find Raylib:** install/configure Raylib for CMake before building the GUI.
- **The app cannot load weights:** check that `models/weights.bin` exists and that the app is launched with `build/` as its working directory.
- **GCC not found:** install a compiler and make sure it is on PATH.
- **Parity test fails:** fix the underlying parity problem; do not remove the test or loosen the tolerance merely to get a green result.

## Done means

- [ ] `python main.py` opens a menu.
- [ ] Invalid choices do not crash the menu.
- [ ] Training and evaluation are separate actions.
- [ ] Evaluation loads the saved checkpoint and does not train.
- [ ] The C app runs from the correct working directory.
- [ ] Failed commands are reported honestly.

---

# Chapter 2 — Root-level Makefile

## Goal

Expose common project tasks as short, memorable commands.

## Why

A menu is useful for a person exploring the project. A Makefile is useful for development, scripts, documentation, and automation. They serve different purposes and can coexist.

## Reference implementation — `Makefile`

Use tabs (not spaces) at the start of each recipe line.

```make
.DEFAULT_GOAL := help

PYTHON ?= python
CMAKE ?= cmake
BUILD_DIR := build
APP := $(BUILD_DIR)/number_guesser

.PHONY: help menu train evaluate export collect build run test parity c-test clean

help:
	@echo "Number Guesser commands:"
	@echo "  make menu      - open the interactive menu"
	@echo "  make train     - train and save the PyTorch model"
	@echo "  make evaluate  - evaluate the saved checkpoint (no training)"
	@echo "  make export    - export checkpoint weights for C inference"
	@echo "  make collect   - collect handwritten digit images"
	@echo "  make build     - configure and build the C/Raylib app"
	@echo "  make run       - build if needed and launch the drawing app"
	@echo "  make test      - run parity and C unit tests"
	@echo "  make parity    - run Python/C numerical parity test"
	@echo "  make c-test    - compile and run C unit tests"
	@echo "  make clean     - remove generated build/test artifacts"

menu:
	$(PYTHON) main.py

train:
	$(PYTHON) python/train.py

evaluate:
	$(PYTHON) python/evaluate.py

export:
	$(PYTHON) python/export.py

collect:
	$(PYTHON) python/digit_collector.py

build:
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR)

run: build
	cd $(BUILD_DIR) && ./number_guesser

parity:
	$(PYTHON) tests/test_parity.py

c-test:
	mkdir -p $(BUILD_DIR)
	gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_nn.c c/src/nn.c -o $(BUILD_DIR)/test_nn -lm
	./$(BUILD_DIR)/test_nn

test: parity c-test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
```

## Explanation

- `make train` and `make evaluate` are intentionally separate. Evaluation should not change model weights.
- `make export` must run after training if you want the C GUI to use the newest checkpoint. Training writes the PyTorch checkpoint; export writes the binary C model.
- `make run` depends on `build`, so Make builds the GUI first. The recipe then changes into the build directory to preserve the current C model-path behavior.
- `.PHONY` prevents files named `train`, `test`, or `build` from accidentally making Make skip the corresponding task.
- `PYTHON ?= python` allows you to override the interpreter, for example `make train PYTHON=python3`.
- `clean` removes generated build artifacts, not the saved model checkpoint or your source code.

## Run

```bash
make help
make train
make evaluate
make export
make build
make run
make test
```

### Platform note

This Makefile's `run` and `c-test` recipes use Unix-style executable paths and `mkdir -p`. Use it in Linux/WSL or a Unix-like shell with Make and GCC installed. On native Windows, use `python main.py` for the menu, or adapt the Makefile's executable names and shell commands for your installed toolchain. Do not assume that a Windows `.exe` can be launched using the Unix recipe unchanged.

The existing parity test also compiles and runs a C executable. Verify its compiler/output naming matches your current platform before relying on `make test`.

## Done means

- [ ] `make help` lists commands.
- [ ] `make train` trains and saves a checkpoint.
- [ ] `make evaluate` only evaluates the saved checkpoint.
- [ ] `make export` creates `models/weights.bin`.
- [ ] `make build` builds the Raylib application.
- [ ] `make run` launches the GUI from the correct working directory.
- [ ] `make test` runs both existing test layers.
- [ ] `make clean` leaves source files and model files untouched.

---

# Chapter 3 — Make the normal user workflow explicit

## Goal

Document the intended sequence so users don't accidentally run stale C weights or retrain when they only want to test the app.

## Recommended workflows

### Use the existing model

```bash
make build
make run
```

This uses the currently exported `models/weights.bin`.

### Train a new model and use it in C

```bash
make train
make evaluate
make export
make build
make run
```

Training updates `models/number_guesser_model.pth`. Evaluation reports the saved checkpoint's test performance. Export refreshes `models/weights.bin`. The GUI then uses the exported weights.

### Test changes without retraining

```bash
make test
```

Testing should not train a model. It should check the Python/C numerical agreement and C primitives. If a test fails, diagnose the first failing layer before changing architecture or model settings.

### Collect personal handwriting

```bash
make collect
```

Review the images and labels before training on them. The current training script uses MNIST through `python/dataset.py`; collecting images alone does **not** automatically add those images to training. Integrating the custom dataset safely is a separate feature: keep training, validation, and test sets separate, and ensure its preprocessing matches the MNIST convention.

## Final checklist

- [ ] New users can discover commands from `make help` or `python main.py`.
- [ ] Training, evaluation, export, testing, and GUI execution remain distinct.
- [ ] Documentation states which files each command reads and writes.
- [ ] No command silently claims success after a failed subprocess.
- [ ] The project still works when invoked from outside the repository root, where supported.
