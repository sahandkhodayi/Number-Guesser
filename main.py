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