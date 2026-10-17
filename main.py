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


def find_executable(build_dir, names):
    """Find a CMake executable in single- or multi-configuration build layouts."""
    for name in names:
        matches = sorted(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def run_app():
    executable_names = ("number_guesser.exe", "number_guesser")
    build_dir = ROOT / "build"
    executable = find_executable(build_dir, executable_names)

    if executable is None:
        print("Application is not built yet. Building it first...")
        build_app()
        executable = find_executable(build_dir, executable_names)

    if executable is None:
        raise SystemExit(
            "Could not find the built application. Check CMake and Raylib."
        )

    # Keep build/ as the working directory because the C app currently loads
    # ../models/weights.bin relative to the process working directory.
    run_command([str(executable)], cwd=build_dir)


def run_tests():
    build_dir = ROOT / "build"

    # Use CMake/CTest so the selected compiler and executable suffix are
    # handled by the active platform/toolchain.
    run_command([
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-DBUILD_GUI=OFF",
    ])
    run_command(["cmake", "--build", str(build_dir), "--target", "test_nn"])
    run_command(["ctest", "--test-dir", str(build_dir), "--output-on-failure"])

    # Parity builds its own headless benchmark target through CMake.
    run_python_script("tests/test_parity.py")


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