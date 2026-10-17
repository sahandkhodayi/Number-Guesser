from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
BUILD_DIR = ROOT / "build" / "parity"
STAGES = [
    "input",
    "conv1", "relu1",
    "conv2", "relu2",
    "pool1",
    "conv3", "relu3",
    "conv4", "relu4",
    "pool2",
    "logits",
]
ATOL = 1e-4


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_floats(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(f"Missing benchmark output: {path}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0:
        raise AssertionError(f"Empty benchmark output: {path}")
    if not np.isfinite(values).all():
        raise AssertionError(f"NaN or infinity in {path}")
    return values


def compare_stage(stage: str) -> None:
    expected = load_floats(BENCHMARK / f"pytorch_{stage}.bin")
    actual = load_floats(BENCHMARK / f"c_{stage}.bin")

    if expected.shape != actual.shape:
        raise AssertionError(
            f"{stage}: element-count mismatch: "
            f"Python={expected.shape}, C={actual.shape}"
        )

    diff = np.abs(expected - actual)
    worst = int(np.argmax(diff))
    maximum = float(diff[worst])
    mean = float(diff.mean())

    print(
        f"{stage:8} max={maximum:.8g} "
        f"mean={mean:.8g} worst_index={worst}"
    )

    if maximum > ATOL:
        raise AssertionError(
            f"{stage}: max difference {maximum:.8g} exceeds {ATOL}; "
            f"Python={expected[worst]}, C={actual[worst]}"
        )


def main() -> None:
    # Regenerate reference tensors from the current model checkpoint.
    run([sys.executable, "benchmark/run_pytorch.py"])

    # Configure/build through CMake so the compiler and executable suffix
    # are selected by the active platform/toolchain.
    run([
        "cmake", "-S", str(ROOT), "-B", str(BUILD_DIR),
        "-DBUILD_GUI=OFF",
    ])
    run(["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "c_benchmark"])

    executable_names = ("c_benchmark.exe", "c_benchmark")
    executable = next(
        (path for name in executable_names for path in sorted(BUILD_DIR.rglob(name))),
        None,
    )
    if executable is None:
        raise FileNotFoundError(
            f"Could not find CMake-built benchmark executable in {BUILD_DIR}"
        )

    # C reads benchmark/pytorch_input.bin, the exact Python input.
    run([str(executable)])

    print(f"\nComparing all stages with absolute tolerance {ATOL:g}")
    for stage in STAGES:
        compare_stage(stage)

    print("\nPARITY PASS: every stage is within tolerance.")


if __name__ == "__main__":
    main()