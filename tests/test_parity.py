from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
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

    # Build the C benchmark from the current source.
    executable = BENCHMARK / "c_benchmark.exe"
    run([
        "gcc", "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
        "benchmark/c_benchmark.c", "c/src/nn.c",
        "-o", str(executable), "-lm",
    ])

    # C reads benchmark/pytorch_input.bin, the exact Python input.
    run([str(executable)])

    print(f"\nComparing all stages with absolute tolerance {ATOL:g}")
    for stage in STAGES:
        compare_stage(stage)

    print("\nPARITY PASS: every stage is within tolerance.")


if __name__ == "__main__":
    main()