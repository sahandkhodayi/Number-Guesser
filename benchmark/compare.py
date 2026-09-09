"""Compare C and PyTorch benchmark tensors and find the first mismatch."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
STAGES = ["input", "conv1", "relu1", "conv2", "relu2", "pool1", "conv3", "relu3", "conv4", "relu4", "pool2", "logits"]


def load(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def main() -> None:
    print("Stage                 max_abs_diff       mean_abs_diff")
    print("-------------------------------------------------------")
    first_bad = None

    for stage in STAGES:
        py_path = ROOT / f"pytorch_{stage}.bin"
        c_path = ROOT / f"c_{stage}.bin"

        if not py_path.exists() or not c_path.exists():
            print(f"{stage:<22} MISSING")
            continue

        py = load(py_path)
        c = load(c_path)

        if py.shape != c.shape:
            print(f"{stage:<22} SHAPE MISMATCH: {py.shape} vs {c.shape}")
            if first_bad is None:
                first_bad = stage
            continue

        diff = np.abs(py - c)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(f"{stage:<22} {max_diff:<18.8g} {mean_diff:.8g}")

        # 1e-4 is a practical debugging threshold for this benchmark.
        if max_diff > 1e-4 and first_bad is None:
            first_bad = stage

    print()
    if first_bad:
        print(f"FIRST STAGE ABOVE TOLERANCE: {first_bad}")
        print("Debug this stage before changing the model architecture.")
    else:
        print("All available stages are within 1e-4 max absolute error.")
        print("C inference matches PyTorch closely; investigate UI preprocessing next.")


if __name__ == "__main__":
    main()
