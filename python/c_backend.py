"""Optional Python wrapper around the C inference shared library."""

from ctypes import CDLL, POINTER, c_float, c_int, pointer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CBackend:
    """Call the C CNN through a tiny ctypes FFI boundary."""

    def __init__(self, library_path: str | Path) -> None:
        self.lib = CDLL(str(library_path))
        self.lib.number_guesser_predict.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_int)
        ]
        self.lib.number_guesser_predict.restype = c_int

    def predict(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        """Run C inference on one normalized 28x28 float32 image."""
        array = np.ascontiguousarray(image, dtype=np.float32).reshape(784)
        logits = np.empty(10, dtype=np.float32)
        prediction = c_int(-1)

        status = self.lib.number_guesser_predict(
            array.ctypes.data_as(POINTER(c_float)),
            logits.ctypes.data_as(POINTER(c_float)),
            pointer(prediction),
        )
        if status != 0:
            raise RuntimeError(f"C backend returned status {status}")

        return prediction.value, logits
