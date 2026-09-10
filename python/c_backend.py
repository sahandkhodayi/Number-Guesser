"""Small Python wrapper for the optional C inference backend.

The recommended first integration boundary is a shared library loaded with
ctypes. The C API stays independent from the GUI: Python passes 784 float32
pixels and receives 10 logits.

This wrapper is intentionally isolated so the GUI can use PyTorch while the C
backend is being verified. Once libnumber_guesser is built, the GUI can switch
backends without changing drawing/preprocessing code.
"""

from ctypes import CDLL, POINTER, c_float, c_int, c_char_p
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CBackend:
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
            POINTER(c_int)(prediction),
        )
        if status != 0:
            raise RuntimeError(f"C backend returned status {status}")
        return prediction.value, logits
