"""Python GUI for Number Guesser.

The GUI owns drawing, preprocessing, PyTorch inference, and visualization.
The C implementation remains in c/ as a separate inference backend that can
be connected later through a shared library/FFI boundary.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

from model import _MainModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "number_guesser_model.pth"
CANVAS_SIZE = 280
MNIST_SIZE = 28


class NumberGuesserGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Number Guesser")
        self.root.resizable(False, False)

        self.model = self._load_model()
        self.last_prediction: int | None = None
        self.last_confidence = 0.0

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=4, padx=16, pady=16)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<Button-1>", self._draw)

        self.prediction = tk.StringVar(value="Draw a digit")
        self.confidence = tk.StringVar(value="")

        ttk.Label(root, textvariable=self.prediction, font=("Arial", 28, "bold")).grid(
            row=0, column=1, padx=16, pady=(30, 8)
        )
        ttk.Label(root, textvariable=self.confidence).grid(
            row=1, column=1, padx=16, pady=8
        )

        ttk.Button(root, text="Predict", command=self.predict).grid(
            row=2, column=1, padx=16, pady=8, sticky="ew"
        )
        ttk.Button(root, text="Clear", command=self.clear).grid(
            row=3, column=1, padx=16, pady=(8, 30), sticky="ew"
        )

    def _load_model(self) -> _MainModel:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def _draw(self, event: tk.Event) -> None:
        radius = 12
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )

    def _canvas_to_mnist(self) -> torch.Tensor:
        # Tkinter canvas is rasterized through a PostScript-free drawing buffer
        # below. For the first GUI version we use the canvas display itself as
        # the drawing surface and sample the visible pixels from its items.
        image = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) != 4:
                continue
            x0, y0, x1, y1 = coords
            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)
            r = max(1, int((x1 - x0) / 2))
            yy, xx = np.ogrid[:CANVAS_SIZE, :CANVAS_SIZE]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            image[mask] = 1.0

        ys, xs = np.where(image > 0.05)
        if len(xs) == 0:
            return torch.zeros((1, 1, MNIST_SIZE, MNIST_SIZE), dtype=torch.float32)

        # Crop the drawn digit and preserve its aspect ratio when resizing.
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        crop = image[y0:y1, x0:x1]

        side = max(crop.shape)
        square = np.zeros((side, side), dtype=np.float32)
        oy = (side - crop.shape[0]) // 2
        ox = (side - crop.shape[1]) // 2
        square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop

        tensor = torch.from_numpy(square)[None, None]
        tensor = torch.nn.functional.interpolate(
            tensor, size=(MNIST_SIZE, MNIST_SIZE), mode="bilinear", align_corners=False
        )
        return tensor.clamp(0.0, 1.0)

    @torch.inference_mode()
    def predict(self) -> None:
        x = self._canvas_to_mnist()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        digit = int(torch.argmax(probs).item())
        confidence = float(probs[digit].item())

        self.last_prediction = digit
        self.last_confidence = confidence
        self.prediction.set(str(digit))
        self.confidence.set(f"Confidence: {confidence:.2%}")

    def clear(self) -> None:
        self.canvas.delete("all")
        self.prediction.set("Draw a digit")
        self.confidence.set("")
        self.last_prediction = None
        self.last_confidence = 0.0


def main() -> None:
    root = tk.Tk()
    NumberGuesserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
