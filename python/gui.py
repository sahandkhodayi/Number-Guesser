"""Python/Tkinter GUI for Number Guesser.

Python owns the drawing UI, preprocessing, and PyTorch inference.
The C CNN remains in c/ as an optional inference backend.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import _MainModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "number_guesser_model.pth"
CANVAS_SIZE = 280
MNIST_SIZE = 28
BRUSH_RADIUS = 12


class NumberGuesserGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Number Guesser")
        self.root.resizable(False, False)

        self.model = self._load_model()
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point: tuple[int, int] | None = None

        self.prediction = tk.StringVar(value="Draw a digit")
        self.confidence = tk.StringVar(value="")

        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=5, padx=16, pady=16)
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        ttk.Label(root, text="Prediction", font=("Arial", 12)).grid(
            row=0, column=1, padx=16, pady=(24, 4)
        )
        ttk.Label(root, textvariable=self.prediction, font=("Arial", 32, "bold")).grid(
            row=1, column=1, padx=16, pady=4
        )
        ttk.Label(root, textvariable=self.confidence).grid(
            row=2, column=1, padx=16, pady=4
        )
        ttk.Button(root, text="Predict", command=self.predict).grid(
            row=3, column=1, padx=16, pady=8, sticky="ew"
        )
        ttk.Button(root, text="Clear", command=self.clear).grid(
            row=4, column=1, padx=16, pady=(4, 24), sticky="ew"
        )

    def _load_model(self) -> _MainModel:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    def _start_draw(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        self._paint(event.x, event.y)

    def _draw(self, event: tk.Event) -> None:
        if self.last_point is None:
            self._start_draw(event)
            return
        x, y = event.x, event.y
        px, py = self.last_point
        self.draw.line((px, py, x, y), fill=255, width=BRUSH_RADIUS * 2)
        self._paint(x, y)
        self.last_point = (x, y)

    def _paint(self, x: int, y: int) -> None:
        r = BRUSH_RADIUS
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        self._refresh_canvas()

    def _stop_draw(self, _event: tk.Event) -> None:
        self.last_point = None

    def _refresh_canvas(self) -> None:
        # PPM lets Tk display the same PIL image that will be fed to the model.
        photo = tk.PhotoImage(data=self._image_to_ppm())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.image = photo

    def _image_to_ppm(self) -> bytes:
        rgb = self.image.convert("RGB")
        return f"P6 {rgb.width} {rgb.height} 255\n".encode("ascii") + rgb.tobytes()

    def _canvas_to_mnist(self) -> torch.Tensor:
        pixels = np.asarray(self.image, dtype=np.float32) / 255.0
        ys, xs = np.where(pixels > 0.05)
        if len(xs) == 0:
            return torch.zeros((1, 1, MNIST_SIZE, MNIST_SIZE), dtype=torch.float32)

        # Remove empty margins before resizing. This makes hand-drawn digits
        # closer to the centered MNIST-style input used during training.
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        crop = pixels[y0:y1, x0:x1]

        side = max(crop.shape)
        square = np.zeros((side, side), dtype=np.float32)
        oy = (side - crop.shape[0]) // 2
        ox = (side - crop.shape[1]) // 2
        square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop

        image = Image.fromarray((square * 255).astype(np.uint8), mode="L")
        image = image.resize((MNIST_SIZE, MNIST_SIZE), Image.Resampling.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(resized)[None, None]

    @torch.inference_mode()
    def predict(self) -> None:
        x = self._canvas_to_mnist()
        probs = torch.softmax(self.model(x), dim=1)[0]
        digit = int(torch.argmax(probs).item())
        confidence = float(probs[digit].item())
        self.prediction.set(str(digit))
        self.confidence.set(f"Confidence: {confidence:.2%}")

    def clear(self) -> None:
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.prediction.set("Draw a digit")
        self.confidence.set("")
        self._refresh_canvas()


def main() -> None:
    root = tk.Tk()
    NumberGuesserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
