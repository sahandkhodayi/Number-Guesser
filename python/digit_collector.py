
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw, ImageTk
import numpy as np


# -------------------- Configuration --------------------

CANVAS_SIZE = 280
OUTPUT_SIZE = 28
DIGIT_SIZE = 20
BRUSH_SIZE = 18

DATA_DIR = Path("my_digit_dataset")

# Create one folder for each digit.
for digit in range(10):
    (DATA_DIR / str(digit)).mkdir(parents=True, exist_ok=True)
# -------------------- Dataset helpers --------------------


def preprocess_digit(image):
    """
    Convert a drawing into an MNIST-style 28x28 image:
    black background, white digit, centered in a 28x28 image.
    """

    # Convert black-on-white drawing to white-on-black.
    grayscale = np.array(image.convert("L"))
    inverted = 255 - grayscale

    # Find the bounding box of the drawn digit.
    mask = inverted > 30
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1

    cropped = Image.fromarray(
        inverted[top:bottom, left:right].astype(np.uint8)
    )

    # Preserve aspect ratio and fit inside a 20x20 region.
    width, height = cropped.size
    scale = DIGIT_SIZE / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Place digit in the center of a 28x28 black image.
    result = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 0)

    x = (OUTPUT_SIZE - new_width) // 2
    y = (OUTPUT_SIZE - new_height) // 2

    result.paste(cropped, (x, y))

    return result



def save_sample(image, label):
    """Save a processed image inside its digit's folder."""

    digit_dir = DATA_DIR / str(label)
    digit_dir.mkdir(parents=True, exist_ok=True)

    # Find the next available sequential filename.
    existing = [
        int(p.stem)
        for p in digit_dir.glob("*.png")
        if p.stem.isdigit()
    ]

    next_id = max(existing, default=0) + 1
    path = digit_dir / f"{next_id:06d}.png"

    image.save(path)
    return path

# -------------------- Drawing application --------------------


class DigitCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Dataset Collector")
        self.root.resizable(False, False)

        self.label = tk.IntVar(value=0)
        self.samples_saved = 0
        self.last_point = None

        # White canvas image; drawing is stored in this PIL image.
        self.image = Image.new(
            "L", (CANVAS_SIZE, CANVAS_SIZE), 255
        )
        self.draw = ImageDraw.Draw(self.image)

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0)

        ttk.Label(
            main,
            text="Draw a digit (0–9)",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(0, 8))

        self.canvas = tk.Canvas(
            main,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="cross",
            highlightthickness=2,
            highlightbackground="#888888"
        )
        self.canvas.grid(row=1, column=0, columnspan=2)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        ttk.Label(main, text="Correct digit label:").grid(
            row=2, column=0, sticky="w", pady=(12, 4)
        )

        digit_frame = ttk.Frame(main)
        digit_frame.grid(row=3, column=0, columnspan=2, pady=4)

        for digit in range(10):
            ttk.Radiobutton(
                digit_frame,
                text=str(digit),
                value=digit,
                variable=self.label
            ).grid(row=0, column=digit, padx=3)

        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=2, pady=12)

        ttk.Button(
            buttons,
            text="Clear",
            command=self.clear
        ).grid(row=0, column=0, padx=4)

        ttk.Button(
            buttons,
            text="Save digit",
            command=self.save
        ).grid(row=0, column=1, padx=4)

        ttk.Button(
            buttons,
            text="Save + next",
            command=self.save_and_clear
        ).grid(row=0, column=2, padx=4)

        ttk.Button(
            buttons,
            text="Preview 28x28",
            command=self.preview
        ).grid(row=0, column=3, padx=4)

        self.status = ttk.Label(
            main,
            text="Draw a digit, select its label, then save."
        )
        self.status.grid(row=5, column=0, columnspan=2)

        ttk.Label(
            main,
            text=f"Dataset folder: {DATA_DIR.resolve()}",
            wraplength=500
        ).grid(row=6, column=0, columnspan=2, pady=(5, 0))

    def start_drawing(self, event):
        self.last_point = (event.x, event.y)

        # A dot should be visible even for a short click.
        r = BRUSH_SIZE // 2
        self.draw.ellipse(
            (event.x - r, event.y - r,
             event.x + r, event.y + r),
            fill=0
        )

        self.canvas.create_oval(
            event.x - r, event.y - r,
            event.x + r, event.y + r,
            fill="black", outline="black"
        )

    def draw_line(self, event):
        if self.last_point is None:
            return

        x1, y1 = self.last_point
        x2, y2 = event.x, event.y

        # Draw onto the image that will actually be saved.
        self.draw.line(
            (x1, y1, x2, y2),
            fill=0,
            width=BRUSH_SIZE
        )

        # Round the line ends.
        r = BRUSH_SIZE // 2
        self.draw.ellipse(
            (x2 - r, y2 - r, x2 + r, y2 + r),
            fill=0
        )

        self.canvas.create_line(
            x1, y1, x2, y2,
            fill="black",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.last_point = (x2, y2)

    def stop_drawing(self, event):
        self.last_point = None

    def clear(self):
        self.image = Image.new(
            "L", (CANVAS_SIZE, CANVAS_SIZE), 255
        )
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        self.status.config(text="Canvas cleared. Draw the next digit.")

    def save(self):
        processed = preprocess_digit(self.image)

        if processed is None:
            messagebox.showwarning(
                "Empty canvas",
                "Draw a digit before saving."
            )
            return

        digit = self.label.get()
        path = save_sample(processed, digit)

        self.samples_saved += 1
        self.status.config(
            text=f"Saved digit {digit} | Session saves: "
                 f"{self.samples_saved} | {path.name}"
        )

    def save_and_clear(self):
        processed = preprocess_digit(self.image)

        if processed is None:
            messagebox.showwarning(
                "Empty canvas",
                "Draw a digit before saving."
            )
            return

        digit = self.label.get()
        path = save_sample(processed, digit)

        self.samples_saved += 1
        self.clear()

        self.status.config(
            text=f"Saved digit {digit}. Draw the next one."
        )

    def preview(self):
        processed = preprocess_digit(self.image)

        if processed is None:
            messagebox.showwarning(
                "Empty canvas", "Draw a digit first."
            )
            return

        window = tk.Toplevel(self.root)
        window.title("Processed 28x28 digit")

        # Enlarge the image for visual inspection.
        enlarged = processed.resize(
            (280, 280), Image.Resampling.NEAREST
        )
        photo = ImageTk.PhotoImage(enlarged)

        label = ttk.Label(window, image=photo)
        label.image = photo
        label.pack(padx=12, pady=12)

        ttk.Label(
            window,
            text=f"Label: {self.label.get()} | Output: 28x28 grayscale"
        ).pack(pady=(0, 12))


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitCollector(root)
    root.mainloop()
