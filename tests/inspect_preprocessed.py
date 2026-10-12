from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("preprocessed.bin")
image = np.fromfile(path, dtype=np.float32)

if image.size != 28 * 28:
    raise SystemExit(f"Expected 784 floats, got {image.size}")

if not np.isfinite(image).all():
    raise SystemExit("Input contains NaN or infinity")

image = image.reshape(28, 28)
print(f"shape: {image.shape}")
print(f"min:   {image.min():.6f}")
print(f"max:   {image.max():.6f}")
print(f"mean:  {image.mean():.6f}")
print("ASCII preview (# = bright, + = mid-tone):")

for row in image:
    print("".join(
        "#" if value >= 0.5 else "+" if value >= 0.15 else " "
        for value in row
    ))