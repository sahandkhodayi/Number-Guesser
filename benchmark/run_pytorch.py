"""Generate one deterministic real-MNIST input and PyTorch reference tensors.

This benchmark intentionally uses the same trained model and preprocessing as
training. It does not train anything.
"""
from pathlib import Path
import sys

import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from model import _MainModel  # noqa: E402

MODEL_PATH = ROOT / "models" / "number_guesser_model.pth"
SAMPLE_INDEX = 0


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("float32").tofile(path)


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing trained model: {MODEL_PATH}")

    dataset = datasets.MNIST(
        root=ROOT / "data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    image, label = dataset[SAMPLE_INDEX]

    save_tensor(OUT / "input.bin", image)
    (OUT / "label.txt").write_text(str(label), encoding="utf-8")

    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    x = image.unsqueeze(0)
    stages = {}

    with torch.inference_mode():
        x = model.block_1[0](x)
        stages["conv1"] = x
        x = model.block_1[1](x)
        stages["relu1"] = x
        x = model.block_1[2](x)
        stages["conv2"] = x
        x = model.block_1[3](x)
        stages["relu2"] = x
        x = model.block_1[4](x)
        stages["pool1"] = x
        x = model.block_2[0](x)
        stages["conv3"] = x
        x = model.block_2[1](x)
        stages["relu3"] = x
        x = model.block_2[2](x)
        stages["conv4"] = x
        x = model.block_2[3](x)
        stages["relu4"] = x
        x = model.block_2[4](x)
        stages["pool2"] = x
        x = model.classifier(x)
        stages["logits"] = x

    for name, tensor in stages.items():
        save_tensor(OUT / f"pytorch_{name}.bin", tensor)

    prediction = int(stages["logits"].argmax(dim=1).item())
    confidence = float(torch.softmax(stages["logits"], dim=1).max().item())
    print(f"MNIST sample index: {SAMPLE_INDEX}")
    print(f"Label: {label}")
    print(f"PyTorch prediction: {prediction}")
    print(f"Confidence: {confidence:.6f}")
    print("Reference tensors written to benchmark/.")


if __name__ == "__main__":
    main()
