from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import _MainModel

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "number_guesser_model.pth"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    total = 0
    correct = 0
    per_digit = {d: {"total": 0, "correct": 0} for d in range(10)}
    wrong_rows = []

    with torch.inference_mode():
        for label in range(10):
            folder = args.dataset / str(label)
            if not folder.is_dir():
                raise SystemExit(f"Missing class directory: {folder}")

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue

                with Image.open(path) as source:
                    image = transform(source.convert("RGB")).unsqueeze(0)

                prediction = int(model(image).argmax(dim=1).item())
                total += 1
                per_digit[label]["total"] += 1

                if prediction == label:
                    correct += 1
                    per_digit[label]["correct"] += 1
                else:
                    wrong_rows.append((str(path), label, prediction))
                    print(f"wrong: true={label}, predicted={prediction}, file={path}")

    if total == 0:
        raise SystemExit("No images found")

    print(f"overall accuracy: {correct}/{total} = {correct / total:.2%}")
    for label, counts in per_digit.items():
        n = counts["total"]
        if n:
            score = counts["correct"] / n
            print(f"{label}: {counts['correct']}/{n} = {score:.2%}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "wrong_predictions.csv"
    with report.open("w", encoding="utf-8", newline="") as file:
        file.write("path,true_label,prediction\n")
        for path, true_label, prediction in wrong_rows:
            file.write(f'"{path}",{true_label},{prediction}\n')
    print(f"Wrong-prediction report: {report}")


if __name__ == "__main__":
    main()