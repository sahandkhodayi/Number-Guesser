"""Evaluate the already-trained Number Guesser model on MNIST.

Training belongs in train.py. This file must never train a model as a side effect.
"""
from pathlib import Path
import torch
from torch import nn

from dataset import get_dataloaders
from model import _MainModel

MODEL_PATH = Path("models/number_guesser_model.pth")


def evaluate(model, data_loader, loss_fn, device):
    """Return average loss and accuracy without changing model weights."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    # CHANGE: evaluation now loads the saved model instead of accidentally training one batch.
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Train/save it first.")

    _, test_loader = get_dataloaders(batch_size=64)
    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    loss, accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), "cpu")
    print(f"Test loss: {loss:.5f} | Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
