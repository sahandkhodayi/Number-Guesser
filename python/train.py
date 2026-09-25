"""Training entry point and one-epoch helpers for Number Guesser."""

from pathlib import Path

import torch
from torch import nn

from dataset import get_dataloaders
from helper_functions import accuracy_fn
from model import _MainModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "number_guesser_model.pth"


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device,
):
    """Train the model for one epoch."""
    model.to(device)
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=logits.argmax(dim=1))

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device,
):
    """Evaluate the model for one epoch without updating weights."""
    model.to(device)
    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=logits.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc


def main():
    epochs = 5
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model, train_loader, loss_fn, optimizer, accuracy_fn, device
        )
        test_loss, test_acc = test_step(
            model, test_loader, loss_fn, accuracy_fn, device
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_loss:.5f} | train acc={train_acc:.2f}% | "
            f"test loss={test_loss:.5f} | test acc={test_acc:.2f}%"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
