"""MNIST Dataset/DataLoader utilities used by the training pipeline."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64):
    # CHANGE: removed unused plotting import and old placeholder comments.
    # ToTensor() is the exact training preprocessing currently used by the model:
    # uint8 MNIST pixels -> float tensors in [0, 1], shape [1, 28, 28].
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
