"""Dataset utilities for the Number Guesser project."""

# Step 0 placeholder.
# MNIST Dataset/DataLoader implementation will be added in the next step.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST( # our train data set
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(   # our test data set
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,          # turning our datasets (images )----> arr or batches of images (64 image per batch) 
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader



