import torch
from torch import nn

#vision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

training_data=datasets.MNIST(
    root="data", # location 
    train=True, # bool -> for training or test
    download=True,
    transform=ToTensor(), # turning or transforing images PLT -> tensor
    target_transform=None # even output can convet to a tensor
)


testing_data=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
        


)
image , lable=training_data[100]
print(lable)