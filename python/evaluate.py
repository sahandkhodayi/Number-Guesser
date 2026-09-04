"""Evaluation utilities for Number Guesser."""

# Step 0 placeholder.
from torch import nn
import torch    
from model import _MainModel
from train import test_step,train_step
from dataset import get_dataloaders
from helper_functions import accuracy_fn
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "number_guesser_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


Train , Test =get_dataloaders()

model=_MainModel(input_shape=1,
                 hidden_units=32,
                 output_shape=10)

loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

train_step(model,Train,loss_fn,optimizer,accuracy_fn,device="cpu")
test_step(model,Test,loss_fn,accuracy_fn,"cpu")

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)