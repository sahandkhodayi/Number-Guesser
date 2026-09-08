"""Export trained PyTorch weights for the C inference implementation."""

# Step 0 placeholder.
# Weight export will be implemented after the PyTorch model is trained.
import torch 
import numpy as np
from pathlib import Path

from model import _MainModel

_modelPath=Path("models/number_guesser_model.pth")

_output=Path("models/weights.bin")

BYTE_VALUE= 175016


def dump(label , tensor):
    flat = tensor.detach().flatten().numpy()
    print(f"{label:8s} shape={tuple(tensor.shape)}  first 5={np.round(flat[:5], 4).tolist()}")



def main():
    
    model=_MainModel(input_shape=1 , hidden_units= 32 , output_shape= 10) # for now we use classes as output then we'll use tokens
    model.load_state_dict(torch.load(_modelPath,map_location="cpu"))

    model.eval()




    if _output.exists():
        raw = np.fromfile(_output, dtype=np.float32)
        x = torch.from_numpy(raw).reshape(1, 1, 28, 28)
    else:
        print(f"[warn] {_output} not found, using a zero image instead")
        x = torch.zeros(1, 1, 28, 28)

    with torch.no_grad():
        dump("input", x)
        a = model.block_1[0](x)
        dump("conv1", a)
        a = model.block_1[1](a)
        dump("relu1", a)
        a = model.block_1[2](a)
        dump("conv2", a)
        a = model.block_1[3](a)
        dump("relu2", a)
        a = model.block_1[4](a) 
        dump("pool1", a)
        a = model.block_2[0](a)
        dump("conv3", a)
        a = model.block_2[1](a) 
        dump("relu3", a)
        a = model.block_2[2](a) 
        dump("conv4", a)
        a = model.block_2[3](a)
        dump("relu4", a)
        a = model.block_2[4](a)
        dump("pool2", a)
        flat = model.classifier[0](a)
        dump("flat", flat)
        logits = model.classifier[1](flat)
        dump("logits", logits)
        print(f"\npredicted digit: {logits.argmax(dim=1).item()}")


if __name__=="__main__":
    main() 