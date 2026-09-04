# Number Guesser

A from-scratch handwritten digit recognition project.

The goal is to build the complete pipeline:

```text
Mouse drawing
     ↓
28 × 28 grayscale image
     ↓
784 input features
     ↓
PyTorch MLP
     ↓
10 digit classes (0–9)
     ↓
Export trained weights
     ↓
C inference + Raylib UI
```

## Project Goals

- Learn PyTorch by building the training pipeline myself.
- Train an MLP on MNIST handwritten digits.
- Understand preprocessing, batching, loss, optimization, and evaluation.
- Export the trained network weights.
- Implement neural-network inference in C without PyTorch.
- Build a Raylib interface for drawing and predicting digits.
- Visualize the network and its activations as part of the final application.

## Structure

```text
Number-Guesser/
├── data/          # Local datasets (not committed)
├── models/        # Generated model weights (not committed)
├── python/        # PyTorch training and export code
├── c/             # C inference and Raylib application
├── tests/         # Tests for math, preprocessing, and inference
├── main.py        # Temporary project entry point
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Status

### Step 0 — Project setup

- [x] Repository structure
- [x] Python/C directories
- [x] Dataset and model directories
- [x] Test directory
- [x] Git ignore rules
- [x] Initial dependency file

### Step 1 — MNIST pipeline

- [ ] Download/load MNIST
- [ ] Build Dataset/DataLoader pipeline
- [ ] Inspect image and label batches
- [ ] Normalize/flatten images

### Step 2 — PyTorch model

- [ ] Implement 784 → 128 → 64 → 10 MLP
- [ ] Implement training loop
- [ ] Evaluate accuracy
- [ ] Save `.pth` weights

### Step 3 — C inference

- [ ] Export weights into a C-friendly format
- [ ] Implement matrix multiplication
- [ ] Implement ReLU
- [ ] Implement forward pass
- [ ] Implement argmax prediction

### Step 4 — Raylib application

- [ ] Drawing canvas
- [ ] Convert drawing to 28 × 28 input
- [ ] Predict button
- [ ] Prediction/confidence display
- [ ] Network visualization

## Philosophy

No black boxes where understanding matters. The project is intentionally built in stages so the mathematics, PyTorch implementation, and final C inference remain understandable.
