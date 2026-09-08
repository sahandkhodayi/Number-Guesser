# GPT branch changes

This branch fixes the highest-impact correctness and maintainability problems found in `UI-C`.

## 1. Architecture decision

- PyTorch remains responsible for training.
- C remains responsible for inference/deployment.
- Raylib remains responsible for the UI.
- The CNN architecture and trained weight layout are not changed.

This is intentional: changing the model architecture would invalidate the currently exported weights.

## 2. Accuracy: preprocessing was the first target

The old C UI mapped the whole 280x280 canvas into 28x28 by averaging fixed 10x10 blocks. That means a digit drawn small or near a corner stays small or off-center in the model input.

The new preprocessing finds the non-empty bounding box, makes a square crop, resizes it into a 20x20 region, and centers it inside the 28x28 input.

This should reduce the domain shift between a user's drawing and MNIST. It is still important to test it against real examples; this change is not a claim that the accuracy problem is completely solved.

## 3. Python evaluation bug

`python/evaluate.py` was not actually an evaluation script: it created a fresh model, ran a training step, ran a test step, and saved the model.

It now:

1. loads the already-trained `.pth`,
2. switches to evaluation mode,
3. runs inference without gradients,
4. reports average test loss and accuracy,
5. never changes the weights.

## 4. C header cleanup

Removed duplicate declarations and magic `WEIGHTS_FILE_BYTES` usage from the header. The model's expected binary size is now derived from `sizeof(CnnModel)` in the loader.

## 5. C UI input bug

Button actions previously used mouse-button-held behavior. Holding the mouse button could execute Clear/Predict every frame.

Buttons now react to a mouse press, while canvas drawing still uses mouse-held behavior.

## 6. Code clarity

The C tensor and convolution code keeps the same numerical algorithm, but the confusing learning comments were replaced with comments that explain the actual memory layout and the PyTorch/C correspondence.

## 7. What was deliberately NOT changed

- CNN architecture
- exported tensor ordering
- Conv2D math
- MaxPool2D math
- PyTorch training behavior

Those need numerical verification before being changed. If C and PyTorch disagree on the same MNIST input, the next task is layer-by-layer comparison, not random architecture changes.

## Next verification order

1. Run `python/evaluate.py` and record the PyTorch test accuracy.
2. Export the exact saved model with `python/export.py`.
3. Run C on the same real MNIST images used by Python.
4. Compare logits layer-by-layer.
5. If C matches Python but hand drawings are poor, continue tuning only the UI preprocessing.
6. Only after parity is proven, add activation visualization and polish the UI.
