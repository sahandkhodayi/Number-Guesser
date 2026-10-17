# Model, preprocessing, and data contract

## Model definition

The PyTorch model in `python/model.py` currently accepts one-channel 28×28 images and returns ten logits, one per digit class.

The architecture has four 3×3 convolution layers with ReLU activations, two 2×2 max-pooling layers, a flatten operation, and a linear classifier. With 32 channels, the spatial dimensions progress from 28×28 to 14×14 to 7×7. The classifier therefore receives 32 × 7 × 7 = 1568 features.

The C implementation must preserve the same operations, dimensions, parameter order, and tensor layout.

## Input preprocessing

The MNIST loader currently uses `torchvision.transforms.ToTensor()`, converting grayscale pixel values to floating-point values in [0, 1] and producing channel-first tensors.

The drawing UI uses its own preprocessing code to convert canvas pixels into a 28×28 input. These two paths are not automatically identical just because they have the same output dimensions. Changes to centering, interpolation, background, stroke thickness, or normalization can change prediction quality.

When changing preprocessing:

1. Document the old and new behavior.
2. Save representative input examples.
3. Compare Python and C on exactly the same pixel values.
4. Evaluate with a fixed validation set and personal handwriting samples.
5. Record any accuracy or distribution changes.

## Checkpoint and export format

The training script saves a PyTorch checkpoint at:

`models/number_guesser_model.pth`

The exporter writes a raw binary file at:

`models/weights.bin`

The exporter uses an explicit `LAYER_KEYS` order. The C model loader must read tensors in the exact same order. A raw float file does not inherently include architecture, shape, endianness, data type, version, or checksum metadata.

The current exporter has an expected byte count check, but currently prints a warning rather than failing if the size differs. This should be changed to a hard failure, along with checks for tensor shapes, dtype, finite values, and exact byte count.

## Artifact lifecycle

1. Train: create or replace the PyTorch checkpoint.
2. Evaluate: load the checkpoint and measure it without training.
3. Export: convert the checkpoint to the C binary format.
4. Run: load the binary into the C inference engine.

Re-export after every checkpoint change. Otherwise, the Python evaluation and C application may be using different model versions.

The checkpoint and binary weights are ignored by Git, so a fresh clone must generate or obtain them separately. Avoid committing large generated artifacts by accident. For releases, document the model's source, architecture version, and compatibility.

## Recommended future binary format

A versioned file format would be safer than an unannotated flat array. A minimal header could include:

- Magic bytes identifying the file as a Number Guesser model.
- Format version.
- Architecture/version identifier.
- Tensor count and tensor metadata.
- Data type and expected byte order.
- Payload length.
- Optional checksum.

The C loader should reject unknown versions, incorrect lengths, invalid tensor dimensions, and truncated files before inference begins. Do not implement a new format until the existing export/load contract is covered by tests; then change the exporter and loader together.

## Accuracy and confidence

The application uses logits to select the most likely class and computes softmax values for the probability bars. Softmax produces normalized scores, but these scores should not be described as calibrated confidence unless calibration has been measured.

Report test-set accuracy separately from handwritten-drawing performance. MNIST performance alone does not establish robustness to real user input.
