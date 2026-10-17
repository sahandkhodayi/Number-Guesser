# Testing and verification

## Current test coverage

### C unit tests: `tests/test_nn.c`

The current suite exercises:

- Tensor allocation, set/get indexing, and freeing.
- ReLU behavior for negative, zero, and positive values.
- Argmax on a known vector.
- Linear layer output for a small hand-computed example.
- Max pooling on a known 4×4 input.

These are useful deterministic unit tests. They do not currently cover every operation or all invalid-input cases.

### Python/C numerical parity: `tests/test_parity.py`

The parity driver:

1. Runs the PyTorch reference benchmark.
2. Compiles the C benchmark.
3. Runs C inference on the matching generated input.
4. Compares intermediate outputs for input, convolution/ReLU stages, pooling stages, and final logits.
5. Fails if a stage's maximum absolute difference exceeds the configured tolerance (`ATOL = 1e-4`).

The test validates the generated reference and C outputs for the benchmark input and configured stages. It does not test generalization, memory safety, malformed model files, or all possible inputs.

## Running the current tests

With Python dependencies and GCC available:

```bash
python tests/test_parity.py
```

The C unit test can be built manually on Linux/WSL:

```bash
mkdir -p build
gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_nn.c c/src/nn.c -o build/test_nn -lm
./build/test_nn
```

The current parity driver and Makefile contain platform-specific assumptions. Native Windows support is a planned improvement, not yet verified by this documentation.

## Test layers to add next

Implement these in small increments; keep each test deterministic and easy to run.

### 1. More C unit tests

- Convolution with a tiny hand-calculated input, kernel, and bias.
- Convolution padding and stride behavior.
- Multiple input and output channels.
- Model load success with a valid fixture and failure with a missing/truncated file.
- Invalid dimensions and zero-length inputs, once the public API's expected behavior is defined.
- Softmax behavior: finite outputs, sum approximately one, and large logits.
- Input preprocessing: blank canvas, boundary strokes, and expected output dimensions/range.

### 2. Python tests

- Check the model's output shape for a small input batch.
- Check evaluation does not modify model parameters.
- Check export fails when the checkpoint is missing or incompatible.
- Check exported byte count and parameter order; a mismatch should be a hard error.
- Check dataset and preprocessing assumptions without downloading MNIST where possible.

### 3. End-to-end tests

- Use a small fixed input and known test weights.
- Verify the C model loader rejects invalid artifacts.
- Compare Python and C outputs for several fixed inputs, not just one.
- Check predicted class agreement in addition to per-stage numerical tolerances.
- Test the checkpoint → export → C load sequence from a clean workspace.

### 4. Memory and undefined-behavior checks

On a compatible GCC/Clang environment, build tests with AddressSanitizer and UndefinedBehaviorSanitizer. Keep sanitizer jobs separate from ordinary tests where the platform does not support them. Sanitizers can detect certain classes of bugs but do not prove the absence of all memory errors.

## CI acceptance criteria

A first GitHub Actions workflow should:

- Run on pull requests and pushes to the main branch.
- Set up a supported Python version and install compatible dependencies.
- Run Python formatting/linting and unit tests.
- Configure and build the C targets.
- Run the C unit tests.
- Run parity only when its model artifacts and required packages are generated or installed reproducibly.
- Report failures without silently skipping tests.

The CI build should not depend on model binaries that are ignored by Git or on data already cached on a developer's machine.
