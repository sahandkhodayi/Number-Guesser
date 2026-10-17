# Docker and CI

## What Docker is used for here

The desktop Raylib UI needs access to a graphical display and a platform-specific Raylib installation. The initial Docker image therefore targets **headless testing and benchmark execution**, not the GUI.

The image installs Python dependencies and a C toolchain, configures CMake with `BUILD_GUI=OFF`, and builds the C unit-test and parity-benchmark executables. Its default command runs CTest. Model checkpoints, exported weights, datasets, and benchmark outputs are not copied into the image.

## Build and run C unit tests

From the repository root:

```bash
docker build -t number-guesser:dev .
docker run --rm number-guesser:dev
```

The default container command runs the C unit tests. It does not train a model or download MNIST.

## Run the Python/C parity test

The parity test requires a matching PyTorch checkpoint and C-exported weights. Generate them on the host first:

```bash
python python/train.py
python python/export.py
```

Then mount the model and data directories into the container and run the parity script:

Linux / WSL:

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  number-guesser:dev python tests/test_parity.py
```

PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/data:/app/data" `
  number-guesser:dev python tests/test_parity.py
```

The benchmark writes generated intermediate tensors inside the container. The MNIST test data may be downloaded into the mounted data directory. Mounting `models/` provides the checkpoint and exported weights without baking them into the Docker image.

Do not mount the entire `benchmark/` directory over `/app/benchmark`: that would hide the benchmark source files included in the image.

## What the Docker image does not do

- It does not build or run the Raylib GUI.
- It does not automatically train a model.
- It does not include personal digit samples or model artifacts in the image.
- It does not replace testing the desktop application natively on Windows and Linux.

## CI workflow

The GitHub Actions workflow is in `.github/workflows/ci.yml`.

- The C unit-test job runs on `ubuntu-latest` and `windows-latest`.
- Python sources are syntax-compiled on both runners.
- The Ubuntu parity job trains a checkpoint, exports the weights, and runs the numerical comparison against the C implementation.

The workflow must pass on GitHub before this project claims automated CI success. A passing headless job does not prove that the Raylib GUI launches correctly on a developer's machine.
