# Number-Guesser — Code-First Project Textbook

This guide continues the existing project. It is not a beginner introduction to C, Python, tensors, CNNs, or PyTorch. The goal is to add real engineering depth to the implementation you already have.

Every chapter follows the same structure:

1. **Goal** — what we are building.
2. **Why** — the problem it solves in this project.
3. **Files** — what to open and change.
4. **Code** — implementation, not just an instruction to “write a test”.
5. **Explanation** — how the code works and why important decisions matter.
6. **Run** — exact commands and expected behavior.
7. **Debugging** — how to diagnose common failures.
8. **Done means** — objective criteria for moving on.

Code blocks marked **complete code** are intended to be usable implementations. If a block is marked **structure only**, it is deliberately not a complete source file.

---

# 0. Where the project stands

The repository already includes:

- PyTorch MNIST training, evaluation, and model export.
- C tensor storage and indexing.
- C Linear, ReLU, Conv2D, MaxPool2D, model loading, and model forward pass.
- A Raylib drawing UI and drawing-to-28×28 preprocessing.
- Python/C intermediate-tensor benchmark scripts.
- The exported model used by C inference.

Do not rebuild those features from scratch. We are improving reliability, testing, evaluation, and then capability.

## The parity milestone is already working

The latest comparison had an exact input match and a maximum difference of approximately 7.63e-6 at the logits. That is excellent numerical agreement for the benchmark input. It proves the two forward passes agree closely on this case; it does not prove memory safety or correctness for every possible input.

## Roadmap

1. Make Python/C parity a one-command regression test.
2. Add C unit tests for individual primitives.
3. Run the tests under memory/undefined-behavior sanitizers.
4. Give the exported model file a validated format.
5. Test preprocessing independently of the CNN.
6. Evaluate a held-out personal-handwriting dataset.
7. Save and classify wrong predictions.
8. Run controlled experiments.
9. Measure and optimize C inference.
10. Refactor application responsibilities when the tests make it safe.
11. Extend single-digit recognition toward multi-digit input.

---

# Chapter 1 — Automate Python/C numerical parity

## Goal

Turn the benchmark you just ran manually into a test that runs both implementations and fails if any intermediate stage differs too much.

## Why

A table that a human reads is useful for debugging, but a later code change can break a layer and go unnoticed. A regression test makes numerical agreement an executable contract.

The current pipeline already exists in:

- benchmark/run_pytorch.py
- benchmark/c_benchmark.c
- benchmark/compare.py

We will add a runner that regenerates the tensors, compiles C, executes the C benchmark, and compares all stages.

## Code — create tests/test_parity.py

~~~python
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
STAGES = [
    "input",
    "conv1", "relu1",
    "conv2", "relu2",
    "pool1",
    "conv3", "relu3",
    "conv4", "relu4",
    "pool2",
    "logits",
]
ATOL = 1e-4


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_floats(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AssertionError(f"Missing benchmark output: {path}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0:
        raise AssertionError(f"Empty benchmark output: {path}")
    if not np.isfinite(values).all():
        raise AssertionError(f"NaN or infinity in {path}")
    return values


def compare_stage(stage: str) -> None:
    expected = load_floats(BENCHMARK / f"pytorch_{stage}.bin")
    actual = load_floats(BENCHMARK / f"c_{stage}.bin")

    if expected.shape != actual.shape:
        raise AssertionError(
            f"{stage}: element-count mismatch: "
            f"Python={expected.shape}, C={actual.shape}"
        )

    diff = np.abs(expected - actual)
    worst = int(np.argmax(diff))
    maximum = float(diff[worst])
    mean = float(diff.mean())

    print(
        f"{stage:8} max={maximum:.8g} "
        f"mean={mean:.8g} worst_index={worst}"
    )

    if maximum > ATOL:
        raise AssertionError(
            f"{stage}: max difference {maximum:.8g} exceeds {ATOL}; "
            f"Python={expected[worst]}, C={actual[worst]}"
        )


def main() -> None:
    # Regenerate reference tensors from the current model checkpoint.
    run([sys.executable, "benchmark/run_pytorch.py"])

    # Build the C benchmark from the current source.
    executable = BENCHMARK / "c_benchmark.exe"
    run([
        "gcc", "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
        "benchmark/c_benchmark.c", "c/src/nn.c",
        "-o", str(executable), "-lm",
    ])

    # C reads benchmark/pytorch_input.bin, the exact Python input.
    run([str(executable)])

    print(f"\nComparing all stages with absolute tolerance {ATOL:g}")
    for stage in STAGES:
        compare_stage(stage)

    print("\nPARITY PASS: every stage is within tolerance.")


if __name__ == "__main__":
    main()
~~~

## Explanation

- ROOT is computed from the script path, so the test does not rely on the terminal's current directory.
- subprocess.run with check=True stops if training files are missing, compilation fails, or the C program exits unsuccessfully.
- The file loader refuses missing, empty, or non-finite tensors. Without these checks, a missing output can be mistaken for a numerical result.
- The comparison checks element counts before comparing values.
- The test prints the maximum difference, mean difference, and worst element index. The worst index gives us a place to start investigating if a layer diverges.
- ATOL is an explicit tolerance. Do not increase it just to hide a bug.

### Platform note

This command assumes GCC is on PATH and creates a Windows executable. In WSL/Linux, name the output c_benchmark and run that executable instead. The compile output path and run path must agree.

## Run

From the repository root in Windows PowerShell:

~~~powershell
python tests/test_parity.py
~~~

## Expected result

Each stage should show a small difference and the final line should be:

~~~text
PARITY PASS: every stage is within tolerance.
~~~

## Debugging

- **Could not open input:** check that Python creates benchmark/pytorch_input.bin and that C reads that exact path.
- **Missing stage file:** make sure the C benchmark writes every stage and check its working directory.
- **Large difference at conv1:** check exported weights and model freshness first.
- **Input difference is nonzero:** stop there; downstream comparisons are not useful until the input is identical.
- **Only a later stage diverges:** debug the first bad stage, not the final logits.

## Done means

- [ ] One command runs both implementations.
- [ ] Missing files and failed commands stop the test.
- [ ] Shape/count mismatch and numerical mismatch stop the test.
- [ ] Current parity passes.
- [ ] Deliberately corrupting a value causes a failure.
- [ ] Reverting the corruption makes it pass again.

---

# Chapter 2 — Build a real C unit-test suite

## Goal

Test individual functions with tiny inputs whose correct answers are known.

## Why

Parity tests tell us whether the full C model agrees with PyTorch for a particular input. Unit tests isolate primitives. If the full network breaks, a small test can tell us whether the problem is tensor indexing, ReLU, Linear, or pooling.

The declarations already exist in c/include/nn.h. We are testing those functions, not rewriting them.

## Code — create tests/test_nn.c

~~~c
#include "../c/include/nn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_tensor_set_get(void) {
    Tensor t = tensor_alloc(2, 3, 4);
    assert(t.data != NULL);

    tensor_set(&t, 1, 2, 3, 42.5f);
    assert(fabsf(tensor_get(&t, 1, 2, 3) - 42.5f) < 1e-6f);

    /* A separate channel must not overwrite this element. */
    assert(tensor_get(&t, 0, 2, 3) == 0.0f);

    tensor_free(&t);
    assert(t.data == NULL);
    assert(t.channels == 0);
    assert(t.height == 0);
    assert(t.width == 0);
}

static void test_relu(void) {
    float values[] = {-3.0f, 0.0f, 2.5f, -0.25f};
    relu(values, 4);

    assert(values[0] == 0.0f);
    assert(values[1] == 0.0f);
    assert(values[2] == 2.5f);
    assert(values[3] == 0.0f);
}

static void test_argmax(void) {
    float values[] = {-2.0f, 0.5f, 9.0f, 3.0f};
    assert(argmax(values, 4) == 2);
}

static void test_linear(void) {
    /*
       W = [1 2]    b = [10]    x = [5]
           [3 4]        [20]        [6]

       y0 = 1*5 + 2*6 + 10 = 27
       y1 = 3*5 + 4*6 + 20 = 59
    */
    const float W[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {10.0f, 20.0f};
    const float x[] = {5.0f, 6.0f};
    float y[2] = {0.0f, 0.0f};

    linear(W, b, x, y, 2, 2);

    assert(fabsf(y[0] - 27.0f) < 1e-6f);
    assert(fabsf(y[1] - 59.0f) < 1e-6f);
}

static void test_maxpool(void) {
    Tensor input = tensor_alloc(1, 4, 4);
    const float values[16] = {
        1, 8, 2, 4,
        3, 5, 7, 6,
        9, 0, 1, 2,
        4, 3, 8, 5
    };

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            tensor_set(&input, 0, y, x, values[y * 4 + x]);
        }
    }

    Tensor output = maxpool2d(&input, 2, 2);
    assert(output.data != NULL);
    assert(output.channels == 1);
    assert(output.height == 2);
    assert(output.width == 2);

    assert(tensor_get(&output, 0, 0, 0) == 8.0f);
    assert(tensor_get(&output, 0, 0, 1) == 7.0f);
    assert(tensor_get(&output, 0, 1, 0) == 9.0f);
    assert(tensor_get(&output, 0, 1, 1) == 8.0f);

    tensor_free(&output);
    tensor_free(&input);
}

int main(void) {
    test_tensor_set_get();
    test_relu();
    test_argmax();
    test_linear();
    test_maxpool();

    puts("All C unit tests passed.");
    return 0;
}
~~~

## Explanation

### Tensor indexing

The test writes one location and reads it back. It also checks a corresponding location in another channel. This catches indexing errors that can be hidden by a final classification result.

### ReLU

The test covers negative, zero, and positive values. These are the three important behavioral cases for this function.

### Linear

The expected answers are calculated by hand. This is important: the test should not call the same implementation or depend on PyTorch to calculate the expected result.

### MaxPool

Each output is the maximum of one non-overlapping 2×2 window. The expected output is:

~~~text
8 7
9 8
~~~

We check the dimensions as well as the values. A function that returns correct values in an incorrectly shaped tensor is still broken.

### Conv2D comes next

Conv2D needs a hand-calculated test after confirming the implementation's exact weight layout, padding, and output-shape behavior. Do not paste a guessed kernel test: the expected answer must match the real function contract. Chapter 1 already gives us a full-network comparison, but a tiny hand-calculated Conv2D test will make future debugging more local.

## Run

~~~powershell
gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_nn.c c/src/nn.c -o tests/test_nn.exe -lm
.\tests\test_nn.exe
~~~

Expected output:

~~~text
All C unit tests passed.
~~~

## Debugging

- **Undefined reference:** compile tests/test_nn.c together with c/src/nn.c.
- **Header not found:** run from the repository root and check the relative include.
- **A test fails:** debug that primitive with the tiny values before changing the CNN.
- **Allocation fails:** inspect tensor_alloc and its initialization/cleanup behavior.

## Done means

- [ ] The C test executable compiles with warnings enabled.
- [ ] Tensor, ReLU, argmax, Linear, and MaxPool tests pass.
- [ ] Changing an expected value makes a test fail.
- [ ] A hand-calculated Conv2D test is added after its layout contract is confirmed.

---

# Chapter 3 — Detect memory errors with sanitizers

## Goal

Run the C tests with tools that detect common memory violations and undefined behavior.

## Why

A C function can return the correct answer once and still access memory outside an allocation, use a freed pointer, or rely on undefined behavior. Numerical parity does not prove memory safety.

## Code — build a separate sanitizer executable

On a GCC environment with AddressSanitizer support:

~~~bash
gcc -std=c11 -Wall -Wextra -Wpedantic -g -O1 \
    -fsanitize=address,undefined \
    tests/test_nn.c c/src/nn.c \
    -o tests/test_nn_sanitized -lm
~~~

Run:

~~~bash
./tests/test_nn_sanitized
~~~

Keep this as a separate executable so the ordinary build remains available.

## Explanation

- **-g** includes debug symbols for source locations.
- **-O1** provides modest optimization while retaining useful diagnostics.
- **-fsanitize=address,undefined** instruments the program to detect supported classes of invalid memory use and undefined behavior.
- A clean run only means the paths exercised by these tests did not trigger a report. It is not a proof that every possible input is safe.

On Windows, sanitizer support depends on the compiler distribution. If your MinGW GCC does not support AddressSanitizer, use WSL GCC or a supported Clang build. A compiler-flag error is an environment limitation, not evidence that your program has a memory bug.

## Debugging

When a sanitizer reports an error, identify the first relevant source location, determine which allocation/index is invalid, fix the cause, and rerun all tests. Do not suppress the report simply to get a clean run.

## Done means

- [ ] Normal C tests pass.
- [ ] A sanitizer build works in a supported environment.
- [ ] All tests pass under the sanitizer.
- [ ] Any reported source location is understood and fixed.

---

# Chapter 4 — Give the model file a real format

## Goal

Make the exported model identify itself and let C reject incompatible or corrupted files.

## Why

Python creates the weights and C consumes them. The file is an interface between two programs. A raw sequence of floats does not describe which architecture it belongs to or which format version it uses. If the model architecture changes, the loader must not silently read the wrong data.

## Code — define the format first

Create models/FORMAT.md:

~~~markdown
# Number-Guesser model format

## Version 1

- Magic: four ASCII bytes NGNN
- Version: unsigned 32-bit integer, little-endian
- Architecture ID: unsigned 32-bit integer
- Payload length: unsigned 64-bit integer, little-endian
- Payload: IEEE-754 float32 values, little-endian

Payload order:
1. conv1 weights
2. conv1 bias
3. conv2 weights
4. conv2 bias
5. conv3 weights
6. conv3 bias
7. conv4 weights
8. conv4 bias
9. fully connected weights
10. fully connected bias

The loader rejects an unknown magic, unsupported version,
unknown architecture ID, wrong payload length, or truncated file.
~~~

## Explanation

The specification comes first so the Python exporter and C loader implement one agreement. Versioning allows the format to evolve safely. Payload length detects truncated or unexpected files. The architecture ID prevents a file for a different layer shape from being accepted accidentally.

## Implementation order

1. Update python/export.py to write the header followed by the payload.
2. Update model_load in c/src/nn.c to validate the header before reading weights.
3. Reject incorrect payload length and truncated files.
4. Add tests for a valid file, wrong magic, unsupported version, and truncated payload.
5. Export a fresh model and rerun Chapter 1.

Do not update the exporter and loader separately and leave them incompatible. Treat the format change as one feature spanning Python, C, and tests.

## Done means

- [ ] The format is documented.
- [ ] Python writes the documented fields and payload order.
- [ ] C validates the header and exact payload size.
- [ ] Invalid files fail with a useful error.
- [ ] A newly exported model passes parity.

---

# Chapter 5 — Test preprocessing independently of the CNN

## Goal

Inspect the final 28×28 input that the application actually sends into the model.

## Why

The model was trained on MNIST, not arbitrary mouse drawings. Cropping, centering, scaling, interpolation, and intensity conventions can cause bad predictions even when C inference matches PyTorch perfectly. When parity passes but the drawing UI predicts poorly, inspect preprocessing before changing the network.

The existing preprocessing is in c/src/ui.c and includes canvas drawing, bounding-box extraction, crop/margin handling, resizing, and conversion to a 28×28 input.

## Code — create tests/inspect_preprocessed.py

This checker expects a debug dump containing exactly 784 float32 values.

~~~python
from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("preprocessed.bin")
image = np.fromfile(path, dtype=np.float32)

if image.size != 28 * 28:
    raise SystemExit(f"Expected 784 floats, got {image.size}")

if not np.isfinite(image).all():
    raise SystemExit("Input contains NaN or infinity")

image = image.reshape(28, 28)
print(f"shape: {image.shape}")
print(f"min:   {image.min():.6f}")
print(f"max:   {image.max():.6f}")
print(f"mean:  {image.mean():.6f}")
print("ASCII preview (# = bright, + = mid-tone):")

for row in image:
    print("".join(
        "#" if value >= 0.5 else "+" if value >= 0.15 else " "
        for value in row
    ))
~~~

## Explanation

The checker does not change the image. It validates the value count, rejects NaN/Inf, prints the intensity range, and gives a rough terminal preview. This helps reveal an empty image, inverted colors, a clipped digit, or a digit placed in the wrong part of the tensor.

The next change in C is to add an optional debug dump immediately after canvas_to_mnist_input has produced the final 784 floats. Do not write a file every frame; make it an explicit debug action.

## Run

~~~powershell
python tests/inspect_preprocessed.py path\to\preprocessed.bin
~~~

## Required fixtures

Create deterministic inputs for:

- Empty canvas — expected output is 784 zeros.
- Centered digit.
- Digit touching the left or top edge.
- Very wide digit.
- Very tall digit.
- Digit near a corner.
- Thick and thin strokes.

These should be known pixel inputs, not screenshots affected by window scaling.

## Done means

- [ ] The final tensor can be saved on demand.
- [ ] Empty canvas produces zeros.
- [ ] Edge cases stay within bounds.
- [ ] Aspect ratio is preserved.
- [ ] Pixel intensity range and foreground/background match training.
- [ ] The output can be inspected without running the CNN.

---

# Chapter 6 — Evaluate on personal handwriting

## Goal

Create a labelled evaluation set that is separate from training data.

## Why

MNIST test accuracy tells you how the model performs on MNIST test images. It does not tell you how it handles your own stroke thickness, slant, spacing, and drawing habits. A held-out dataset gives you evidence about the application you are building.

## Dataset structure

~~~text
my_digits/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
~~~

Start with 20 labelled images per digit if practical. Keep these images out of training while using them to measure generalization.

## Code — create python/evaluate_folder.py

This starter evaluator expects image files containing one digit on an MNIST-like background. It does not automatically reproduce the Raylib crop/centering pipeline; for a fair evaluation of the C app, later feed the exact preprocessed 28×28 dumps into the evaluation pipeline too.

~~~python
from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import _MainModel

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "number_guesser_model.pth"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    total = 0
    correct = 0
    per_digit = {d: {"total": 0, "correct": 0} for d in range(10)}
    wrong_rows = []

    with torch.inference_mode():
        for label in range(10):
            folder = args.dataset / str(label)
            if not folder.is_dir():
                raise SystemExit(f"Missing class directory: {folder}")

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue

                with Image.open(path) as source:
                    image = transform(source.convert("RGB")).unsqueeze(0)

                prediction = int(model(image).argmax(dim=1).item())
                total += 1
                per_digit[label]["total"] += 1

                if prediction == label:
                    correct += 1
                    per_digit[label]["correct"] += 1
                else:
                    wrong_rows.append((str(path), label, prediction))
                    print(f"wrong: true={label}, predicted={prediction}, file={path}")

    if total == 0:
        raise SystemExit("No images found")

    print(f"overall accuracy: {correct}/{total} = {correct / total:.2%}")
    for label, counts in per_digit.items():
        n = counts["total"]
        if n:
            score = counts["correct"] / n
            print(f"{label}: {counts['correct']}/{n} = {score:.2%}")

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "wrong_predictions.csv"
    with report.open("w", encoding="utf-8", newline="") as file:
        file.write("path,true_label,prediction\n")
        for path, true_label, prediction in wrong_rows:
            file.write(f'"{path}",{true_label},{prediction}\n')
    print(f"Wrong-prediction report: {report}")


if __name__ == "__main__":
    main()
~~~

## Explanation

- Folder names are the ground-truth labels.
- eval mode and inference_mode disable training behavior and gradient tracking.
- Overall accuracy is reported alongside per-digit accuracy, so a weak class is not hidden by the average.
- Wrong predictions are recorded with paths so the actual examples can be inspected.
- The evaluation set is not used to update weights. If you train on it, it is no longer a clean held-out evaluation set.

## Run

~~~powershell
python python/evaluate_folder.py my_digits
~~~

## Done means

- [ ] Every image has a correct label.
- [ ] Evaluation does not train or update weights.
- [ ] Overall and per-digit accuracy are printed.
- [ ] Wrong examples are recorded.
- [ ] The dataset remains separate from training data.

---

# Chapter 7 — Turn failures into experiments

## Goal

Classify wrong predictions before changing the model.

## Why

Randomly adding layers is not a debugging strategy. A failure can be caused by preprocessing, ambiguous handwriting, a wrong label, or model confusion. Each cause requires a different fix.

## Code — maintain an experiment log

Create experiments/README.md:

~~~markdown
# Experiment log

For every experiment record:

- ID and date
- question being tested
- baseline commit/checkpoint
- exactly one primary change
- dataset and split
- overall and per-digit accuracy
- parity result
- runtime if performance is involved
- conclusion and next action

Never overwrite baseline results.
~~~

## Example experiment

Question: does a slightly larger margin around a digit improve personal-handwriting accuracy?

1. Save the current results as the baseline.
2. Change only the margin.
3. Use the same checkpoint and same evaluation images.
4. Record overall and per-digit accuracy.
5. Inspect examples that improved and regressed.
6. Keep the change only if the evidence supports it.

Use the same method for target size, centering, interpolation, or model changes. Do not change five things at once; then you would not know which change mattered.

## Done means

- [ ] Each experiment starts with a question.
- [ ] Baseline and changed results are preserved.
- [ ] Only one primary factor changes at a time.
- [ ] Parity is rerun after inference changes.
- [ ] Conclusions are based on recorded results.

---

# Chapter 8 — Measure before optimizing C

## Goal

Measure inference time and identify the bottleneck before changing the implementation.

## Why

The code that looks slow is not necessarily the code that dominates runtime. A baseline prevents wasted work and lets you quantify whether an optimization helped.

## Timing-loop structure

The following is **structure only**, not a complete file: the timer function must be implemented with a monotonic clock supported by your target platform.

~~~c
const int warmup_runs = 20;
const int measured_runs = 500;

for (int i = 0; i < warmup_runs; ++i) {
    model_forward(&model, &input, logits);
}

double start = monotonic_time_seconds();

for (int i = 0; i < measured_runs; ++i) {
    model_forward(&model, &input, logits);
}

double elapsed = monotonic_time_seconds() - start;
printf("mean inference: %.6f ms\n",
       elapsed * 1000.0 / measured_runs);
~~~

## Explanation

Warm-up runs reduce first-use effects. Repeated runs reduce the influence of one noisy measurement. Timing only model_forward isolates inference from model loading and UI rendering. Measure preprocessing separately if end-to-end latency is the concern.

## Optimization sequence

1. Record baseline timing.
2. Identify the measured bottleneck.
3. Change one implementation detail.
4. Rerun unit tests, sanitizer tests, and parity.
5. Measure again.
6. Keep the change only if it improves the measured result without breaking correctness.

Possible future work includes reducing repeated index calculations, improving memory locality, reusing temporary buffers, and changing convolution loop order. Do not begin with these guesses before collecting a baseline.

## Done means

- [ ] A reproducible benchmark exists.
- [ ] Baseline timing is recorded.
- [ ] A bottleneck is identified from measurement.
- [ ] Before/after results exist for one optimization.
- [ ] Tests, sanitizers, and parity still pass.

---

# Chapter 9 — Refactor only when tests make it safe

## Goal

Separate inference, preprocessing, and application/UI control when the current code is difficult to test or change.

## Why

UI code should not be responsible for tensor arithmetic or model-file parsing. Separation lets us test inference without launching Raylib. However, a large refactor before tests exist is risky, so do this after the earlier chapters.

## Possible target structure

~~~text
c/
├── include/
│   ├── nn.h
│   ├── ui.h
│   ├── preprocessing.h
│   └── app.h
└── src/
    ├── nn.c
    ├── ui.c
    ├── preprocessing.c
    ├── app.c
    └── main.c
~~~

## Safe refactoring procedure

1. Choose one responsibility that can be separated without changing behavior.
2. Move its declaration to the correct header and implementation to its source file.
3. Update the build configuration.
4. Compile with warnings enabled.
5. Run unit tests and parity.
6. Commit the working change before moving another responsibility.

Do not create empty abstraction layers just to match the diagram. Split a module when the current code is hard to test or when different responsibilities need to change independently.

## Done means

- [ ] Inference can be tested without launching Raylib.
- [ ] Preprocessing has deterministic tests.
- [ ] UI code does not own tensor math or model serialization.
- [ ] Existing behavior and parity remain unchanged.

---

# Chapter 10 — Move from one digit to multiple digits

## Goal

Extend from “which digit is in this image?” to “where are the digits and what sequence do they form?”

## Why

A classifier recognizes one digit. Multi-digit recognition additionally needs segmentation: identifying where each digit begins and ends.

## First architecture

~~~text
input image
    ↓
foreground mask
    ↓
connected components
    ↓
filter tiny noise components
    ↓
sort boxes left-to-right
    ↓
crop each digit
    ↓
reuse existing 28×28 preprocessing
    ↓
run the existing C CNN per crop
    ↓
concatenate predictions
~~~

## Implementation order

1. Add connected-component extraction for a binary image.
2. Unit-test it with synthetic images containing known rectangles.
3. Sort components from left to right.
4. Crop each component and reuse the documented preprocessing.
5. Call the existing inference function once per crop.
6. Display the resulting digit string.
7. Track segmentation errors separately from classifier errors.

Start with clean, separated, horizontally written digits. Do not begin with overlapping cursive writing or a transformer OCR model.

## Done means

- [ ] Synthetic component tests pass.
- [ ] Components are sorted in reading order.
- [ ] Each crop uses the same documented preprocessing.
- [ ] Segmentation errors are separated from classification errors.
- [ ] A fixed evaluation set measures progress.

---

# What not to do now

Do not rewrite Conv2D or tensor storage without evidence of a bug. Do not add random layers because a few drawings are misclassified. Do not train on your held-out evaluation set and still call it held out. Do not optimize before measuring. Do not start C backpropagation before the inference pipeline is robust.

# Your next action

Start with Chapter 1 and make the successful manual parity workflow repeatable with one command. Then implement the C unit tests in Chapter 2. Your current parity numbers are already excellent; the next progress should come from automation and test coverage, not from trying to make the floating-point differences even smaller.

# Source-of-truth rule

The repository code is authoritative. If a function signature or architecture changes, update this guide to match it. Compile every new C example against the current headers. Never assume an example is compatible just because it looks plausible.
