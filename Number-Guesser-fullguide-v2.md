# Number Guesser — Strategic ML/C Engineering Roadmap

> **This is the guide to use from the current state of the repository.**
>
> The project already has a working PyTorch CNN, a C inference implementation, a Raylib drawing application, model export, and a Python/C intermediate-tensor benchmark.
>
> We are NOT going back to beginner Python/C/CNN lessons.
>
> The purpose of this guide is to take the existing project from **"it works"** to:
>
> **verified → testable → debuggable → reproducible → measurable → optimized → capable of real-world handwriting → extensible toward OCR.**

---

# 0. WHERE YOU ARE RIGHT NOW

Repository:

`github.com/sahandkhodayi/Number-Guesser`

Current model:

```text
1 × 28 × 28
    ↓
Conv 1 → 32, 3×3, padding 1
    ↓
ReLU
    ↓
Conv 32 → 32, 3×3, padding 1
    ↓
ReLU
    ↓
MaxPool 2×2
    ↓
Conv 32 → 32, 3×3, padding 1
    ↓
ReLU
    ↓
Conv 32 → 32, 3×3, padding 1
    ↓
ReLU
    ↓
MaxPool 2×2
    ↓
Flatten 1568
    ↓
Linear 1568 → 10
```

The repository already contains:

```text
python/
    model.py
    train.py
    evaluate.py
    export.py
    dataset.py
    ...

c/
    include/nn.h
    src/nn.c
    include/ui.h
    src/ui.c
    src/main.c

benchmark/
    run_pytorch.py
    c_benchmark.c
    compare.py

tests/
    (currently no real test suite)

models/
    number_guesser_model.pth
    weights.bin

fullguide.md
foundations-reference.md
CMakeLists.txt
```

The current C implementation already contains:

```text
Tensor
tensor_alloc
tensor_free
tensor_get
tensor_set
Linear
ReLU
argmax
Conv2D
MaxPool2D
model_load
model_forward
```

The Raylib application already contains:

```text
drawing
brush
canvas clearing
bounding-box extraction
square crop
margin
bilinear resize
28×28 conversion
prediction
softmax probabilities
confidence display
```

Therefore:

## DO NOT REBUILD

Do not restart with:

- "What is a tensor?"
- "What is a pointer?"
- "What is ReLU?"
- "How does Conv2D work?"
- "How do I write a CNN?"
- "How do I create a Python class?"
- "How does PyTorch work?"

Those belong to `foundations-reference.md`.

---

# 1. THE NEW STANDARD FOR THIS PROJECT

Previously:

```text
write code
↓
compile
↓
it runs
↓
DONE
```

From now on:

```text
design
↓
implement
↓
run
↓
measure
↓
test
↓
break intentionally
↓
debug
↓
verify again
↓
document
↓
commit
```

A feature is not considered complete merely because the application works.

For this project:

```text
DONE =
implemented
+
tested
+
failure behavior understood
+
measured when appropriate
+
documented
```

This is the main transition from a learning project into a serious engineering project.

---

# 2. THE ROADMAP

Follow these stages in order.

```text
                    CURRENT PROJECT
                          │
                          ▼
              ┌─────────────────────┐
              │ 1. Numerical Parity │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 2. C Unit Testing   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 3. Sanitizers       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 4. Model Format     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 5. Preprocessing    │
              │    Test System      │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 6. Real Handwriting │
              │    Evaluation       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 7. Error Analysis   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 8. Experiments      │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ 9. C Profiling      │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │10. Architecture     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │11. Multi-digit OCR  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │12. Advanced ML      │
              └─────────────────────┘
```

Do not jump to OCR because it sounds more impressive.

The engineering depth of the current single-digit system is the actual lesson.

---

# 3. STAGE 1 — AUTOMATIC PYTHON ↔ C NUMERICAL PARITY

## Goal

Prove that:

```text
PyTorch
```

and

```text
C
```

perform the same computation.

Not "they usually predict the same digit."

The actual requirement is:

```text
same input
    ↓
same weights
    ↓
same operation
    ↓
same intermediate tensors
    ↓
same output within floating-point tolerance
```

---

## 3.1 What already exists

You already have:

```text
benchmark/run_pytorch.py
benchmark/c_benchmark.c
benchmark/compare.py
```

`run_pytorch.py` generates:

```text
input.bin

pytorch_conv1.bin
pytorch_relu1.bin
pytorch_conv2.bin
pytorch_relu2.bin
pytorch_pool1.bin
pytorch_conv3.bin
pytorch_relu3.bin
pytorch_conv4.bin
pytorch_relu4.bin
pytorch_pool2.bin
pytorch_logits.bin
```

The C benchmark produces matching:

```text
c_conv1.bin
c_relu1.bin
...
c_logits.bin
```

This is excellent.

But the current system is still a **benchmark/debugging script**, not yet a real automated test.

That is what we build now.

---

# 4. FIRST RUN — DO NOT CHANGE CODE

From the repository root:

```bash
python benchmark/run_pytorch.py
```

Then compile the benchmark.

If using GCC:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    benchmark/c_benchmark.c \
    c/src/nn.c \
    -o benchmark/c_benchmark.exe \
    -lm
```

Run:

```bash
./benchmark/c_benchmark.exe
```

On Windows PowerShell:

```powershell
.\benchmark\c_benchmark.exe
```

Then:

```bash
python benchmark/compare.py
```

You want something approximately like:

```text
Stage                  max_abs_diff       mean_abs_diff
-------------------------------------------------------
input                  0                  0
conv1                  very small         very small
relu1                  very small         very small
conv2                  very small         very small
...
logits                 very small         very small

All available stages are within 1e-4...
```

The exact floating-point values may differ slightly.

---

# 5. WHY THE FIRST DIVERGENCE MATTERS

Suppose:

```text
conv1      0.000001
relu1      0.000001
conv2      0.000002
relu2      0.000002
pool1      0.000002
conv3      0.72
relu3      0.72
conv4      0.80
...
```

Do NOT debug `conv4`.

The first meaningful failure is:

```text
conv3
```

Therefore investigate:

```text
conv3 weights
conv3 bias
conv3 input
Conv2D indexing
padding
stride
tensor layout
```

This is one of the most important debugging skills in the entire project:

> **Find the first divergence, not the final symptom.**

---

# 6. UPGRADE THE COMPARISON TOOL

The current `compare.py` only reports:

```text
max difference
mean difference
```

Upgrade it so that a failure tells you exactly where it happened.

Create:

```text
tests/test_parity.py
```

Use:

```python
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"

STAGES = [
    "input",
    "conv1",
    "relu1",
    "conv2",
    "relu2",
    "pool1",
    "conv3",
    "relu3",
    "conv4",
    "relu4",
    "pool2",
    "logits",
]

ATOL = 1e-4


def load_tensor(path: Path) -> np.ndarray:
    if not path.exists():
        raise AssertionError(f"Missing tensor file: {path}")

    return np.fromfile(path, dtype=np.float32)


def compare_stage(stage: str) -> None:
    py_path = BENCHMARK / f"pytorch_{stage}.bin"
    c_path = BENCHMARK / f"c_{stage}.bin"

    expected = load_tensor(py_path)
    actual = load_tensor(c_path)

    if expected.shape != actual.shape:
        raise AssertionError(
            f"{stage}: shape mismatch: "
            f"{expected.shape} != {actual.shape}"
        )

    diff = np.abs(expected - actual)

    max_index = int(np.argmax(diff))
    max_diff = float(diff[max_index])
    mean_diff = float(diff.mean())

    if max_diff > ATOL:
        raise AssertionError(
            f"\n"
            f"PARITY FAILURE\n"
            f"stage: {stage}\n"
            f"elements: {expected.size}\n"
            f"max_abs_diff: {max_diff:.8g}\n"
            f"mean_abs_diff: {mean_diff:.8g}\n"
            f"worst_index: {max_index}\n"
            f"expected: {expected[max_index]:.8g}\n"
            f"actual: {actual[max_index]:.8g}\n"
            f"tolerance: {ATOL}\n"
        )


def main() -> None:
    print("Running Python ↔ C numerical parity test...\n")

    for stage in STAGES:
        compare_stage(stage)
        print(f"PASS  {stage}")

    print("\nALL PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
```

Run:

```bash
python tests/test_parity.py
```

---

# 7. WHAT YOU JUST BUILT

This is not just another Python script.

You created an **executable correctness contract**.

Before:

```text
"I looked at the numbers."
```

Now:

```text
the program itself decides whether
the two implementations agree.
```

That is a major software-engineering upgrade.

---

# 8. MAKE THE TEST ACTUALLY FAIL

Never trust a test you have never seen fail.

Temporarily modify the C benchmark:

```c
logits[0] += 100.0f;
```

Run:

```bash
python tests/test_parity.py
```

It must fail.

You should see:

```text
PARITY FAILURE
stage: logits
...
```

Undo the change.

Run again.

It must pass.

This proves the test is not decorative.

---

# 9. IMPORTANT IMPROVEMENT — DO NOT CONFUSE TOLERANCE WITH A FIX

Do NOT do this:

```text
1e-4
↓
1e-2
↓
1e-1
↓
1
```

just to make the test pass.

Tolerance exists because floating-point operations can produce tiny numerical differences.

A large difference means:

```text
possible implementation bug
```

The parity system should make bugs painful and visible.

That is exactly what we want.

---

# 10. STAGE 1 DEFINITION OF DONE

Do not move on until:

```text
[ ] run_pytorch.py works
[ ] C benchmark works
[ ] compare.py works
[ ] test_parity.py works
[ ] all stages are checked
[ ] missing files fail
[ ] shape mismatch fails
[ ] numerical mismatch fails
[ ] first failing stage is reported
[ ] tolerance is documented
[ ] deliberate corruption fails
[ ] restored implementation passes
```

---

# 11. STAGE 2 — BUILD THE C UNIT TEST SUITE

Parity answers:

> Does C match PyTorch?

Unit tests answer:

> Is each C component correct by itself?

These are different questions.

We want:

```text
Tensor
Linear
ReLU
argmax
Conv2D
MaxPool
model loading
```

tested independently.

---

# 12. TESTING PHILOSOPHY

For mathematical code, use tiny examples where you know the answer yourself.

Do not start with:

```text
28 × 28 × 32
```

Start with:

```text
2 numbers
3 numbers
3×3 matrices
```

A tiny test is easier to reason about.

---

# 13. CREATE `tests/test_nn.c`

Start with:

```c
#include "../c/include/nn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>


static void assert_float_close(float actual, float expected, float tolerance)
{
    assert(fabsf(actual - expected) <= tolerance);
}


static void test_argmax(void)
{
    float x[] = {1.0f, 7.0f, 3.0f, 2.0f};

    int result = argmax(x, 4);

    assert(result == 1);
}


static void test_relu(void)
{
    float x[] = {-2.0f, -1.0f, 0.0f, 2.0f};

    relu(x, 4);

    assert_float_close(x[0], 0.0f, 1e-6f);
    assert_float_close(x[1], 0.0f, 1e-6f);
    assert_float_close(x[2], 0.0f, 1e-6f);
    assert_float_close(x[3], 2.0f, 1e-6f);
}


static void test_linear(void)
{
    float W[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float b[] = {
        10.0f,
        20.0f
    };

    float x[] = {
        5.0f,
        6.0f
    };

    float y[2];

    linear(W, b, x, y, 2, 2);

    /*
        y0 = 1*5 + 2*6 + 10 = 27
        y1 = 3*5 + 4*6 + 20 = 59
    */

    assert_float_close(y[0], 27.0f, 1e-6f);
    assert_float_close(y[1], 59.0f, 1e-6f);
}


static void test_tensor_get_set(void)
{
    Tensor t = tensor_alloc(2, 3, 4);

    tensor_set(&t, 1, 2, 3, 42.0f);

    float value = tensor_get(&t, 1, 2, 3);

    assert_float_close(value, 42.0f, 1e-6f);

    tensor_free(&t);

    assert(t.data == NULL);
    assert(t.channels == 0);
    assert(t.height == 0);
    assert(t.width == 0);
}


static void test_conv2d(void)
{
    Tensor input = tensor_alloc(1, 3, 3);

    float values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    for (int i = 0; i < 9; i++) {
        input.data[i] = values[i];
    }

    /*
        Kernel:

        1 0
        0 1
    */

    float kernel[] = {
        1, 0,
        0, 1
    };

    float bias[] = {0};

    Tensor output = conv2d(
        &input,
        kernel,
        bias,
        1,
        2,
        1,
        0
    );

    /*
        Expected:

        6  8
        12 14
    */

    assert_float_close(tensor_get(&output, 0, 0, 0), 6, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 0, 1), 8, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 1, 0), 12, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 1, 1), 14, 1e-6f);

    tensor_free(&input);
    tensor_free(&output);
}


static void test_maxpool(void)
{
    Tensor input = tensor_alloc(1, 4, 4);

    float values[] = {
        1, 8, 2, 4,
        3, 5, 7, 6,
        9, 0, 1, 2,
        4, 3, 8, 5
    };

    for (int i = 0; i < 16; i++) {
        input.data[i] = values[i];
    }

    Tensor output = maxpool2d(&input, 2, 2);

    /*
        Expected:

        8 7
        9 8
    */

    assert_float_close(tensor_get(&output, 0, 0, 0), 8, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 0, 1), 7, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 1, 0), 9, 1e-6f);
    assert_float_close(tensor_get(&output, 0, 1, 1), 8, 1e-6f);

    tensor_free(&input);
    tensor_free(&output);
}


int main(void)
{
    test_argmax();
    test_relu();
    test_linear();
    test_tensor_get_set();
    test_conv2d();
    test_maxpool();

    printf("ALL C UNIT TESTS PASSED\n");

    return 0;
}
```

---

# 14. WHY THESE TESTS ARE IMPORTANT

Notice what we did.

For Linear we manually calculated:

```text
27
59
```

For Conv2D:

```text
6  8
12 14
```

For MaxPool:

```text
8 7
9 8
```

This gives you a powerful debugging strategy.

If parity fails at:

```text
conv3
```

you can trust that the primitive itself has already been tested.

That narrows the search.

---

# 15. COMPILE THE TEST

For GCC:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    tests/test_nn.c \
    c/src/nn.c \
    -o tests/test_nn.exe \
    -lm
```

Run:

```powershell
.\tests\test_nn.exe
```

Expected:

```text
ALL C UNIT TESTS PASSED
```

---

# 16. EXPAND THE TEST SUITE

After the first version works, add tests for:

```text
Tensor
├── dimensions
├── allocation
├── zero initialization
├── set/get
├── channel separation
└── free/reset

ReLU
├── negative
├── zero
└── positive

argmax
├── first
├── middle
├── last
└── ties

Linear
├── one input
├── multiple inputs
├── bias
└── negative values

Conv2D
├── simple kernel
├── multiple channels
├── bias
├── padding
└── stride

MaxPool
├── basic
├── multiple channels
└── stride
```

Do not write them all at once.

Add them one group at a time.

---

# 17. STAGE 2 DEFINITION OF DONE

```text
[ ] test_nn.c exists
[ ] one-command test execution exists
[ ] Tensor tested
[ ] ReLU tested
[ ] argmax tested
[ ] Linear tested
[ ] Conv2D tested
[ ] MaxPool tested
[ ] free/reset tested
[ ] deliberate bug causes failure
[ ] restored code passes
```

---

# 18. STAGE 3 — SANITIZERS

Now we have a test suite.

This makes sanitizers useful.

A sanitizer watches the program while the tests execute.

The two important ones are:

```text
AddressSanitizer
UndefinedBehaviorSanitizer
```

They can expose problems such as:

```text
out-of-bounds access
use-after-free
double-free
invalid memory access
undefined behavior
```

---

# 19. GCC SANITIZER BUILD

Compile:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    -g -O1 \
    -fsanitize=address,undefined \
    tests/test_nn.c \
    c/src/nn.c \
    -o tests/test_nn_sanitized.exe \
    -lm
```

Run:

```powershell
.\tests\test_nn_sanitized.exe
```

The important result is:

```text
ALL C UNIT TESTS PASSED
```

with no sanitizer report.

---

# 20. INTENTIONALLY TRIGGER A SANITIZER

Do not just read about sanitizers.

Make one find something.

Temporarily create an invalid access in a test:

```c
float bad = t.data[100000];
(void)bad;
```

Run the sanitizer build.

You should get a report.

Study:

```text
error type
↓
stack trace
↓
source file
↓
line number
↓
root cause
```

Then remove the bug.

This is how you learn to debug C professionally.

---

# 21. STAGE 3 DEFINITION OF DONE

```text
[ ] normal tests pass
[ ] sanitizer build compiles
[ ] sanitizer tests pass
[ ] intentional memory bug is detected
[ ] sanitizer output can be understood
[ ] no sanitizer errors remain
```

---

# 22. STAGE 4 — REPLACE THE MAGIC `weights.bin` FORMAT

Current situation:

```text
Python exporter
       ↓
raw float bytes
       ↓
C loader
```

The exporter and loader already agree.

But they agree because both programmers know the secret layout.

That is fragile.

---

# 23. WHY RAW BINARY IS DANGEROUS

Suppose you change:

```text
32 channels
```

to:

```text
64 channels
```

Now the C loader may still expect the old number of floats.

The file has no useful self-description.

A robust model file should contain metadata.

Conceptually:

```text
┌─────────────────────┐
│ magic               │
│ version             │
│ architecture ID     │
│ tensor count        │
│ data format         │
├─────────────────────┤
│ tensor 1            │
│ tensor 2            │
│ ...                 │
└─────────────────────┘
```

---

# 24. DESIGN THE HEADER FIRST

Create a format contract.

For example:

```c
#include <stdint.h>

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t architecture;
    uint32_t tensor_count;
} ModelHeader;
```

Use:

```text
magic = "NGNN"
version = 1
architecture = 1
tensor_count = 10
```

The exact format can evolve.

The important lesson is:

> A model file is an interface between two programs.

Python writes it.

C reads it.

Therefore the interface needs a contract.

---

# 25. LOADER VALIDATION

The future loader should follow:

```text
open file
   ↓
read header
   ↓
validate magic
   ↓
validate version
   ↓
validate architecture
   ↓
validate tensor count
   ↓
validate file size
   ↓
read tensors
   ↓
success
```

If something fails:

```text
print useful error
return failure
do not run inference
```

---

# 26. WHY THIS IS BETTER THAN `EXPECTED_BYTES`

The current exporter contains:

```python
EXPECTED_BYTES = 175016
```

This is useful as a sanity check.

But it is not a real serialization protocol.

A hardcoded byte count tells you:

```text
"the file has the size I expected"
```

It does not tell you:

```text
which architecture
which version
which tensor
which format
```

Eventually replace this with metadata-driven validation.

---

# 27. STAGE 4 DEFINITION OF DONE

```text
[ ] model format documented
[ ] magic exists
[ ] version exists
[ ] architecture ID exists
[ ] tensor count exists
[ ] exporter writes header
[ ] C loader reads header
[ ] invalid magic fails
[ ] invalid version fails
[ ] invalid architecture fails
[ ] truncated file fails
[ ] extra/corrupt data is detected
[ ] valid model loads
```

---

# 28. STAGE 5 — PREPROCESSING IS PART OF THE MODEL SYSTEM

Current inference path:

```text
Raylib canvas
      ↓
bounding box
      ↓
square crop
      ↓
margin
      ↓
20×20 resize
      ↓
center into 28×28
      ↓
CNN
```

This is already implemented.

Do not rewrite it.

The next task is to **test it**.

---

# 29. WHY PREPROCESSING MATTERS SO MUCH

Your CNN was trained on:

```text
MNIST distribution
```

The user produces:

```text
your application distribution
```

Those are not identical.

Therefore:

```text
excellent MNIST accuracy
```

does not guarantee:

```text
excellent handwriting accuracy
```

The pipeline between the UI and the CNN matters.

---

# 30. CREATE DETERMINISTIC PREPROCESSING CASES

At minimum:

### Case 1 — empty canvas

Expected:

```text
all 28×28 values = 0
```

### Case 2 — centered square

Verify:

```text
bounding box
resize
centering
```

### Case 3 — digit in upper-left

The final digit should still be centered.

### Case 4 — digit in lower-right

Same.

### Case 5 — very wide input

Verify aspect ratio.

### Case 6 — very tall input

Verify aspect ratio.

### Case 7 — single pixel

Verify the function does not crash.

### Case 8 — canvas edge

Verify no out-of-bounds access.

---

# 31. SAVE INTERMEDIATE PREPROCESSING OUTPUTS

Add a debug mode that can save:

```text
original canvas
↓
bounding box
↓
square crop
↓
resized 20×20
↓
final 28×28
```

Why?

Because if the 28×28 image is wrong:

```text
DO NOT DEBUG THE CNN
```

The CNN is receiving the wrong input.

This is a general ML debugging principle:

> Verify the data pipeline before changing the model.

---

# 32. STAGE 5 DEFINITION OF DONE

```text
[ ] empty input tested
[ ] centered input tested
[ ] corner input tested
[ ] wide input tested
[ ] tall input tested
[ ] edge input tested
[ ] intermediate images can be inspected
[ ] 28×28 output can be inspected
[ ] sanitizer passes
[ ] preprocessing never writes out of bounds
```

---

# 33. STAGE 6 — BUILD A PERSONAL HANDWRITING EVALUATION DATASET

Now we move from:

```text
benchmarking the implementation
```

to:

```text
benchmarking the actual ML system
```

Create:

```text
personal_digits/
    0/
    1/
    2/
    3/
    4/
    5/
    6/
    7/
    8/
    9/
```

Start with:

```text
20 examples per digit
```

That gives:

```text
200 images
```

Later:

```text
50 per digit = 500
100 per digit = 1000
```

---

# 34. IMPORTANT — DO NOT TRAIN ON THIS DATASET YET

At this stage it is an evaluation set.

You want to answer:

> How well does a model trained on MNIST generalize to my handwriting?

If you immediately train on these images, you lose that measurement.

Keep:

```text
MNIST training
MNIST test
personal evaluation
```

separate.

---

# 35. RECORD MORE THAN ACCURACY

For every image record:

```text
true label
predicted label
confidence
```

Then calculate:

```text
overall accuracy
accuracy per digit
confusion matrix
```

---

# 36. CONFUSION MATRIX

Instead of:

```text
accuracy = 92%
```

you want:

```text
true ↓ / predicted →

      0  1  2  3  4  5  6  7  8  9

0    ...
1    ...
2    ...
3    ...
...
```

This reveals patterns such as:

```text
2 → 7
4 → 9
5 → 6
```

The exact errors will come from your dataset.

Do not guess them before measuring.

---

# 37. STAGE 6 DEFINITION OF DONE

```text
[ ] personal dataset exists
[ ] no personal data is used for training yet
[ ] evaluation script exists
[ ] prediction is recorded
[ ] confidence is recorded
[ ] per-digit accuracy exists
[ ] confusion matrix exists
[ ] wrong examples can be inspected
```

---

# 38. STAGE 7 — ERROR ANALYSIS

This is where the project becomes much more ML-oriented.

Instead of:

> "Accuracy is bad. Let's make the CNN bigger."

ask:

> "Why did this example fail?"

For every incorrect prediction store:

```text
original image
preprocessed image
true label
prediction
confidence
```

---

# 39. CLASSIFY FAILURES

Create categories:

```text
PREPROCESSING
    digit too small
    digit too large
    off-center
    cropped incorrectly
    weak contrast

HANDWRITING
    ambiguous
    unusual style
    overlapping strokes

MODEL
    visually similar digits
    insufficient learned representation

DATA
    wrong label
    corrupted image
```

You are building a dataset of failures.

That dataset tells you what to fix.

---

# 40. THE MODEL-CHANGE RULE

Never immediately do:

```text
accuracy bad
↓
more layers
↓
more filters
↓
more epochs
```

Instead:

```text
wrong prediction
      ↓
check preprocessing
      ↓
check Python/C parity
      ↓
check MNIST performance
      ↓
inspect example
      ↓
classify failure
      ↓
design experiment
      ↓
change one thing
      ↓
measure
```

This is the difference between experimentation and random tweaking.

---

# 41. STAGE 8 — CONTROLLED EXPERIMENTS

Now test preprocessing/model hypotheses.

Example:

```text
Experiment A
baseline

Experiment B
larger margin

Experiment C
smaller margin

Experiment D
different centering

Experiment E
different resize strategy
```

Never change five things simultaneously.

---

# 42. EXPERIMENT LOG

Create:

```text
experiments/
    results.csv
    README.md
```

Record:

```text
experiment
date
change
training configuration
MNIST accuracy
personal accuracy
per-digit results
notes
```

Example:

```text
Experiment,Change,MNIST,Personal,Notes
A,Baseline,98.1,74.0,Current system
B,Margin +2px,98.0,77.5,Better centering
C,Margin -2px,98.0,71.2,Worse edge cases
```

Do not invent these values.

Fill them from actual experiments.

---

# 43. EXPERIMENT RULE

A good experiment changes one meaningful variable.

Bad:

```text
new architecture
+
new optimizer
+
new learning rate
+
new preprocessing
```

If performance changes, you do not know why.

Good:

```text
same model
same dataset
same seed
same training configuration
ONLY preprocessing changed
```

Now the result teaches you something.

---

# 44. STAGE 8 DEFINITION OF DONE

```text
[ ] experiment log exists
[ ] baseline recorded
[ ] one-variable experiments performed
[ ] MNIST metric recorded
[ ] personal metric recorded
[ ] observations recorded
[ ] failed experiments are kept
```

A failed experiment is still useful.

It tells you what did not help.

---

# 45. STAGE 9 — PROFILE C INFERENCE

Only now do performance work.

Why?

Because:

```text
fast wrong code
```

is useless.

First:

```text
correct
```

then:

```text
fast
```

---

# 46. WHAT TO MEASURE

Measure:

```text
total inference
preprocessing
Conv1
Conv2
Pool1
Conv3
Conv4
Pool2
Linear
```

Use a timing function around each stage.

Conceptually:

```c
start = timer();

conv1(...);

end = timer();

printf("conv1: %.3f ms\n", elapsed);
```

Do not optimize until you know the bottleneck.

---

# 47. LIKELY OPTIMIZATION AREAS

Once measured, investigate:

```text
repeated index calculations
memory locality
pointer arithmetic
temporary allocations
loop ordering
cache behavior
compiler optimization
```

Do not assume which one is the bottleneck.

Measure it.

---

# 48. OPTIMIZATION LOOP

Every optimization follows:

```text
baseline benchmark
↓
identify bottleneck
↓
change one thing
↓
benchmark
↓
parity test
↓
unit tests
↓
sanitizers
```

If an optimization makes C faster but breaks parity:

```text
optimization is not finished
```

---

# 49. STAGE 9 DEFINITION OF DONE

```text
[ ] inference benchmark exists
[ ] layer timings exist
[ ] bottleneck identified
[ ] one optimization implemented
[ ] before/after measurement recorded
[ ] parity passes
[ ] unit tests pass
[ ] sanitizers pass
```

---

# 50. STAGE 10 — CLEAN UP APPLICATION ARCHITECTURE

Only after the previous stages.

The current project has:

```text
main.c
nn.c
ui.c
```

That is acceptable right now.

Eventually:

```text
c/
├── include/
│   ├── nn.h
│   ├── ui.h
│   ├── preprocessing.h
│   ├── app.h
│   └── model_io.h
│
└── src/
    ├── nn.c
    ├── ui.c
    ├── preprocessing.c
    ├── app.c
    ├── model_io.c
    └── main.c
```

But do not refactor merely because the diagram looks nicer.

Refactor when the current structure creates real problems.

---

# 51. RESPONSIBILITY SEPARATION

Eventually:

```text
main.c
    ↓
application lifecycle

app.c
    ↓
application state/control

ui.c
    ↓
Raylib interaction

preprocessing.c
    ↓
canvas → model input

nn.c
    ↓
neural-network operations

model_io.c
    ↓
model serialization/loading
```

This creates a clean boundary:

```text
UI
↓
application
↓
preprocessing
↓
inference engine
↓
model
```

---

# 52. STAGE 10 DEFINITION OF DONE

```text
[ ] UI does not contain neural-network math
[ ] preprocessing is independently callable
[ ] model loading is isolated
[ ] inference is independently callable
[ ] application control is separated
[ ] tests still pass
[ ] no behavior was accidentally changed
```

---

# 53. STAGE 11 — MULTI-DIGIT RECOGNITION

Only begin after single-digit recognition is reliable.

Current:

```text
one image
↓
one digit
```

Future:

```text
one image
↓
find digits
↓
split digits
↓
preprocess each
↓
CNN
↓
predictions
↓
"12345"
```

The first version should NOT require a new neural network.

---

# 54. SIMPLE OCR PIPELINE

Start with classical computer vision:

```text
input image
    ↓
threshold / binary mask
    ↓
connected components
    ↓
bounding boxes
    ↓
filter noise
    ↓
sort left → right
    ↓
crop each digit
    ↓
MNIST preprocessing
    ↓
CNN
    ↓
concatenate predictions
```

This teaches an important lesson:

> A complete ML system does not necessarily consist only of a neural network.

---

# 55. SEGMENTATION IS THE NEW PROBLEM

The difficult question becomes:

```text
Where does digit 1 end?
Where does digit 2 begin?
```

This introduces:

```text
connected components
bounding boxes
overlap
spacing
noise
segmentation failures
```

That is a real computer-vision problem.

---

# 56. STAGE 11 DEFINITION OF DONE

```text
[ ] multiple digits can be drawn
[ ] connected components are detected
[ ] components are sorted left-to-right
[ ] each component becomes a model input
[ ] CNN predicts each component
[ ] output is combined into a number/string
[ ] segmentation failures are visible
```

---

# 57. STAGE 12 — ADVANCED OCR

Only after the simple system works should you consider:

```text
CNN + sequence model
CNN + CTC
modern vision model
transformer-based OCR
```

The purpose is not to use the fanciest model.

The purpose is to discover when classical segmentation stops being enough.

---

# 58. OPTIONAL — TRAINING IN C

This is intentionally far away.

Current architecture:

```text
Python = training
C      = inference/deployment
```

That is a perfectly valid real-world architecture.

Training in C would require:

```text
forward
↓
loss
↓
gradient
↓
backpropagation
↓
optimizer
↓
parameter update
```

You would need to implement gradients for:

```text
Linear
ReLU
MaxPool
Conv2D
```

plus:

```text
loss
optimizer
parameter storage
training loop
dataset batching
```

This is a separate project inside the project.

Do it only after the inference system is mature.

---

# 59. WHAT NOT TO DO NOW

Do NOT:

```text
rewrite Conv2D
rewrite Tensor
rewrite the CNN
switch frameworks
rewrite Raylib
add random layers
implement C backprop
optimize without profiling
train on your evaluation dataset
jump to transformers
```

None of those solves the current highest-value problems.

Your current weakness is not:

```text
"we need more neural-network code"
```

It is:

```text
"we need stronger engineering around the neural network we already built."
```

---

# 60. THE REAL LEARNING PROGRESSION

This project should now teach you:

## Phase A — ML implementation

Already completed:

```text
dataset
CNN
training
evaluation
export
```

## Phase B — systems implementation

Already completed:

```text
Tensor
Conv2D
MaxPool
Linear
C inference
Raylib
```

## Phase C — correctness engineering

Now:

```text
parity
unit tests
sanitizers
serialization
```

## Phase D — ML engineering

Then:

```text
real-world evaluation
error analysis
experiments
distribution shift
```

## Phase E — systems optimization

Then:

```text
profiling
cache behavior
memory layout
optimization
```

## Phase F — computer vision

Then:

```text
segmentation
multi-digit recognition
OCR
```

This is a much stronger progression than simply adding another CNN layer.

---

# 61. EXACT CURRENT MISSION

You are here:

```text
CURRENT PROJECT
      ↓
STAGE 1
```

Do exactly this:

### Task 1

Run:

```bash
python benchmark/run_pytorch.py
```

### Task 2

Compile the benchmark:

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic \
    benchmark/c_benchmark.c \
    c/src/nn.c \
    -o benchmark/c_benchmark.exe \
    -lm
```

### Task 3

Run:

```powershell
.\benchmark\c_benchmark.exe
```

### Task 4

Run:

```bash
python benchmark/compare.py
```

### Task 5

Create:

```text
tests/test_parity.py
```

using the implementation in this guide.

### Task 6

Run:

```bash
python tests/test_parity.py
```

### Task 7

Intentionally corrupt one value.

### Task 8

Confirm the test fails.

### Task 9

Restore the code.

### Task 10

Confirm the test passes.

### Task 11

Create:

```text
tests/test_nn.c
```

### Task 12

Implement the first C unit tests.

### Task 13

Only after they pass:

```text
STAGE 3 — sanitizers
```

---

# 62. GIT CHECKPOINTS

Do not make one enormous commit at the end.

Use milestones.

Example:

```bash
git add tests/test_parity.py
git commit -m "Add Python C numerical parity test"
```

Then:

```bash
git add tests/test_nn.c
git commit -m "Add C neural network unit tests"
```

Then:

```bash
git commit -m "Add sanitizer test configuration"
```

Then:

```bash
git commit -m "Add versioned model serialization"
```

Each commit should represent a meaningful engineering milestone.

---

# 63. HOW YOU SHOULD USE THIS GUIDE

Do not read the whole document like a textbook.

Use:

```text
one stage
↓
read
↓
open repository
↓
write code
↓
run it
↓
break it
↓
fix it
↓
test it
↓
commit
↓
next stage
```

The guide gives you the problem.

The repository gives you the environment.

The code gives you the implementation.

The tests prove correctness.

The failures teach you.

---

# 64. WHEN YOU GET STUCK

Do not immediately ask for the complete solution.

First collect:

```text
what command you ran
what you expected
what actually happened
error output
relevant code
```

Then debug.

For example:

```text
PARITY FAILURE
stage: conv3
max diff: 0.82
```

Do not say:

> "Conv3 is broken."

Say:

```text
conv1 passes
relu1 passes
conv2 passes
relu2 passes
pool1 passes
conv3 fails
```

Now we know the bug is downstream of:

```text
pool1
```

and at or before:

```text
conv3
```

That is already useful information.

---

# 65. THE CORE DEBUGGING MENTAL MODEL

For this project, constantly ask:

```text
WHAT IS THE CONTRACT?
```

Examples:

Tensor:

```text
index(c,y,x)
```

Conv2D:

```text
weight[output_channel, input_channel, ky, kx]
```

Model:

```text
Conv → ReLU → Conv → ReLU → Pool → ...
```

Serialization:

```text
Python format == C format
```

Parity:

```text
same input → same tensors within tolerance
```

Preprocessing:

```text
canvas → valid MNIST-like tensor
```

Evaluation:

```text
test data must remain unseen during training
```

Optimization:

```text
faster AND still correct
```

Thinking in contracts is one of the main engineering skills this project is supposed to teach.

---

# 66. FINAL ROADMAP

```text
YOU ARE HERE
     │
     ▼
[1] AUTOMATIC NUMERICAL PARITY
     │
     ├── compare every stage
     ├── failure diagnostics
     ├── tolerance
     └── regression test
     │
     ▼
[2] C UNIT TESTS
     │
     ├── Tensor
     ├── Linear
     ├── ReLU
     ├── Conv2D
     ├── MaxPool
     └── model utilities
     │
     ▼
[3] SANITIZERS
     │
     ├── AddressSanitizer
     └── UndefinedBehaviorSanitizer
     │
     ▼
[4] MODEL SERIALIZATION
     │
     ├── magic
     ├── version
     ├── architecture
     └── validation
     │
     ▼
[5] PREPROCESSING TESTING
     │
     ├── deterministic inputs
     ├── edge cases
     └── visual inspection
     │
     ▼
[6] PERSONAL HANDWRITING DATASET
     │
     ├── evaluation only
     ├── per-digit accuracy
     └── confusion matrix
     │
     ▼
[7] ERROR ANALYSIS
     │
     ├── wrong examples
     ├── preprocessing failures
     ├── model failures
     └── distribution shift
     │
     ▼
[8] CONTROLLED EXPERIMENTS
     │
     ├── one variable at a time
     └── experiment log
     │
     ▼
[9] PROFILING
     │
     ├── layer timings
     ├── bottleneck
     └── optimization
     │
     ▼
[10] ARCHITECTURE
     │
     ├── preprocessing module
     ├── model I/O
     └── application separation
     │
     ▼
[11] MULTI-DIGIT
     │
     ├── segmentation
     ├── connected components
     └── recognition
     │
     ▼
[12] ADVANCED OCR
     │
     ├── sequence models
     ├── CTC
     └── modern vision architectures
```

---

# 67. FINAL RULE

The goal is no longer:

> "Can I make the Number Guesser work?"

You already did that.

The new question is:

> **"Can I prove that it works, explain why it works, detect when it stops working, measure how well it works, and improve it scientifically?"**

That is the level this project should reach.

---

# APPENDIX A — EXISTING FOUNDATION

When you forget how an existing implementation works, consult:

```text
foundations-reference.md
```

Do not restart that document from the beginning.

Use it selectively for:

```text
Tensor memory layout
pointer arithmetic
Linear
ReLU
Conv2D
MaxPool
forward pass
model loading
Raylib
preprocessing
PyTorch training
export
```

The foundation is already implemented.

---

# APPENDIX B — SOURCE OF TRUTH

When this guide disagrees with the repository:

```text
repository code wins
```

The important files are:

```text
c/include/nn.h
c/src/nn.c
c/include/ui.h
c/src/ui.c
c/src/main.c

python/model.py
python/train.py
python/evaluate.py
python/export.py

benchmark/run_pytorch.py
benchmark/c_benchmark.c
benchmark/compare.py

CMakeLists.txt
```

If a future architecture change modifies these files, update this guide too.

---

# APPENDIX C — ONE-PAGE CHECKLIST

## NOW

```text
[ ] Run PyTorch reference
[ ] Run C benchmark
[ ] Compare tensors
[ ] Create automated parity test
[ ] Break parity intentionally
[ ] Verify failure
[ ] Restore
[ ] Verify pass
```

## NEXT

```text
[ ] C unit tests
[ ] Sanitizers
[ ] Model serialization
[ ] Preprocessing tests
[ ] Personal evaluation set
[ ] Confusion matrix
[ ] Error analysis
[ ] Controlled experiments
[ ] Profiling
[ ] Optimization
[ ] Architecture cleanup
[ ] Multi-digit recognition
[ ] OCR
```

**Start at Stage 1. Do not skip ahead.**
