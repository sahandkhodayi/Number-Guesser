# Number Guesser — Next-Stage Project Guide

> **This is the guide you read from now.**
>
> You already built the first version of Number Guesser. This document does NOT teach C, Python, tensors, Conv2D, ReLU, MaxPool, or CNNs from zero again.
>
> The existing implementation is the foundation. The goal now is to turn the working project into a measured, tested, debuggable, robust, and progressively more capable ML/C system.

---

# 0. READ THIS FIRST

## Your current state

The repository already contains:

- PyTorch MNIST training
- the CNN architecture
- C Tensor storage
- C Linear
- C ReLU
- C Conv2D
- C MaxPool2D
- C model loading
- C forward pass
- C argmax
- Raylib drawing
- drawing → MNIST preprocessing
- Python model export
- Python evaluation
- Python/C intermediate-tensor benchmark tooling
- a working weights.bin
- an existing C/PyTorch comparison program

So these are NOT future chapters.

You are not starting another beginner CNN tutorial.

## What you should do now

Follow this order:

~~~text
CURRENT PROJECT
      ↓
1. Make Python ↔ C parity automatic
      ↓
2. Build a real C unit-test suite
      ↓
3. Add sanitizers + memory correctness
      ↓
4. Make model serialization/versioning robust
      ↓
5. Build serious preprocessing tests
      ↓
6. Evaluate real handwriting instead of only MNIST
      ↓
7. Analyze failures and improve the pipeline
      ↓
8. Profile and optimize C inference
      ↓
9. Improve the application architecture
      ↓
10. Move from single-digit recognition toward OCR
~~~

**Do not jump ahead.**

Each chapter ends with a concrete definition of done.

---

# 1. PROJECT STATUS MAP

| Area | Current state | What to do |
|---|---|---|
| Tensor | DONE | Reference only |
| Linear | DONE | Reference only |
| ReLU | DONE | Reference only |
| Conv2D | DONE | Reference only |
| MaxPool | DONE | Reference only |
| CNN architecture | DONE | Reference only |
| Forward pass | DONE | Reference only |
| Model export | DONE | Improve later |
| Model loading | DONE | Harden later |
| Raylib UI | DONE | Improve later |
| MNIST training | DONE | Improve later |
| Evaluation | DONE | Expand later |
| Python/C benchmark | STARTED | **DO NOW** |
| Automated tests | NOT BUILT | **DO NOW** |
| Sanitizers | NOT BUILT | Next |
| Model format | Basic raw binary | Later |
| Handwriting dataset | NOT BUILT | Later |
| Error analysis | Basic | Later |
| Performance profiling | NOT BUILT | Later |
| Multi-digit OCR | NOT BUILT | Much later |

---

# 2. THE RULE FOR THIS GUIDE

Every new chapter follows the same pattern:

1. What problem are we solving?
2. Why does the project need it?
3. What is already present?
4. What are we going to change?
5. Exact code to write
6. Explain the new code
7. Run it
8. Break it intentionally
9. Test it
10. Definition of done
11. Only then move on

You do not need to memorize every line before continuing.

The important loop is:

~~~text
understand
   ↓
implement
   ↓
run
   ↓
observe
   ↓
test
   ↓
debug
   ↓
measure
~~~

---

# 3. CHAPTER 1 — MAKE PYTHON ↔ C PARITY AUTOMATIC

## Status

**🟡 TOOLING ALREADY EXISTS — YOUR JOB IS TO FINISH IT**

This is the chapter you should start with.

You already have:

- benchmark/run_pytorch.py
- benchmark/c_benchmark.c
- benchmark/compare.py

The project can already dump intermediate tensors.

The missing step is turning that into a reliable automated verification system.

---

# 4. THE PARITY PROBLEM

Right now, it is possible for this to happen:

~~~text
PyTorch model works
       ↓
C model compiles
       ↓
C model predicts something
       ↓
you think everything is correct
       ↓
one layer is actually numerically wrong
~~~

A prediction matching by coincidence is not proof.

We want:

~~~text
same input
   ↓
PyTorch ───────── C
   ↓                ↓
conv1             conv1
   ↓                ↓
relu1             relu1
   ↓                ↓
...
   ↓                ↓
logits            logits
~~~

Then compare every stage.

---

# 5. UNDERSTAND THE PARITY CONTRACT

The two implementations must agree on:

### Input

~~~text
1 × 28 × 28
float32
channel-first
same pixel values
~~~

### Weight ordering

~~~text
conv1 weights
conv1 bias
conv2 weights
conv2 bias
conv3 weights
conv3 bias
conv4 weights
conv4 bias
linear weights
linear bias
~~~

### Tensor ordering

C uses:

~~~text
((channel * height) + y) * width + x
~~~

Python/PyTorch tensors must be exported consistently with that layout.

### Operation ordering

~~~text
Conv1
ReLU
Conv2
ReLU
Pool1
Conv3
ReLU
Conv4
ReLU
Pool2
Flatten
Linear
~~~

If any one of these contracts changes, parity can fail.

---

# 6. FIRST TASK — RUN THE EXISTING PARITY PIPELINE

Before changing code, run the existing tools.

Conceptually:

~~~bash
python benchmark/run_pytorch.py
~~~

Then build/run the C benchmark.

Then:

~~~bash
python benchmark/compare.py
~~~

Your first goal is NOT to improve the model.

Your goal is to answer:

> Do the two implementations currently agree?

---

# 7. HOW TO READ A PARITY FAILURE

Suppose you get:

~~~text
Stage                 max_abs_diff
input                 0
conv1                 0.000001
relu1                 0.000001
conv2                 0.000002
relu2                 0.000002
pool1                 0.000002
conv3                 0.81
relu3                 0.82
...
~~~

Do NOT debug relu3.

The first meaningful failure is:

~~~text
conv3
~~~

Therefore investigate:

- conv3 weight ordering
- input tensor shape
- C convolution indexing
- padding
- stride
- bias
- exported weights

The key debugging idea:

> **Find the first divergence, not the final symptom.**

---

# 8. TURN THE COMPARISON INTO A REAL TEST

Create:

~~~text
tests/
    test_parity.py
~~~

The test should:

1. generate/reference deterministic input
2. run both implementations
3. compare every stage
4. use a defined tolerance
5. return a non-zero exit code when a stage fails

Basic comparison logic:

~~~python
import numpy as np

def assert_close(name, expected, actual, atol=1e-4):
    if expected.shape != actual.shape:
        raise AssertionError(
            f"{name}: shape mismatch: "
            f"{expected.shape} != {actual.shape}"
        )

    diff = np.abs(expected - actual)
    max_diff = float(diff.max())

    if max_diff > atol:
        raise AssertionError(
            f"{name}: max difference {max_diff} > {atol}"
        )
~~~

## What is new here?

You already know NumPy.

The new concept is **turning numerical correctness into an executable contract**.

Instead of:

> "I looked at the numbers and they seem fine."

you get:

> "The program refuses to pass if Conv3 differs beyond tolerance."

---

# 9. TOLERANCE IS NOT "MAKE IT PASS"

Do not solve failures by changing:

~~~text
1e-4
→
1e-1
→
1
~~~

until the test passes.

The tolerance represents expected floating-point differences.

If C and PyTorch perform mathematically equivalent float32 operations, tiny differences are normal.

A large difference is evidence of a bug.

Record:

- max absolute difference
- mean absolute difference
- tensor shape
- first failing stage

Later we can add relative error and worst-index reporting.

---

# 10. MAKE THE FAILURE USEFUL

Upgrade the failure message.

Target output:

~~~text
PARITY FAILURE

Stage: conv3
Expected shape: (32, 7, 7)
Actual shape:   (32, 7, 7)

max abs diff:   0.83241
mean abs diff:  0.09124

Worst element:
channel = 17
y       = 3
x       = 5

Expected: 1.23891
Actual:   0.40650
~~~

This turns the benchmark from a demo into a debugging tool.

---

# 11. CHAPTER 1 — DEFINITION OF DONE

- [ ] Python reference tensors are deterministic
- [ ] C tensors are dumped at the same stages
- [ ] comparison checks every stage
- [ ] shape mismatches fail
- [ ] numerical mismatches fail
- [ ] the first bad stage is reported
- [ ] tolerance is explicitly documented
- [ ] a successful run has a clear PASS message
- [ ] intentionally corrupting one value makes the test fail

### Deliberate bug

Temporarily change one C value:

~~~c
sum += 100.0f;
~~~

Run parity.

It MUST fail.

Undo the change.

Run again.

It MUST pass.

That is your first real regression test.

---

# 12. CHAPTER 2 — BUILD A REAL C UNIT TEST SUITE

## Status

**🔴 NOT BUILT**

The repository currently has tests/.gitkeep.

That means the project has a place for tests but not an actual test suite.

Now we build one.

---

# 13. WHY PARITY IS NOT ENOUGH

Parity tests answer:

> Does our implementation match PyTorch?

Unit tests answer:

> Does this individual function behave correctly?

For example:

~~~text
tensor_get()
tensor_set()
relu()
argmax()
linear()
conv2d()
maxpool2d()
model_load()
~~~

A parity failure tells you something is wrong.

A unit test can tell you **which primitive is broken**.

---

# 14. FIRST C TEST — ARGMAX

Create:

~~~text
tests/test_nn.c
~~~

Start with tiny deterministic tests.

~~~c
#include "../c/include/nn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_argmax(void) {
    float x[] = {1.0f, 7.0f, 3.0f, 2.0f};

    int result = argmax(x, 4);

    assert(result == 1);
}

int main(void) {
    test_argmax();

    printf("All tests passed.\\n");
    return 0;
}
~~~

---

# 15. WHY THIS TEST MATTERS

The test has three parts:

~~~text
Arrange
   ↓
Act
   ↓
Assert
~~~

Arrange:

~~~c
float x[] = {1.0f, 7.0f, 3.0f, 2.0f};
~~~

Act:

~~~c
int result = argmax(x, 4);
~~~

Assert:

~~~c
assert(result == 1);
~~~

This pattern will be used throughout the test suite.

---

# 16. TESTS TO ADD IN ORDER

Do not write 100 tests at once.

Build them in this order.

## Tensor tests

~~~text
allocation dimensions
allocation initializes memory
set/get round trip
different channels do not overlap
free resets the tensor
~~~

## Math tests

~~~text
ReLU positive
ReLU negative
ReLU zero
argmax first element
argmax middle element
argmax last element
~~~

## Linear tests

Use tiny hand-calculated values.

~~~text
W = [1 2
     3 4]

b = [10
     20]

x = [5
     6]
~~~

Expected:

~~~text
y0 = 1*5 + 2*6 + 10 = 27
y1 = 3*5 + 4*6 + 20 = 59
~~~

Then assert:

~~~c
assert(fabsf(y[0] - 27.0f) < 1e-6f);
assert(fabsf(y[1] - 59.0f) < 1e-6f);
~~~

This is more useful than testing only the final CNN.

---

# 17. CONV2D UNIT TEST

You already implemented Conv2D.

**Do not rewrite Conv2D.**

Now test it with a tiny tensor.

Input:

~~~text
1 channel
3 × 3

1 2 3
4 5 6
7 8 9
~~~

Kernel:

~~~text
1 0
0 1
~~~

Expected output:

~~~text
6 8
12 14
~~~

First window:

~~~text
1*1 + 2*0
4*0 + 5*1
= 6
~~~

The lesson is creating **small problems whose correct answer you can calculate yourself**.

---

# 18. MAXPOOL TEST

Input:

~~~text
1 8 2 4
3 5 7 6
9 0 1 2
4 3 8 5
~~~

For:

~~~text
kernel = 2
stride = 2
~~~

Expected:

~~~text
8 7
9 8
~~~

Test exactly that.

---

# 19. MEMORY TESTS

Now test ownership.

Test:

~~~c
Tensor t = tensor_alloc(...);

assert(t.data != NULL);

tensor_free(&t);

assert(t.data == NULL);
assert(t.channels == 0);
assert(t.height == 0);
assert(t.width == 0);
~~~

You are no longer only testing ML mathematics.

You are testing whether your C program manages memory correctly.

---

# 20. CHAPTER 2 — DEFINITION OF DONE

- [ ] test executable exists
- [ ] tensor tests exist
- [ ] ReLU test exists
- [ ] argmax test exists
- [ ] Linear test exists
- [ ] Conv2D test exists
- [ ] MaxPool test exists
- [ ] memory cleanup is tested
- [ ] one intentional bug causes a test failure
- [ ] tests can be run with one command

---

# 21. CHAPTER 3 — SANITIZERS AND MEMORY CORRECTNESS

## Status

**🔴 NEW**

Once unit tests exist, run them with tools that look for C memory bugs.

Important tools:

~~~text
AddressSanitizer
UndefinedBehaviorSanitizer
~~~

For GCC/Clang, conceptually:

~~~text
-fsanitize=address,undefined
-g
-O1
~~~

Do not memorize the flags.

Understand what they provide.

---

# 22. WHY SANITIZERS COME AFTER TESTS

A sanitizer without meaningful tests may never execute the broken path.

You want:

~~~text
unit test
   ↓
execute code
   ↓
sanitizer observes memory
   ↓
bug reported
~~~

Possible bugs:

- reading outside a tensor
- writing outside an array
- use-after-free
- double free
- invalid pointer usage
- undefined behavior

---

# 23. INTENTIONALLY BREAK THE PROGRAM

Create a temporary test bug.

For example, deliberately access one element outside the allocated tensor.

Run the sanitizer build.

Observe the report.

Then fix it.

Learn to read:

~~~text
ERROR
  ↓
type of memory error
  ↓
stack trace
  ↓
source line
  ↓
root cause
~~~

---

# 24. CHAPTER 3 — DEFINITION OF DONE

- [ ] normal test build works
- [ ] sanitizer build works
- [ ] tests pass under sanitizer
- [ ] you intentionally triggered an error
- [ ] you can identify the source line from the sanitizer report
- [ ] no sanitizer errors remain

---

# 25. CHAPTER 4 — STOP USING A "MAGIC" RAW MODEL FILE

## Status

**🟡 CURRENT IMPLEMENTATION WORKS — DESIGN NEEDS TO MATURE**

Currently weights.bin is basically raw float data in a known order.

The C loader knows exactly how many bytes it expects.

That works.

But it has a weakness.

What happens if you change the model?

---

# 26. THE CURRENT FAILURE MODE

Suppose tomorrow you change:

~~~text
32 channels
→
64 channels
~~~

The loader and exported model may no longer agree.

A binary file should be able to identify itself.

Eventually move toward:

~~~text
HEADER
──────
magic
version
dtype
architecture ID
tensor count
metadata
──────
TENSOR DATA
~~~

---

# 27. DESIGN THE FORMAT BEFORE CODING

Do not immediately write a serializer.

First design the contract.

Example:

~~~c
typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t tensor_count;
    uint32_t dtype;
} ModelHeader;
~~~

Possible magic:

~~~text
NGNN
~~~

The exact format can change during this chapter.

The important new concept:

> **A model file is an interface between two programs.**

Python produces it.

C consumes it.

Therefore the format must be explicit.

---

# 28. VERSIONING

Imagine:

~~~text
version 1
version 2
version 3
~~~

The loader should not silently interpret version 3 as version 1.

Instead:

~~~text
read header
   ↓
check magic
   ↓
check version
   ↓
check architecture
   ↓
check file size
   ↓
load tensors
~~~

If anything fails:

~~~text
clear error
+
do not run inference
~~~

---

# 29. CHAPTER 4 — DEFINITION OF DONE

- [ ] model format documented
- [ ] magic value exists
- [ ] version exists
- [ ] architecture/model identifier exists
- [ ] loader validates the header
- [ ] corrupted files fail cleanly
- [ ] unsupported versions fail clearly
- [ ] exporter and loader agree automatically

---

# 30. CHAPTER 5 — TEST THE PREPROCESSING PIPELINE

## Status

**🟡 IMPLEMENTED — NOT SERIOUSLY TESTED**

Your current pipeline is:

~~~text
280×280 drawing
      ↓
find bounding box
      ↓
square crop
      ↓
margin
      ↓
bilinear resize
      ↓
20×20
      ↓
center in 28×28
~~~

This is one of the most important parts of the real application.

The model was trained on MNIST.

The user does not draw MNIST.

---

# 31. WHY THIS MATTERS

You can have:

~~~text
excellent MNIST accuracy
+
bad user predictions
~~~

without the CNN being wrong.

The difference is:

~~~text
training distribution
        vs
real input distribution
~~~

This is a machine-learning engineering problem, not merely a UI problem.

---

# 32. CREATE PREPROCESSING TEST INPUTS

You need deterministic cases.

### Test A — empty canvas

Expected:

~~~text
all zeros
~~~

### Test B — centered square

Check:

- bounding box
- output location
- output size

### Test C — tiny digit in upper-left

Check that the digit becomes centered.

### Test D — digit touching an edge

Check crop bounds.

### Test E — very wide digit

Check aspect-ratio preservation.

### Test F — very tall digit

Same.

---

# 33. SAVE PREPROCESSING OUTPUTS

Add a debugging mode that can save:

~~~text
original 280×280
bounding box
crop
resized 20×20
final 28×28
~~~

If the final image looks wrong, debugging the CNN is pointless.

---

# 34. CHAPTER 5 — DEFINITION OF DONE

- [ ] empty canvas is tested
- [ ] edge cases are tested
- [ ] preprocessing output can be saved
- [ ] final 28×28 input can be inspected
- [ ] preprocessing never writes outside bounds
- [ ] several real handwritten digits produce sensible 28×28 inputs

---

# 35. CHAPTER 6 — BUILD A REAL HANDWRITING EVALUATION SET

## Status

**🔴 NEW**

MNIST test accuracy answers:

> How well does the model recognize MNIST test images?

It does NOT answer:

> How well does it recognize my handwriting?

Create a small personal evaluation dataset.

Example:

~~~text
my_digits/
    0/
    1/
    2/
    ...
    9/
~~~

Start small.

Even 20 examples per digit gives 200 evaluation images.

---

# 36. DO NOT TRAIN ON IT YET

At first this dataset is for evaluation.

If you train on it immediately, you lose the ability to measure generalization to your handwriting.

Track:

~~~text
overall accuracy
accuracy per digit
confusion matrix
confidence
wrong examples
~~~

---

# 37. CONFUSION MATRIX

Instead of only:

~~~text
accuracy = 91%
~~~

you want:

~~~text
true digit → predicted digit
~~~

Example:

~~~text
        predicted
        0 1 2 3 ...
true 0  18 0 1 0 ...
true 1   0 20 0 0 ...
true 2   1 0 16 2 ...
~~~

This tells you what the model actually struggles with.

---

# 38. CHAPTER 6 — DEFINITION OF DONE

- [ ] personal evaluation dataset exists
- [ ] evaluation does not train
- [ ] predictions are recorded
- [ ] per-digit accuracy exists
- [ ] confusion matrix exists
- [ ] wrong examples can be inspected
- [ ] confidence is recorded

---

# 39. CHAPTER 7 — ERROR ANALYSIS

## Status

**🔴 NEW**

Now stop asking:

> "Is the accuracy good?"

and start asking:

> "Why is this example wrong?"

For every wrong prediction, collect:

~~~text
image
true label
prediction
confidence
preprocessed image
~~~

Then classify the failure:

~~~text
preprocessing
ambiguous handwriting
model confusion
too small
too large
off-center
broken drawing
low contrast
wrong label
~~~

This is much more useful than randomly changing the CNN.

---

# 40. THE MODEL-IMPROVEMENT RULE

Do not change the architecture because:

> "Maybe more layers will fix it."

First determine the failure source.

Use:

~~~text
Wrong prediction
      ↓
Is preprocessing correct?
      ↓
yes
      ↓
Does MNIST evaluation work?
      ↓
yes
      ↓
Does C match PyTorch?
      ↓
yes
      ↓
Analyze handwriting distribution
      ↓
Only then consider model changes
~~~

This prevents architecture changes from hiding engineering bugs.

---

# 41. CHAPTER 8 — PREPROCESSING EXPERIMENTS

Now scientifically compare preprocessing strategies.

Possible experiments:

### Version A

Current:

~~~text
bounding box
→ square
→ margin
→ bilinear
→ center
~~~

### Version B

Different margin.

### Version C

Different target size.

### Version D

Different centering method.

### Version E

Stroke normalization.

Do not change five things at once.

Use an experiment table:

| Experiment | Change | MNIST | Personal | Notes |
|---|---|---:|---:|---|
| A | baseline | ... | ... | current |
| B | margin | ... | ... | |
| C | target size | ... | ... | |
| D | centering | ... | ... | |

The point is learning experimental methodology.

---

# 42. CHAPTER 9 — PROFILE THE C INFERENCE ENGINE

## Status

**🔴 NEW**

Once correctness is established, measure performance.

Do not optimize before this point.

Measure at minimum:

~~~text
total inference time
Conv1 time
Conv2 time
Pool1 time
Conv3 time
Conv4 time
Pool2 time
Linear time
preprocessing time
~~~

You may discover that one layer dominates runtime.

Then optimize that layer instead of guessing.

---

# 43. OPTIMIZATION ORDER

Always:

~~~text
correctness
↓
measure
↓
find bottleneck
↓
optimize bottleneck
↓
measure again
↓
parity test again
~~~

Never:

~~~text
"I think this loop is slow"
↓
rewrite everything
~~~

Possible future optimizations:

- reduce repeated index calculations
- improve pointer arithmetic
- improve memory locality
- reuse temporary buffers
- reduce unnecessary allocations
- compiler optimization flags
- better convolution loop ordering

Every optimization must still pass:

- unit tests
- sanitizer tests
- parity tests

---

# 44. CHAPTER 9 — DEFINITION OF DONE

- [ ] inference benchmark exists
- [ ] layer timings exist
- [ ] bottleneck identified from measurements
- [ ] one optimization implemented
- [ ] before/after numbers recorded
- [ ] parity still passes
- [ ] sanitizer still passes
- [ ] unit tests still pass

---

# 45. CHAPTER 10 — IMPROVE THE C APPLICATION ARCHITECTURE

The current main.c contains UI and application control logic.

That is acceptable at the current size.

Eventually a cleaner structure may be:

~~~text
c/
├── include/
│   ├── nn.h
│   ├── ui.h
│   ├── app.h
│   └── preprocessing.h
│
└── src/
    ├── nn.c
    ├── ui.c
    ├── app.c
    ├── preprocessing.c
    └── main.c
~~~

Do not refactor this immediately.

First finish correctness and testing.

When this chapter arrives, the refactor should be driven by actual pain in the codebase.

---

# 46. CHAPTER 11 — MULTI-DIGIT RECOGNITION

## Status

**🔮 MUCH LATER**

Only start this after single-digit recognition is reliable.

Current problem:

~~~text
one canvas
→ one digit
~~~

Future:

~~~text
image
 ↓
segment digits
 ↓
digit 1 → CNN
digit 2 → CNN
digit 3 → CNN
digit 4 → CNN
digit 5 → CNN
 ↓
"12345"
~~~

This introduces a new computer-vision problem:

> Where does one digit end and the next digit begin?

---

# 47. FIRST MULTI-DIGIT APPROACH

Do not immediately build a transformer.

Start with segmentation.

Conceptually:

~~~text
binary image
    ↓
find connected regions
    ↓
bounding boxes
    ↓
sort left → right
    ↓
crop each region
    ↓
preprocess each crop
    ↓
CNN prediction
    ↓
concatenate digits
~~~

This is the first step toward OCR.

---

# 48. CHAPTER 12 — SEQUENCE/OCR MODEL

## Status

**🔮 OPTIONAL FUTURE**

Only investigate this if segmentation becomes limiting.

Possible future direction:

~~~text
CNN
+
sequence model
~~~

or a modern vision architecture.

This is intentionally far away.

Do not jump here while single-digit inference is still being hardened.

---

# 49. CHAPTER 13 — OPTIONAL C BACKPROPAGATION

## Status

**🔮 VERY LATE**

You already have inference.

Training in C would require:

~~~text
forward
↓
loss
↓
gradient calculation
↓
backpropagation
↓
parameter update
↓
optimizer
~~~

This is substantially larger than inference.

The current architecture already gives you an important engineering separation:

~~~text
Python = training
C      = deployment
~~~

C backpropagation is a later research/learning phase, not the next task.

---

# 50. WHAT YOU SHOULD NOT DO NOW

Do not currently:

- rewrite Conv2D
- rewrite the Tensor struct
- rewrite the CNN architecture
- rewrite the UI from scratch
- add random layers
- add C backprop
- switch frameworks
- optimize without measurements
- train on your personal evaluation set immediately
- replace working code just to make the project look more advanced

Your next progress should come from **engineering depth**, not random new features.

---

# 51. YOUR CURRENT EXACT MISSION

## START HERE

### Step 1

Run:

~~~text
benchmark/run_pytorch.py
~~~

### Step 2

Build/run the C benchmark.

### Step 3

Run:

~~~text
benchmark/compare.py
~~~

### Step 4

Record the result.

### Step 5

Turn the comparison into a test that can fail automatically.

### Step 6

Intentionally break one value.

### Step 7

Confirm the test catches it.

### Step 8

Undo the bug.

### Step 9

Only when parity passes, begin:

~~~text
tests/test_nn.c
~~~

---

# 52. PROJECT ROADMAP FROM HERE

~~~text
                 YOU ARE HERE
                      │
                      ▼
          ┌──────────────────────┐
          │ 1. Numerical Parity  │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 2. C Unit Tests      │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 3. Sanitizers        │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 4. Model Format      │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 5. Preprocessing     │
          │    Test Suite        │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 6. Personal Dataset  │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 7. Error Analysis    │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 8. Experiments       │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 9. Profiling         │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 10. Architecture     │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 11. Multi-digit OCR  │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ 12. Advanced ML      │
          └──────────────────────┘
~~~

---

# 53. HOW TO USE THE OLD MATERIAL

The old code-first textbook is preserved as a **foundation/reference document**, not the main path.

You should NOT reread it from line 1.

Use it only when you need to revisit:

- Tensor memory layout
- C pointer arithmetic
- Conv2D implementation
- MaxPool
- Linear
- model loading
- forward pass
- Raylib
- Python training
- export
- basic CNN mathematics

Reference:

**[Foundation / completed implementation reference](foundations-reference.md)**

---

# 54. COMPLETED IMPLEMENTATIONS — QUICK REFERENCE

These are already in the project.

## C

~~~text
c/src/nn.c
├── linear
├── relu
├── argmax
├── tensor_alloc
├── tensor_free
├── tensor_get
├── tensor_set
├── relu_tensor
├── conv2d
├── maxpool2d
├── model_load
└── model_forward
~~~

## UI

~~~text
c/src/ui.c
├── canvas_clear
├── canvas_draw_point
├── canvas_draw_line
├── bilinear sampling
└── canvas_to_mnist_input
~~~

## Python

~~~text
python/model.py
python/train.py
python/evaluate.py
python/export.py
python/dataset.py
~~~

## Benchmark

~~~text
benchmark/run_pytorch.py
benchmark/c_benchmark.c
benchmark/compare.py
~~~

These are foundations.

**Do not mistake "already implemented" for "never think about it again."**

We will test, measure, and improve them later.

---

# 55. WHAT "DONE" MEANS FROM NOW ON

A feature is not DONE just because:

~~~text
it compiles
~~~

or:

~~~text
it works once
~~~

For this project, DONE means:

~~~text
implemented
+
tested
+
measured where appropriate
+
failure behavior understood
+
documented
~~~

That is the standard for the next stage.

---

# 56. FINAL RULE

Do not read this guide like a normal book.

Read **one chapter**, then return to the repository.

For example:

~~~text
Read Chapter 1
     ↓
open benchmark/
     ↓
run it
     ↓
modify code
     ↓
break it
     ↓
fix it
     ↓
commit it
     ↓
Chapter 2
~~~

The project is the textbook.

The guide tells you what experiment to perform.

The code is what you learn from.

The tests are what prove you learned it.

---

# APPENDIX — COMPLETED FOUNDATION MATERIAL

Everything below the main roadmap is intentionally reference-oriented.

If you already understand a section, skip it.

The previous long code-first textbook is preserved in:

**[foundations-reference.md](foundations-reference.md)**

Use it when you need to inspect the reasoning behind an existing implementation.

---

# CURRENT SOURCE OF TRUTH

When the guide and the repository disagree:

**the repository wins.**

Important current files:

- c/include/nn.h
- c/src/nn.c
- c/include/ui.h
- c/src/ui.c
- c/src/main.c
- python/model.py
- python/train.py
- python/evaluate.py
- python/export.py
- benchmark/run_pytorch.py
- benchmark/c_benchmark.c
- benchmark/compare.py

---

# ONE-PAGE CHECKLIST

## Current milestone

### Numerical parity

- [ ] run PyTorch reference
- [ ] run C reference
- [ ] compare all stages
- [ ] report first mismatch
- [ ] automate pass/fail
- [ ] intentionally break it
- [ ] verify failure
- [ ] restore
- [ ] commit

### Then

- [ ] C unit tests
- [ ] sanitizers
- [ ] robust model format
- [ ] preprocessing tests
- [ ] personal handwriting evaluation
- [ ] confusion matrix
- [ ] error analysis
- [ ] preprocessing experiments
- [ ] profiling
- [ ] optimization
- [ ] architecture cleanup
- [ ] multi-digit recognition
- [ ] advanced OCR

**Start with Chapter 1. Do not jump to Chapter 11.**
