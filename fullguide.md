
# Number Guesser — Project Guide

> This is the living textbook for the repository.
>
> Read this file while working on the project. It is written around the actual current repository, not an imagined finished version.
>
> The goal is not to give you code to blindly copy. The goal is to make you understand why the code exists, how the pieces connect, how to prove they are correct, and what to build next.

---

# 0. Start Here

## 0.1 What this project actually is

Number Guesser is now a small end-to-end machine-learning systems project:

    MNIST
      ↓
    PyTorch CNN
      ↓
    training
      ↓
    saved model
      ↓
    binary weight export
      ↓
    C inference engine
      ↓
    Raylib drawing application
      ↓
    user draws a digit
      ↓
    preprocessing
      ↓
    CNN inference
      ↓
    10 logits
      ↓
    prediction + probabilities

You are learning several things at once:

- Python
- PyTorch
- neural-network architecture
- convolution
- tensors and shapes
- training and evaluation
- C
- pointers and memory
- manual tensor operations
- binary serialization
- numerical correctness
- CMake
- debugging
- benchmarking
- computer-vision preprocessing
- eventually OCR and sequence models

The important part is that the same mathematical operation exists in multiple forms.

For example:

    Math:
        y = Wx + b

    PyTorch:
        nn.Linear(...)

    C:
        linear(W, b, x, y, ...)

Your job is to understand the relationship between all three.

---

# 1. Repository Status — Read This First

The repository already contains a substantial first version.

## 1.1 Already implemented

These are NOT your next implementation tasks.

### Python

Already present:

- MNIST dataset loading
- train/test DataLoaders
- CNN model
- training loop
- evaluation loop
- saved PyTorch state dict
- binary weight exporter
- accuracy helper

Files:

    python/dataset.py
    python/model.py
    python/train.py
    python/evaluate.py
    python/export.py
    python/helper_functions.py

### C

The inference engine already contains:

- tensor allocation/free
- tensor indexing
- linear layer
- ReLU
- argmax
- Conv2D
- MaxPool2D
- model loading
- complete forward pass

Files:

    c/include/nn.h
    c/src/nn.c

### Raylib application

Already present:

- 280×280 drawing canvas
- brush drawing
- line interpolation
- clearing
- preprocessing to 28×28
- model inference
- softmax
- probability bars
- prediction display
- keyboard/mouse controls

Files:

    c/include/ui.h
    c/src/ui.c
    c/src/main.c

### Verification tooling

There is already a C verification executable:

    c/tools/verify.c

It runs the C network layer by layer and prints shapes and sample values.

### Benchmarking

There is already a benchmark directory:

    benchmark/
    ├── Makefile
    ├── README.md
    ├── c_benchmark.c
    ├── compare.py
    ├── model.py
    └── run_pytorch.py

### Build system

The application uses:

    CMakeLists.txt

### Model artifact

The repository currently contains:

    models/weights.bin

The exported file is 175016 bytes.

### Not implemented yet

The repository does NOT yet contain a completed:

- automated C unit-test suite
- automated layer-by-layer Python↔C parity report
- CI pipeline
- sanitizer test workflow
- profiling/optimization system
- custom handwriting dataset
- two-digit segmentation system
- variable-length OCR system
- CTC implementation
- C backpropagation/training system

Those are future milestones.

---

# 2. The Real Learning Path

Do not follow the old guide from page 1 to page 200.

Follow this path:

    0. Understand the current repository
            ↓
    1. Understand the CNN mathematically
            ↓
    2. Understand the Python training pipeline
            ↓
    3. Understand the C tensor/memory system
            ↓
    4. Prove Python and C are numerically equivalent
            ↓
    5. Build a real automated test suite
            ↓
    6. Make the runtime/build contract robust
            ↓
    7. Understand preprocessing deeply
            ↓
    8. Measure real handwriting performance
            ↓
    9. Improve the ML system through experiments
            ↓
    10. Profile and optimize C
            ↓
    11. Build multi-digit recognition
            ↓
    12. Study sequence OCR / CTC
            ↓
    13. Optional: implement backpropagation yourself

The order matters.

Do not optimize a network that you have not verified.

Do not change the CNN because your own handwriting performs badly before checking preprocessing.

Do not implement C backpropagation before the C forward pass is trustworthy.

---

# 3. The Model — Understand This Before Coding More

The current model is:

    Input
    1 × 28 × 28

    Conv1
    1 → 32 channels
    3 × 3
    stride 1
    padding 1

    ReLU

    Conv2
    32 → 32
    3 × 3
    stride 1
    padding 1

    ReLU

    MaxPool
    2 × 2
    stride 2

    Conv3
    32 → 32
    3 × 3
    padding 1

    ReLU

    Conv4
    32 → 32
    3 × 3
    padding 1

    ReLU

    MaxPool
    2 × 2

    Flatten

    Linear
    1568 → 10

    Output
    10 logits

Shape flow:

    1 × 28 × 28
        ↓
    32 × 28 × 28
        ↓
    32 × 28 × 28
        ↓
    32 × 14 × 14
        ↓
    32 × 14 × 14
        ↓
    32 × 14 × 14
        ↓
    32 × 14 × 14
        ↓
    32 × 7 × 7
        ↓
    1568
        ↓
    10

Why 1568?

    32 × 7 × 7 = 1568

Why 10?

MNIST contains ten classes:

    0 1 2 3 4 5 6 7 8 9

The final layer therefore produces ten raw scores.

---

# 4. Convolution — You Must Understand the Loops

Conceptually:

    output[oc, oy, ox]
      =
    bias[oc]
      +
    Σ input[ic, iy, ix] * weight[oc, ic, ky, kx]

The C implementation turns that equation into nested loops:

    for each output channel:
        for each output y:
            for each output x:
                sum = bias

                for each input channel:
                    for each kernel y:
                        for each kernel x:
                            sum += input * weight

                output = sum

Do not memorize the loops.

Understand why every loop exists.

There are six meaningful dimensions:

- output channel
- output row
- output column
- input channel
- kernel row
- kernel column

That is why Conv2D is much more expensive than a simple linear layer.

---

# 5. Convolution Shape Mathematics

For one spatial dimension:

    out = floor((N + 2P - K) / S) + 1

Where:

- N = input size
- P = padding
- K = kernel size
- S = stride

For this project:

    N = 28
    P = 1
    K = 3
    S = 1

Therefore:

    (28 + 2 - 3) / 1 + 1
    = 28

So the convolution preserves 28×28.

This is why all four convolutions keep the spatial dimensions unchanged.

---

# 6. MaxPool Mathematics

For pooling:

    out = floor((N - K) / S) + 1

For:

    K = 2
    S = 2

we get:

    28 → 14
    14 → 7

Therefore:

    32 × 28 × 28
    →
    32 × 14 × 14
    →
    32 × 7 × 7

This is the reason the linear layer receives exactly 1568 values.

---

# 7. ReLU

ReLU is:

    ReLU(x) = max(0, x)

Examples:

    -4 → 0
    -1 → 0
     0 → 0
     2 → 2
     7 → 7

In C it is simply a loop over the tensor.

Important idea:

ReLU does not change tensor shape.

It changes values.

---

# 8. Logits, Softmax, and Prediction

The final linear layer produces:

    z = Wx + b

These values are logits.

Example:

    [1.2, -0.4, 3.7, 0.8, ...]

The predicted class is:

    argmax(z)

You do not need softmax to choose the class.

Softmax is useful when you want normalized probabilities:

    p_i = exp(z_i) / Σ exp(z_j)

The C UI subtracts the maximum logit before exponentiating:

    exp(z_i - max(z))

This is a numerical-stability trick.

It does not change the resulting probability distribution.

---

# 9. Python Training Pipeline

## 9.1 dataset.py

The current dataset pipeline uses ToTensor().

This converts MNIST images into tensors approximately shaped as:

    [1, 28, 28]

with values in:

    [0, 1]

The training loader shuffles.

The test loader does not.

Understand why.

Training benefits from random ordering.

Evaluation should be deterministic and representative.

---

# 10. train.py — What Actually Happens

A training step is:

    input
      ↓
    model(input)
      ↓
    logits
      ↓
    loss(logits, labels)
      ↓
    optimizer.zero_grad()
      ↓
    loss.backward()
      ↓
    optimizer.step()

Important distinctions:

### Forward pass

Computes predictions.

### Loss

Measures how wrong the predictions are.

### Backward pass

Computes gradients.

### Optimizer step

Changes the parameters using those gradients.

The current project uses:

    CrossEntropyLoss
    Adam
    learning rate = 0.001
    batch size = 64
    epochs = 5

Do not blindly change these values.

Later, experiments should change one variable at a time and record the result.

---

# 11. Why Cross Entropy Works Here

For the correct class y:

    L = -log(p_y)

If the model assigns high probability to the correct answer:

    p_y → 1
    L → 0

If it assigns very low probability:

    p_y → 0
    L → large

PyTorch's CrossEntropyLoss combines the necessary log-softmax and negative-log-likelihood operations.

Important practical point:

The model should output logits, not softmax probabilities, when using CrossEntropyLoss.

That is exactly what the current model does.

---

# 12. evaluate.py

Evaluation is intentionally separate from training.

The current evaluator:

1. loads the saved .pth model
2. creates the same model architecture
3. loads its parameters
4. switches to evaluation mode
5. disables gradient tracking
6. runs the test set
7. reports average loss and accuracy

This separation matters.

A program called evaluate.py should not silently train the model.

---

# 13. export.py — The Python/C Contract

This is one of the most important files in the project.

The exporter takes PyTorch tensors and writes their raw float32 bytes into:

    models/weights.bin

The order is explicitly defined:

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

The C loader must read exactly that order.

This is an API.

Even though it is a binary file rather than JSON or HTTP, it is still a contract between two programs.

If Python writes one order and C reads another, the program may still run while producing nonsense.

That is one of the most dangerous classes of bugs in this project.

---

# 14. Why weights.bin Is 175016 Bytes

Calculate it yourself.

### Conv1

    32 × 1 × 3 × 3 = 288 weights
    32 biases
    = 320 floats

### Conv2

    32 × 32 × 3 × 3 = 9216
    32
    = 9248 floats

Conv3 is another 9248.

Conv4 is another 9248.

### Linear

    10 × 1568 = 15680
    10 biases
    = 15690 floats

Total:

    320
    + 9248
    + 9248
    + 9248
    + 15690
    = 43754 floats

Each float32 is four bytes:

    43754 × 4 = 175016 bytes

If the file size changes unexpectedly after an architecture change, investigate immediately.

---

# 15. C Tensor Representation

The core tensor is conceptually:

    typedef struct {
        float *data;
        int channels;
        int height;
        int width;
    } Tensor;

The struct stores metadata.

The actual tensor values live in a flat heap allocation.

The layout is channel-first:

    index(c, y, x)
    =
    ((c * height) + y) * width + x

This layout is not arbitrary.

It is part of the Python↔C contract.

---

# 16. Why Flattened Memory Matters

A 3D tensor does not need a 3D allocation.

Instead:

    C × H × W

elements can live in one contiguous array.

For:

    32 × 28 × 28

the number of floats is:

    25088

The pointer points to the first float.

Indexing converts a logical coordinate into a physical offset.

This is the foundation of almost every tensor operation in the C engine.

---

# 17. C Memory Ownership

For every allocation, ask:

> Who owns this memory?

Example:

    Tensor a = tensor_alloc(...);

The caller owns a.

When finished:

    tensor_free(&a);

Mental model:

    allocate
       ↓
    own
       ↓
    use
       ↓
    free

If you lose track of ownership:

- memory leaks
- double frees
- use-after-free
- invalid pointers
- corrupted tensors

This is why the C implementation is part of the ML learning experience.

---

# 18. The C Forward Pass

The current C forward pass mirrors the PyTorch model:

    input
      ↓
    conv1
      ↓
    relu
      ↓
    conv2
      ↓
    relu
      ↓
    pool
      ↓
    conv3
      ↓
    relu
      ↓
    conv4
      ↓
    relu
      ↓
    pool
      ↓
    linear
      ↓
    logits

Every temporary tensor has a lifetime.

A good implementation frees intermediate tensors as soon as they are no longer needed.

That is both a correctness and memory-management lesson.

---

# 19. The First Major Milestone: Python ↔ C Parity

This is the next serious engineering milestone.

Do not stop at:

> Both programs predict the same digit.

That is weak evidence.

Two incorrect implementations can accidentally choose the same class.

You want numerical parity.

The same input must go through:

    PyTorch
    and
    C

and their intermediate results must agree within a small tolerance.

---

# 20. What Numerical Parity Means

Suppose Python produces:

    0.481239

and C produces:

    0.481241

That is probably floating-point rounding.

But if Python produces:

    0.481239

and C produces:

    0.127891

something is wrong.

Useful measurements include:

    abs(a - b)

and:

    max(abs(a - b))

For each stage, record:

    max absolute difference
    mean absolute difference

Do not obsess over an arbitrary tolerance before understanding the source of the difference.

---

# 21. The First-Divergence Rule

Compare:

    input
    conv1
    relu1
    conv2
    relu2
    pool1
    conv3
    relu3
    conv4
    relu4
    pool2
    linear
    logits

Find the first stage that differs.

Example:

    input      PASS
    conv1      PASS
    relu1      PASS
    conv2      FAIL

Do not debug MaxPool.

The bug is almost certainly in:

- Conv2
- Conv1 output handling
- Conv2 weights/bias
- tensor layout
- memory lifetime

This rule saves enormous amounts of time.

---

# 22. What Can Cause Python/C Divergence?

Memorize these categories.

## Shape bug

Wrong output dimensions.

## Indexing bug

Wrong tensor offset.

## Weight-layout bug

Wrong interpretation of:

    [out_channels, in_channels, kernel_h, kernel_w]

## Bias bug

Bias added to the wrong output channel.

## Padding bug

Out-of-bounds kernel positions handled incorrectly.

## Pooling bug

Wrong window or stride.

## Memory bug

Freed or overwritten tensor.

## Serialization bug

Wrong layer order or wrong byte count.

## Floating-point difference

Same mathematics, slightly different accumulation order.

Your debugging process should classify the problem before changing code.

---

# 23. The Existing verify.c Tool

c/tools/verify.c already performs a useful manual layer dump.

It reports:

- tensor shape
- first few values
- final logits
- predicted digit

This is a good debugging instrument.

But it is not yet a complete automated parity system.

Your future job is to turn the concept into a repeatable comparison pipeline.

---

# 24. Build a Real Parity Test

Target workflow:

    1. Choose a deterministic MNIST image.
    2. Save exactly its float32 input.
    3. Run PyTorch.
    4. Dump intermediate tensors.
    5. Run C.
    6. Dump intermediate tensors.
    7. Compare corresponding tensors.
    8. Print max/mean error.
    9. Fail automatically if tolerance is exceeded.

Conceptual output:

    stage       max_abs_diff    status
    ----------------------------------
    input       0.000000       PASS
    conv1       0.000001       PASS
    relu1       0.000001       PASS
    conv2       0.000003       PASS
    ...
    logits      0.000012       PASS

That is much more useful than manually reading two text files.

---

# 25. Testing — The Repository Needs This

The current tests directory is not a finished test framework.

Build it.

Start small.

## Tensor allocation

Verify:

- dimensions are stored correctly
- expected number of elements is allocated
- initialization matches the documented contract

## Tensor indexing

Set one element at:

    (c, y, x)

Read it back.

Then verify neighboring elements were not modified.

## ReLU

Input:

    [-3, -1, 0, 2, 5]

Expected:

    [0, 0, 0, 2, 5]

## Argmax

Input:

    [1, 7, 3, 2]

Expected:

    1

Also test ties.

## Linear

Use tiny hand-computable matrices.

## Conv2D

Start with a tiny one-channel input and tiny kernel.

Calculate the expected answer manually.

## MaxPool

Use:

    [1 7
     3 2]

Expected:

    7

Then test negative values.

This is important because a bad implementation may accidentally assume zero is a valid initial maximum.

---

# 26. CMake

The repository currently builds the Raylib application through CMake.

Important concepts:

- project declaration
- C standard
- compiler warnings
- include directories
- executable sources
- linking libraries

Current application sources:

    c/src/main.c
    c/src/nn.c
    c/src/ui.c

Current public headers:

    c/include/nn.h
    c/include/ui.h

Later, add separate CMake targets for:

- unit tests
- verification tool
- benchmark executable

The goal is to build each independently.

---

# 27. Compiler Warnings

Do not disable warnings just to make the project compile.

The project already uses:

    -Wall
    -Wextra
    -Wpedantic

Treat warnings as useful information.

Later, experiment with stricter warnings gradually.

The point is to learn what each warning means, not to collect flags.

---

# 28. Sanitizers

After normal tests work, introduce:

- AddressSanitizer
- UndefinedBehaviorSanitizer

They can reveal:

- out-of-bounds access
- use-after-free
- invalid pointer use
- double free
- some forms of undefined behavior

Important distinction:

A sanitizer can tell you that your program accesses invalid memory.

It cannot tell you that your convolution mathematics are wrong.

Therefore:

    sanitizers
    +
    unit tests
    +
    numerical parity

are all necessary.

---

# 29. Runtime and Path Robustness

The current application and verification tools use relative model/input paths.

That is fine during development, but relative paths depend on the current working directory.

Your next job is to make runtime behavior explicit.

A good application should:

- clearly report which model path it is trying
- fail clearly when the model is missing
- reject an invalid model
- avoid silently using a missing/zero input as if it were real data
- document the expected working directory until a more robust path strategy exists

This is a systems-engineering issue, not an ML issue.

---

# 30. Preprocessing Is Part of the Model

The CNN does not see your drawing.

It sees a tensor.

Therefore:

    user drawing
          ↓
    canvas pixels
          ↓
    bounding box
          ↓
    crop
          ↓
    resize
          ↓
    center
          ↓
    28×28 tensor
          ↓
    CNN

If preprocessing produces something unlike MNIST, the model may perform badly even if the CNN is excellent.

This is not merely a UI problem.

It is an ML pipeline problem.

---

# 31. Current C Preprocessing

The current UI:

1. finds the non-empty bounding box
2. computes a square crop
3. adds a small margin
4. resizes the digit to 20×20
5. places it in the center of a 28×28 canvas
6. uses bilinear sampling

This is already more meaningful than blindly shrinking the entire 280×280 canvas.

But it is not automatically correct.

You need to measure it.

---

# 32. Why Centering Matters

MNIST images are not arbitrary screenshots of a drawing canvas.

A digit that occupies:

    x = 5..45
    y = 200..240

is very different from a centered digit.

A CNN can learn some translation tolerance, but that does not mean unlimited translation invariance.

Preprocessing reduces unnecessary variation.

The correct question is not:

> Does this preprocessing look reasonable?

The correct question is:

> Does this preprocessing make the input distribution more similar to the data used to train the model?

---

# 33. Your Next ML Experiment

Collect a small personal test set.

For example:

    10 examples of each digit
    = 100 drawings

Do not train on them initially.

Use them only for evaluation.

Record:

    digit
    prediction
    confidence
    correct/incorrect

Then calculate:

    overall accuracy
    per-digit accuracy
    confusion matrix

This gives you better information than testing a few drawings manually.

---

# 34. Confusion Matrix

A confusion matrix answers:

> When the true digit is X, what does the model think it is?

For example:

    true 0 → mostly 0
    true 1 → mostly 1
    true 7 → sometimes 1
    true 9 → sometimes 4

This tells you where the model fails.

Do not immediately change the architecture.

First inspect the failed images.

---

# 35. Error Analysis

Every incorrect prediction is useful.

For a failed example, record:

    input image
    true label
    predicted label
    confidence
    preprocessed image
    logits

Then ask:

- Is the digit ambiguous?
- Is it too small?
- Is it off-center?
- Is the stroke too thick?
- Is it inverted?
- Is the crop wrong?
- Is the model genuinely confused?

This turns “the model is bad” into a concrete engineering question.

---

# 36. Confidence Is Not Correctness

A model can be:

    99.8% confident

and still be wrong.

Softmax confidence tells you how strongly the model prefers one class over the others.

It does not prove that the probability is statistically calibrated.

Later, study:

- calibration
- reliability diagrams
- expected calibration error
- temperature scaling

Do not treat confidence as a truth meter.

---

# 37. Experiments — Change One Thing

Bad experiment:

    change architecture
    change optimizer
    change learning rate
    add augmentation
    change preprocessing
    train for longer

Then accuracy changes.

You learn almost nothing about why.

Better:

    baseline
      ↓
    change exactly one variable
      ↓
    train
      ↓
    evaluate
      ↓
    record

Keep an experiment table:

    ID | change | epochs | test accuracy | notes

Your goal is not merely a higher number.

Your goal is understanding causality.

---

# 38. Useful Future ML Experiments

After parity and evaluation are solid:

## Data augmentation

Try:

- small rotations
- translations
- scaling
- small affine changes

Measure whether personal handwriting improves.

## Normalization

Compare the current ToTensor pipeline against carefully chosen normalization.

Do not assume normalization helps.

Measure it.

## Architecture

Experiment with:

- fewer channels
- more channels
- fewer layers
- dropout
- batch normalization

Record parameter count and accuracy.

## Optimizer

Compare:

- SGD
- Adam

Keep the comparison fair.

## Learning rate

Try a small controlled sweep.

---

# 39. Parameter Count

Learn to calculate trainable parameters.

For a convolution:

    out_channels × in_channels × kernel_h × kernel_w
    +
    out_channels

For the first convolution:

    32 × 1 × 3 × 3 + 32
    = 320

For the final linear layer:

    10 × 1568 + 10
    = 15690

Parameter count helps you understand model size, memory, and eventually performance.

---

# 40. Computational Cost

A convolution performs many multiply-add operations.

Roughly:

    output_h × output_w
    × output_channels
    × input_channels
    × kernel_h
    × kernel_w

This is why convolution is a natural target for later profiling.

But:

> Do not optimize it until you measure it.

---

# 41. Benchmarking

The repository already contains benchmark tooling.

Use it to answer:

- How long does PyTorch inference take?
- How long does C inference take?
- Which C layer is expensive?
- Does an optimization actually help?

Measure multiple runs.

Do not trust one timing.

When appropriate, report:

    minimum
    mean
    median
    standard deviation

Warm-up can matter in some environments.

---

# 42. Optimization Roadmap

Only after correctness is established.

## Level 1 — obvious inefficiencies

- unnecessary allocations
- repeated conversions
- avoidable file I/O
- unnecessary copies

## Level 2 — memory reuse

Instead of allocating a new tensor for every layer, investigate reusable buffers.

Question:

> Which tensor lifetimes overlap?

If two buffers are never needed at the same time, they may share memory.

## Level 3 — cache behavior

Loop ordering affects memory locality.

Mathematically identical loops can have very different performance.

## Level 4 — SIMD

Only after measuring.

## Level 5 — quantization

Eventually experiment with lower precision and measure the accuracy tradeoff.

---

# 43. Model Format Improvements

The current weights.bin format is intentionally simple.

It is essentially raw float32 values.

That is useful for learning.

But it has weaknesses:

- no magic number
- no version
- no architecture metadata
- no dtype declaration
- no layer count
- no checksum
- no explicit tensor shapes

A future model format could contain:

    HEADER
    ├── magic
    ├── version
    ├── dtype
    ├── layer count
    └── metadata

    DATA
    ├── tensor 1
    ├── tensor 2
    └── ...

Then the C loader can reject incompatible models cleanly.

This is where ML becomes systems engineering.

---

# 44. Multi-Digit Recognition

Once single-digit recognition is reliable, the next computer-vision problem is:

> How do we recognize 427 rather than only 4?

The easiest first approach is segmentation:

    image
      ↓
    find connected regions
      ↓
    separate digits
      ↓
    crop each digit
      ↓
    resize each to 28×28
      ↓
    CNN
      ↓
    digit predictions
      ↓
    combine

For example:

    [4] [2] [7]
     ↓   ↓   ↓
     4   2   7

    => 427

---

# 45. Connected Components

A connected-component algorithm groups pixels that touch.

Basic idea:

1. choose a foreground pixel
2. flood-fill connected foreground pixels
3. record the bounding box
4. repeat for unvisited foreground pixels

This teaches:

- graphs
- queues/stacks
- recursion vs iteration
- visited arrays
- bounding boxes
- image processing

Implement it in C after single-digit evaluation is stable.

---

# 46. Why Segmentation Will Eventually Fail

Imagine two digits touching:

    12

There may be no empty column between them.

Simple projection-based segmentation may see one object instead of two.

This leads to increasingly difficult approaches:

- connected components
- projection profiles
- contour analysis
- sliding windows
- learned detection
- sequence models

Do not jump to CTC immediately.

Build the simpler systems first.

---

# 47. Variable-Length OCR

Single-digit classification has a fixed output space:

    {0, ..., 9}

OCR is different.

The output might be:

    7
    42
    128
    123456789

The number of output symbols is variable.

This changes the problem from:

    image → one class

to:

    image → sequence

That is a fundamentally different modeling problem.

---

# 48. CTC — Later, Not Now

Connectionist Temporal Classification is useful when the alignment between input positions and target characters is unknown.

Conceptually:

    image
      ↓
    feature sequence
      ↓
    per-position character probabilities
      ↓
    CTC
      ↓
    decoded sequence

You do not need to implement CTC now.

First understand:

- sequence models
- logits over time
- blank tokens
- collapsing repeated symbols
- dynamic programming
- forward/backward probabilities

CTC is a later milestone.

---

# 49. Optional Keystone: Implement Backprop Yourself

This is optional and should come after the C inference engine is solid.

Start with:

    y = Wx + b

Derive:

    ∂L/∂W
    ∂L/∂b
    ∂L/∂x

Then:

1. linear
2. ReLU
3. softmax/cross-entropy
4. maxpool
5. Conv2D
6. complete tiny CNN

Do not start with a large MNIST CNN.

Start with a tiny network where every number can be checked by hand.

---

# 50. Gradient Checking

A numerical gradient can be approximated with finite differences:

    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

Compare this with your analytical gradient.

If they agree within an appropriate tolerance, your derivative is probably correct.

This is one of the best ways to understand backpropagation.

---

# 51. What to Put in Obsidian

Do not copy the entire guide into notes.

Use Obsidian for things you personally understand.

For each important concept, write:

### Concept

What is it?

### Equation

What is the mathematical definition?

### Example

One small numeric example.

### Number Guesser

Where is it used?

### Bug

What happens if it is wrong?

### Test

How would you prove it works?

Example:

    ## Conv2D

    ### Equation
    ...

    ### Shape
    ...

    ### Number Guesser
    c/src/nn.c

    ### Common bug
    Weight index order

    ### Test
    1-channel 3×3 input + hand-computed kernel

This produces durable knowledge rather than copied documentation.

---

# 52. Git Workflow for Learning

Make commits around meaningful milestones.

Good commit messages:

    add tensor unit tests
    add python c parity comparison
    fix conv2d padding
    add preprocessing evaluation
    add personal handwriting evaluator

Avoid:

    stuff
    changes
    fix
    final
    final2

A commit should explain what changed.

Before risky work:

    git status
    git branch
    git log --oneline

Create a backup branch before major architectural experiments.

---

# 53. Debugging Protocol

When something breaks:

## Step 1

Reproduce it.

## Step 2

Make the smallest input that demonstrates the bug.

## Step 3

Identify the first incorrect stage.

## Step 4

Check:

- shape
- values
- indexing
- ownership
- file bytes
- assumptions

## Step 5

Change one thing.

## Step 6

Re-run the smallest test.

## Step 7

Add a regression test if the bug was real.

That last step matters.

A bug fixed without a test can return later.

---

# 54. Do Not Debug by Guessing

Avoid:

    Maybe Conv2D is wrong, I'll rewrite it.

Instead:

    input       PASS
    conv1       PASS
    relu1       PASS
    conv2       FAIL

Now you have evidence.

Good engineering is disciplined narrowing of the search space.

---

# 55. Current File Map

## Python

    python/dataset.py
        MNIST loading

    python/model.py
        CNN architecture

    python/train.py
        training

    python/evaluate.py
        evaluation

    python/export.py
        PyTorch → weights.bin

    python/helper_functions.py
        general learning utilities

## C

    c/include/nn.h
        tensor/model API

    c/src/nn.c
        neural-network implementation

    c/include/ui.h
        application state/API

    c/src/ui.c
        drawing + preprocessing

    c/src/main.c
        Raylib application

    c/tools/verify.c
        manual C inference inspection

## Benchmark

    benchmark/model.py
    benchmark/run_pytorch.py
    benchmark/c_benchmark.c
    benchmark/compare.py
    benchmark/Makefile
    benchmark/README.md

## Build

    CMakeLists.txt

---

# 56. Definition of Done for Each Milestone

## Milestone 1 — Understand current model

Done when you can explain:

- every layer
- every tensor shape
- why there are 10 outputs
- why flatten size is 1568
- what each important C loop does

without reading the source.

## Milestone 2 — Python training

Done when you can explain:

- batch
- epoch
- logits
- loss
- gradient
- optimizer
- evaluation

and run training intentionally.

## Milestone 3 — C inference

Done when you can explain:

- Tensor
- flat indexing
- ownership
- Conv2D
- MaxPool
- linear
- model loading

without treating the code as magic.

## Milestone 4 — Numerical parity

Done when:

- the same input reaches both systems
- every layer is compared
- differences are measured
- the comparison can fail automatically
- you understand first-divergence debugging

## Milestone 5 — Testing

Done when core C operations have automated tests.

## Milestone 6 — Real handwriting

Done when you have a held-out personal dataset and measured per-digit performance.

## Milestone 7 — Optimization

Done when you have before/after benchmarks and can explain why an optimization helps.

## Milestone 8 — Multi-digit recognition

Done when the system can segment and recognize a controlled multi-digit dataset with measured accuracy.

## Milestone 9 — OCR

Done when you understand why variable-length recognition needs a different architecture and can explain CTC mathematically.

---

# 57. Questions You Should Be Able to Answer

## C

- What is a pointer?
- What is the difference between an array and a pointer?
- What is heap memory?
- What is stack memory?
- Who owns a Tensor?
- What is a dangling pointer?
- What is a memory leak?
- Why is tensor data contiguous?

## Tensor math

- What does [C,H,W] mean?
- How does (c,y,x) become one integer offset?
- Why is channel-first important?
- Why does padding preserve the spatial size here?
- Why does pooling halve the image?
- Why is flatten size 1568?

## CNN

- What does a convolution actually calculate?
- Why are there multiple output channels?
- What does a kernel learn?
- What does ReLU do?
- What does MaxPool do?
- Why does the final layer have 10 outputs?

## Training

- What is a loss?
- What is a gradient?
- Why do we subtract the gradient?
- What does learning rate control?
- What is an optimizer?
- Why can training accuracy increase while generalization gets worse?

## Systems

- Why export weights instead of loading PyTorch from C?
- Why does the binary format need a contract?
- Why can two correct floating-point implementations differ?
- Why is parity stronger than matching one prediction?
- Why should you profile before optimizing?

## Computer vision

- Why is preprocessing part of the ML pipeline?
- Why does centering matter?
- Why can connected components separate digits?
- Why can touching digits break segmentation?

If you cannot answer one, stop and study that concept.

---

# 58. A Better Daily Workflow

For each session:

    1. Pick ONE milestone.
    2. Read the relevant section.
    3. Inspect the current implementation.
    4. Predict what the code should do.
    5. Run it.
    6. Compare prediction vs reality.
    7. Make one change or implement one small test.
    8. Verify.
    9. Write an Obsidian note.
    10. Commit.

Do not measure progress by lines of code.

Measure progress by questions you can now answer.

---

# 59. Project Philosophy

The project should follow:

    THEORY
      ↓
    MATH
      ↓
    IMPLEMENTATION
      ↓
    EXPERIMENT
      ↓
    MEASUREMENT
      ↓
    DEBUGGING
      ↓
    EXPLANATION
      ↓
    TEST
      ↓
    COMMIT

Every important concept should eventually travel through this loop.

For example:

    learn convolution
      ↓
    derive output shape
      ↓
    inspect PyTorch
      ↓
    inspect C loops
      ↓
    hand-calculate tiny example
      ↓
    unit test
      ↓
    compare against PyTorch
      ↓
    profile

That is much deeper than simply knowing the definition of Conv2D.

---

# 60. Immediate Roadmap

## NEXT 1 — Make Python/C parity automatic

Use the existing benchmark and verification code.

Create a deterministic input.

Dump the same stages from both implementations.

Automatically compare them.

Do not change the CNN yet.

## NEXT 2 — Build C unit tests

Start with:

    tensor_alloc/free
    tensor_get/set
    relu
    argmax
    linear
    conv2d
    maxpool2d

Use tiny hand-computed cases.

## NEXT 3 — Fix the runtime/build contract

Make sure model paths are robust and do not accidentally depend on the current working directory.

The application should clearly report:

- model found
- model missing
- model invalid

Do not silently continue with a broken model.

## NEXT 4 — Add sanitizers

Run C tests and verification tools under memory/undefined-behavior sanitizers.

Fix every real issue.

## NEXT 5 — Build a personal handwriting evaluator

Collect held-out examples.

Measure:

- total accuracy
- per-digit accuracy
- confusion matrix
- confidence
- failure examples

## NEXT 6 — Improve preprocessing scientifically

Compare preprocessing versions on the same held-out dataset.

Record the results.

## NEXT 7 — Experiment with the model

Only after the evaluation pipeline is reliable.

Change one variable at a time.

## NEXT 8 — Profile C

Measure layer timings.

Find the actual bottleneck.

Then optimize.

## NEXT 9 — Multi-digit recognition

Start with controlled segmentation.

## NEXT 10 — OCR

Study sequence modeling and CTC.

---

# 61. What NOT To Do Yet

Do not:

- add more convolution layers just because you can
- implement C backpropagation immediately
- optimize with SIMD before profiling
- build CTC before understanding single-digit evaluation
- change five training variables simultaneously
- call the model good because one drawing worked
- call Python and C equivalent because they predict the same digit
- add complicated abstractions before understanding the current code
- copy large code blocks without being able to explain them
- turn the guide into a list of commands you blindly execute

The project is a learning system.

Complexity should be earned.

---

# 62. Final Standard

The standard is not:

> It works on my machine.

The standard is:

> I understand what it computes, I can measure it, I can reproduce it, and I can prove when it is correct.

For every important subsystem, aim to have:

    implementation
    +
    small test
    +
    real test
    +
    measurement
    +
    documentation

When those five exist, you are not merely building a project.

You are learning how the project works.

---

# Appendix A — One-Page Architecture

    MNIST
      │
      ▼
    python/dataset.py
      │
      ▼
    python/model.py
      │
      ▼
    python/train.py
      │
      ▼
    number_guesser_model.pth
      │
      ▼
    python/export.py
      │
      ▼
    weights.bin
      │
      ▼
    c/src/nn.c
      │
      ├───────────────┐
      │               │
      ▼               ▼
    c/tools/verify.c  c/src/main.c
                          │
                          ▼
                      c/src/ui.c
                          │
                          ▼
                    280×280 drawing
                          │
                          ▼
                    preprocessing
                          │
                          ▼
                        28×28
                          │
                          ▼
                    CNN inference
                          │
                          ▼
                    logits → softmax
                          │
                          ▼
                      prediction

---

# Appendix B — Shape Cheat Sheet

    Input:    1 × 28 × 28
    Conv1:   32 × 28 × 28
    ReLU:    32 × 28 × 28
    Conv2:   32 × 28 × 28
    ReLU:    32 × 28 × 28
    Pool1:   32 × 14 × 14
    Conv3:   32 × 14 × 14
    ReLU:    32 × 14 × 14
    Conv4:   32 × 14 × 14
    ReLU:    32 × 14 × 14
    Pool2:   32 × 7 × 7
    Flatten: 1568
    Linear:  10
    Logits:  10

---

# Appendix C — Core Equations

### Linear

    y = Wx + b

### ReLU

    ReLU(x) = max(0, x)

### Conv2D output size

    floor((N + 2P - K) / S) + 1

### MaxPool output size

    floor((N - K) / S) + 1

### Softmax

    p_i = exp(z_i) / Σ exp(z_j)

Numerically stable:

    p_i = exp(z_i - max(z)) / Σ exp(z_j - max(z))

### Cross entropy

    L = -log(p_y)

### Tensor offset

    offset(c,y,x) = ((c × H) + y) × W + x

### Central finite difference

    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

---

# Appendix D — The Rules

1. Understand before extending.
2. Find the first divergence.
3. Test tiny cases before real cases.
4. Measure before optimizing.
5. Change one experimental variable at a time.
6. Treat preprocessing as part of the model.
7. Treat the binary model format as an API.
8. Use sanitizers for memory correctness.
9. Use numerical parity for implementation correctness.
10. Use real held-out data for ML evaluation.
11. Write down what you personally learned.
12. Do not claim a milestone is complete without evidence.

---

# Appendix E — Your Current Mission

If you open this file today and ask:

> What should I work on?

the answer is:

    UNDERSTAND
        ↓
    Python model
        ↓
    C model
        ↓
    tensor layout
        ↓
    weights.bin contract

    THEN BUILD
        ↓
    automatic Python/C parity
        ↓
    C unit tests
        ↓
    sanitizer workflow

    THEN MEASURE
        ↓
    personal handwriting dataset
        ↓
    confusion matrix
        ↓
    preprocessing experiments

    THEN ADVANCE
        ↓
    optimization
        ↓
    multi-digit recognition
        ↓
    OCR
        ↓
    CTC

    OPTIONAL DEEP DIVE
        ↓
    backpropagation from scratch

Do not rush to the end.

The point of Number Guesser is not to eventually say “I made a digit recognizer.”

The point is to reach the point where you can explain why every number, loop, tensor, pointer, gradient, byte, and prediction exists.
