# Prompt for the AI That Will Write the Number Guesser Project Guide

Copy the prompt below into another AI agent/research assistant. Its job is **not** to modify the repository. Its job is to study the existing Number Guesser repository and write a comprehensive Markdown **project guidebook + roadmap** that explains exactly how to continue the project.

---

## PROMPT START

You are writing the long-term technical guidebook for the repository:

**GitHub:** https://github.com/sahandkhodayi/Number-Guesser

The project is a career-keystone ML systems project by a CS/ML student.

Your output must be a Markdown file named:

`PROJECT_GUIDE.md`

Do NOT modify source code. Do NOT invent completed work. Do NOT silently redesign the project.

Your job is to inspect the repository first, understand what is actually implemented, and then produce a rigorous step-by-step guide for continuing it.

# 1. Project north star

The project should eventually demonstrate this complete chain:

```
Mathematical idea
      ↓
PyTorch training/reference implementation
      ↓
trained CNN weights
      ↓
weight export
      ↓
C tensor/runtime implementation
      ↓
C/PyTorch numerical parity
      ↓
C preprocessing
      ↓
Raylib C user interface
      ↓
prediction + probabilities
      ↓
activation/logit visualization
      ↓
tests + sanitizers + CI
      ↓
profiling + optimization
      ↓
reproducible deployment
```

The final application should use the **C/Raylib UI** as the main runtime.

Python remains important for:

- training,
- evaluation,
- model export,
- reference calculations,
- benchmark generation,
- experiments.

Python should not be required for normal application inference.

# 2. Inspect before writing

Before producing the guide, inspect:

- `README.md`
- `.gitignore`
- `python/model.py`
- `python/dataset.py`
- `python/train.py`
- `python/evaluate.py`
- `python/export.py`
- `python/gui.py`
- `python/c_backend.py`
- `c/include/nn.h`
- `c/include/ui.h`
- `c/src/nn.c`
- `c/src/ui.c`
- `c/src/main.c`
- `c/src/backend_api.c`
- `c/tools/verify.c`
- everything in `benchmark/`
- build configuration
- model/weight-loading code
- tests
- repository history when useful

Use the actual repository as the source of truth.

If documentation and code disagree, explicitly identify the disagreement instead of silently choosing one.

# 3. The guidebook must teach the project, not merely list tasks

For every major stage explain:

1. **What we are building**
2. **Why we are building it**
3. **What prerequisite knowledge is needed**
4. **The mathematics**
5. **The tensor shapes**
6. **The memory representation in C**
7. **How PyTorch implements the reference**
8. **How C should reproduce it**
9. **How to test it**
10. **How to prove it is correct**
11. **Common bugs**
12. **What success looks like**
13. **What to do next**

The guide should teach the reader enough that they can implement the system themselves rather than blindly copy code.

# 4. Required mathematics coverage

Explain the mathematics needed for the actual project.

At minimum cover:

## Linear algebra

- vectors
- matrices
- matrix multiplication
- dot products
- tensor shapes
- indexing
- flattening
- linear layers

For:

[
y = Wx+b
]

explain exactly what each dimension means.

## Convolution

Explain:

- kernel/filter
- channels
- output channels
- spatial dimensions
- stride
- padding
- cross-correlation vs mathematical convolution
- output-size formula

Use the actual project's:

- kernel = 3
- stride = 1
- padding = 1

Show why spatial dimensions remain unchanged.

## ReLU

Explain:

[
operatorname{ReLU}(x)=max(0,x)
]

and why it does not change tensor shape.

## Max pooling

Explain:

- 2×2 window
- stride 2
- selecting the maximum
- why 28×28 becomes 14×14
- why 14×14 becomes 7×7

## Flattening

Explain:

[
32	imes7	imes7=1568
]

and exactly how the contiguous C buffer corresponds to the PyTorch tensor.

## Softmax

Explain:

[
p_i=rac{e^{z_i}}{sum_j e^{z_j}}
]

and why subtracting the maximum logit before exponentiation improves numerical stability.

## Loss

Explain CrossEntropyLoss at the level needed to understand training, logits, softmax, and classification.

## Gradient descent

Explain:

- loss
- gradient
- learning rate
- parameter update
- why training and inference are different

Do not turn this into a generic math textbook. Only include mathematics that helps understand this project.

# 5. Exact CNN shape walkthrough

Show the entire shape transformation:

```
[1, 28, 28]
      ↓
[32, 28, 28]
      ↓
[32, 28, 28]
      ↓
[32, 14, 14]
      ↓
[32, 14, 14]
      ↓
[32, 14, 14]
      ↓
[32, 14, 14]
      ↓
[32, 7, 7]
      ↓
1568
      ↓
10
```

Explain every transition.

# 6. C implementation tutorial

Teach the C implementation from the bottom upward:

### Step A — Tensor structure

Explain:

```c
typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} Tensor;
```

Explain ownership.

Explain allocation.

Explain freeing.

Explain contiguous indexing:

```
((c * height) + y) * width + x
```

Explain why `size_t` matters for memory indexing/allocation.

### Step B — Linear layer

Explain:

```
y[o] = b[o] + Σ W[o,i]x[i]
```

Then connect it to the C row-major implementation.

### Step C — ReLU

### Step D — Conv2D

Walk through the nested loops line by line conceptually.

Explain the weight index:

```
(((oc * in_channels + ic) * k + ky) * k + kx)
```

### Step E — MaxPool2D

### Step F — Full forward pass

Explain memory lifetime after every layer.

### Step G — Model loading

Explain why the binary weight order is an ABI-like contract.

# 7. PyTorch ↔ C parity tutorial

This must be one of the most detailed sections.

Explain how to generate reference tensors from PyTorch:

```
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
logits
```

Then explain how C produces the same tensors.

For every stage explain:

- shape comparison
- max absolute difference
- mean absolute difference
- tolerance
- first mismatch diagnosis

Explain why comparing only the final prediction is insufficient.

Include a debugging decision tree:

```
Mismatch?
   ↓
input?
   ↓
weights?
   ↓
conv1?
   ↓
relu?
   ↓
pool?
   ↓
flatten?
   ↓
linear?
```

# 8. Preprocessing tutorial

This deserves its own chapter.

Explain the difference between:

- MNIST training distribution
- arbitrary mouse drawings

Explain domain shift.

Then explain the intended preprocessing pipeline:

```
canvas
→ foreground detection
→ bounding box
→ square crop
→ margin
→ resize
→ centering
→ 28×28
→ normalization
→ CNN
```

Explain why preprocessing can destroy otherwise correct model performance.

Explain how to create deterministic preprocessing tests.

# 9. C/Raylib UI tutorial

The main application is the C/Raylib UI.

Explain:

- event loop
- mouse input
- drawing
- brush
- canvas storage
- clear button
- predict button
- keyboard shortcuts
- rendering
- probability bars
- prediction display

Then explain how the UI connects to the CNN.

The UI should remain understandable and should not become a giant monolithic file.

Recommend a clean future separation:

```
app/
ui/
preprocess/
model/
tensor/
io/
debug/
```

Do not force this refactor prematurely; explain when it becomes justified.

# 10. Testing roadmap

Design tests for:

### Tensor

- allocation
- indexing
- set/get
- free

### Math

- ReLU
- argmax
- linear layer
- Conv2D
- MaxPool

### Preprocessing

- blank canvas
- centered digit
- shifted digit
- tiny digit
- large digit
- different stroke widths

### Model

- weight loading
- invalid model size
- forward pass

### Integration

- C/PyTorch parity
- end-to-end prediction

Explain what each test proves.

# 11. Debugging and failure modes

Include detailed explanations of likely bugs:

- wrong tensor indexing
- NHWC vs NCHW confusion
- wrong weight order
- wrong convolution indexing
- padding errors
- stride errors
- flatten order mismatch
- model file path mismatch
- wrong working directory
- float precision differences
- preprocessing mismatch
- stale model weights
- shape mismatch
- memory leaks
- double free
- use-after-free
- uninitialized memory
- integer overflow
- wrong compiler/linker flags

# 12. Performance roadmap

Do not optimize before correctness.

First measure:

- total inference time
- preprocessing time
- each layer
- allocation count
- memory usage

Then explain optimization possibilities:

- buffer reuse
- allocation reduction
- cache locality
- loop ordering
- compiler optimization
- SIMD
- parallelism
- quantization

For every optimization explain:

- why it might help,
- what it changes,
- what could break,
- how to benchmark it.

# 13. Build and run guide

Write exact commands for:

### Windows + MinGW

Include:

- prerequisites
- raylib installation
- compiler
- build
- run
- model path

### WSL / Ubuntu

Include:

- GCC
- CMake
- raylib
- build
- run

### Python

Include:

- virtual environment
- dependency installation
- dataset download
- evaluation
- export

### Verification

Show exactly how to run:

- PyTorch reference generation
- C benchmark
- comparison

If exact commands depend on a file that does not currently exist, say so and explain what needs to be added rather than inventing a command.

# 14. Project phases

Produce a staged roadmap:

## Phase 1
Baseline + reproducibility

## Phase 2
C/PyTorch parity

## Phase 3
Preprocessing

## Phase 4
C/Raylib application

## Phase 5
Activation visualization

## Phase 6
Tests + sanitizers + CI

## Phase 7
Profiling + optimization

## Phase 8
Versioned model format

## Phase 9
ML experiments

For each phase provide:

- goal
- prerequisites
- exact tasks
- math to study
- code to write
- tests to add
- benchmark to run
- definition of done

# 15. Learning roadmap

The author is learning ML and mathematics while building this.

Connect the project to:

### D2L

- tensors
- CNNs
- training
- evaluation
- generalization
- optimization

### Mathematics for Machine Learning

- linear algebra
- calculus
- probability
- optimization

For every mathematical topic explain exactly where it appears in Number Guesser.

# 16. AI-assisted development rules

The author will use AI coding agents.

Explain a safe workflow:

```
Read
→ understand
→ propose
→ implement
→ compile
→ test
→ benchmark
→ inspect diff
→ commit
```

AI agents must not:

- invent benchmark results
- claim parity without measurement
- rewrite architecture casually
- delete working code
- optimize without profiling
- hide failures
- add unnecessary dependencies

# 17. Definition of a professional milestone

A milestone is complete only when:

- implementation works
- tests exist
- benchmark passes
- documentation is updated
- build is reproducible
- no known memory errors remain
- C/PyTorch parity is demonstrated where relevant

# 18. Writing style

Write the final guidebook like a **technical textbook + lab manual + project roadmap**.

Use:

- Markdown
- diagrams in ASCII where useful
- equations
- code snippets
- tables
- checklists
- troubleshooting sections

Do NOT write generic motivational filler.

Do NOT write "learn C first" without connecting it to the exact code.

Do NOT simply summarize the repository.

The guide must tell the reader **exactly what to study, exactly what to implement, exactly how to verify it, and why each step matters.**

End with a concise "next 10 tasks" checklist based on the repository's actual current state.

## PROMPT END
