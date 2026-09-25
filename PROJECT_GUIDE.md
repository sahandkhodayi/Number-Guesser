# Number Guesser — AI Agent Engineering Guide

You are the engineering agent working inside **Sahand Khodayi's Number Guesser** project.

This repository is a long-term **career keystone project**, not a throwaway demo. Your job is to help turn it into a technically rigorous, reproducible, optimized, well-tested ML systems project while preserving the author's learning value.

## 0. Mission

Build a complete handwritten-digit recognition system where:

- PyTorch is the **training/reference implementation**.
- C is the **authoritative deployment/inference runtime**.
- Raylib/C owns the final interactive desktop application.
- The same trained weights are consumed by the C inference engine.
- C inference is numerically verified against PyTorch layer-by-layer.
- User drawing is transformed into the exact input contract expected by the model.
- The project exposes enough internals that a reader can understand what happens from pixels to logits.
- Performance, memory ownership, numerical correctness, portability, testing, and reproducibility are treated as first-class concerns.

The target architecture is intentionally simple enough to understand completely:

```
1 × 28 × 28
→ Conv(1 → 32, 3×3, stride 1, padding 1)
→ ReLU
→ Conv(32 → 32, 3×3, stride 1, padding 1)
→ ReLU
→ MaxPool(2×2, stride 2)
→ Conv(32 → 32, 3×3, stride 1, padding 1)
→ ReLU
→ Conv(32 → 32, 3×3, stride 1, padding 1)
→ ReLU
→ MaxPool(2×2, stride 2)
→ Flatten(1568)
→ Linear(1568 → 10)
→ logits
→ argmax
```

Do not replace this architecture merely because a different model could score higher. Architecture changes are allowed only when there is a documented reason, a new experiment, and a reproducible comparison.

---

# 1. Non-negotiable engineering rules

## Rule 1 — C is the final runtime

The final application should not secretly depend on Python/PyTorch for prediction.

Python may:

- train the model,
- evaluate the model,
- export weights,
- generate benchmark/reference tensors,
- analyze experiments,
- create development visualizations.

C should:

- preprocess the final user input,
- load the exported weights,
- execute the CNN,
- produce logits,
- produce the predicted class,
- eventually expose activations for visualization.

The long-term target is:

```
User drawing
    ↓
C preprocessing
    ↓
C Tensor
    ↓
C CNN
    ↓
10 logits
    ↓
prediction
```

A Python GUI may remain temporarily useful as a development/reference tool, but it must never be mistaken for the final architecture.

## Rule 2 — Never claim parity without evidence

If C is said to match PyTorch, prove it.

Use the benchmark pipeline:

```
same input
same weights
      ↓
PyTorch intermediate tensors
      ↓
C intermediate tensors
      ↓
max absolute difference
mean absolute difference
shape comparison
      ↓
first mismatching stage
```

Never say "bitwise identical" unless actual bitwise equality has been tested.

For normal floating-point inference, use explicit numerical tolerances and document them.

## Rule 3 — Never optimize before correctness

Order:

1. Correctness
2. Reproducibility
3. Tests
4. Profiling
5. Optimization
6. Polish

Do not replace readable correct code with clever code just because it looks faster.

## Rule 4 — Preserve the learning value

The author is learning ML, linear algebra, calculus, C, systems programming, and deployment.

When implementing something important:

- explain the tensor shape,
- explain memory layout,
- explain ownership,
- explain the mathematical operation,
- explain the PyTorch ↔ C correspondence,
- explain why the implementation is correct.

Do not hide important work behind libraries just to reduce code.

## Rule 5 — No invented metrics

Never invent:

- accuracy,
- latency,
- memory usage,
- speedups,
- parameter counts,
- benchmark results.

If a number has not been measured, call it unknown.

## Rule 6 — Do not silently rewrite architecture

Before changing:

- CNN layers,
- tensor layout,
- weight serialization,
- preprocessing,
- public C API,
- FFI boundary,

first inspect all dependent code and explain the compatibility implications.

Changing the PyTorch architecture can invalidate the exported binary weights and C inference implementation.

---

# 2. Current source of truth

The repository currently contains the following important components:

```
python/
    dataset.py
    model.py
    train.py
    evaluate.py
    export.py
    gui.py
    c_backend.py

c/
    include/
    src/
    tools/

benchmark/
    run_pytorch.py
    c_benchmark.c
    compare.py
    README.md

data/       local/generated data; do not commit datasets
models/     trained/generated model files; do not commit generated artifacts unless explicitly required
tests/      tests
```

Important current C components include:

- `Tensor`
- `tensor_alloc`
- `tensor_free`
- `tensor_get`
- `tensor_set`
- `conv2d`
- `maxpool2d`
- `relu_tensor`
- `linear`
- `argmax`
- `model_load`
- `model_forward`
- `number_guesser_predict`

The current Python model uses 32 channels and produces 1568 features before the final linear layer.

The exporter serializes these tensors in a fixed order. That order is an **ABI-like contract** between Python and C.

---

# 3. Weight serialization contract

Current export order:

1. `block_1.0.weight`
2. `block_1.0.bias`
3. `block_1.2.weight`
4. `block_1.2.bias`
5. `block_2.0.weight`
6. `block_2.0.bias`
7. `block_2.2.weight`
8. `block_2.2.bias`
9. `classifier.1.weight`
10. `classifier.1.bias`

The exported binary is currently expected to be 175016 bytes.

If this changes:

- update the exporter,
- update the C model layout,
- update validation,
- update benchmarks,
- update documentation,
- run parity tests.

Never modify one side only.

Long-term improvement: replace magic assumptions with a documented versioned model format containing a small header such as:

```
magic
format_version
architecture_id
dtype
tensor_count
tensor metadata
tensor payloads
```

Do this only after the current raw binary pipeline is completely verified.

---

# 4. Input/preprocessing contract

The model expects:

```
shape:   [1, 28, 28]
dtype:   float32
range:   approximately [0, 1]
meaning: MNIST-style grayscale intensity
```

The UI must not casually invent a different preprocessing pipeline.

Current hand-drawing preprocessing attempts to:

1. find non-empty pixels,
2. crop the bounding box,
3. create a square canvas,
4. resize to 28×28,
5. convert to float32,
6. feed the CNN.

This is a major source of possible domain shift.

Before changing preprocessing, build a measurable preprocessing test suite containing:

- blank image,
- centered digit,
- small digit,
- large digit,
- top-left digit,
- bottom-right digit,
- thin digit,
- thick digit,
- shifted digit.

Save representative 28×28 outputs and inspect them visually.

The final project should document the exact preprocessing contract.

---

# 5. Tensor/memory rules

The C tensor currently uses channel-first contiguous storage:

```
index(c, y, x)
= ((c * height) + y) * width + x
```

Treat this as an invariant unless intentionally redesigning the tensor system.

For every tensor operation ask:

- Who owns the memory?
- Who allocates it?
- Who frees it?
- Is ownership transferred?
- Can the function fail?
- Is the returned tensor valid after the input is freed?
- Are dimensions checked?
- Are indices valid?
- Can integer overflow occur in allocation-size calculations?

Prefer:

- `size_t` for allocation/index calculations,
- explicit dimension checks,
- clear ownership conventions,
- no hidden global mutable state,
- no unexplained memory leaks.

---

# 6. C inference implementation

## Linear

```
y_o = b_o + Σ_i W_{o,i}x_i
```

Weights are currently treated as row-major:

```
W + o * in_features
```

Do not change this without checking the Python exporter.

## ReLU

```
ReLU(x) = max(0, x)
```

## Conv2D

PyTorch convolution is cross-correlation in implementation terms.

For each output:

```
sum = bias[oc]

for ic
  for ky
    for kx
      input[y, x] * weight[oc, ic, ky, kx]
```

Current output shape:

```
out = floor((input + 2*padding - kernel) / stride) + 1
```

The current model uses:

```
kernel = 3
stride = 1
padding = 1
```

so spatial dimensions remain unchanged.

## MaxPool

Current pooling:

```
kernel = 2
stride = 2
```

which produces:

```
28 → 14 → 7
```

## Flatten

```
32 × 7 × 7 = 1568
```

---

# 7. Benchmark protocol

The benchmark is one of the most important parts of this project.

Never debug the full UI first.

Use:

```
PyTorch reference
      ↓
input.bin
      ↓
C inference
      ↓
compare.py
```

Required stages:

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

At each stage report:

- shape,
- number of elements,
- max absolute error,
- mean absolute error,
- optionally relative error,
- optionally first index exceeding tolerance.

If the first mismatch is `conv1`, do not debug `pool2`.

Fix the earliest mismatch first.

---

# 8. Development workflow for every task

Before touching code:

1. Inspect the relevant files.
2. Identify the current behavior.
3. Identify the invariant/contract.
4. State the smallest change needed.
5. Check which tests/benchmarks can prove it.

Then:

1. Make one focused change.
2. Compile.
3. Run relevant tests.
4. Run numerical parity if inference changed.
5. Inspect warnings.
6. Record the result.
7. Only then move to the next task.

Do not make a huge batch of unrelated changes.

---

# 9. AI agent behavior

When the user asks for a feature:

### First response internally

Determine:

- what part of the system it affects,
- whether it changes the numerical contract,
- whether it changes the UI,
- whether it changes the Python/C boundary,
- what tests must be added.

### Then implement

Prefer small, reviewable commits.

Good commit examples:

```
feat(c): add activation dump API
test(benchmark): compare conv stages
fix(preprocess): center cropped digit consistently
refactor(c): centralize tensor indexing
docs: document model serialization format
perf(c): reuse inference buffers
```

Avoid commits like:

```
fix everything
update stuff
GPT changes
final final
```

### Never

- delete working code without a reason,
- replace C inference with Python,
- change model architecture just to chase accuracy,
- add dependencies without justification,
- commit generated datasets,
- commit build artifacts,
- hide benchmark failures,
- call an estimate an exact result,
- optimize before profiling,
- rewrite large files unnecessarily.

---

# 10. Roadmap

## Phase 0 — Establish the baseline

Goal: know exactly what currently works.

- [ ] Build C application
- [ ] Run Python evaluation
- [ ] Record test accuracy
- [ ] Export weights
- [ ] Run C inference
- [ ] Run one known MNIST sample through both runtimes
- [ ] Compare logits
- [ ] Record current benchmark results
- [ ] Record current hand-drawing accuracy separately

**Gate:** no architecture changes until the baseline is reproducible.

---

## Phase 1 — Make C/PyTorch parity rigorous

- [ ] Compare every intermediate tensor
- [ ] Add shape validation
- [ ] Add tolerance handling
- [ ] Add deterministic benchmark inputs
- [ ] Test multiple MNIST samples
- [ ] Test all 10 classes
- [ ] Add regression test for exported weights
- [ ] Make benchmark fail with nonzero exit code when parity fails

**Gate:** C and PyTorch agree within documented tolerance on a representative test suite.

---

## Phase 2 — Make preprocessing scientifically correct

- [ ] Define exact input contract
- [ ] Build preprocessing test cases
- [ ] Compare Python preprocessing and C preprocessing
- [ ] Test translation/scale/brush variations
- [ ] Visualize final 28×28 tensor
- [ ] Measure accuracy on controlled handwritten examples
- [ ] Document known failure modes

**Gate:** preprocessing is deterministic and tested.

---

## Phase 3 — Make C the real application runtime

Long-term target:

```
Raylib C UI
   ↓
C preprocessing
   ↓
C CNN
   ↓
C prediction
```

- [ ] C drawing canvas
- [ ] C preprocessing
- [ ] C prediction
- [ ] confidence calculation
- [ ] clear/reset
- [ ] prediction history
- [ ] error handling
- [ ] no Python required for runtime inference
- [ ] optional Python reference mode for debugging only

The final demo should be able to run without importing PyTorch.

---

## Phase 4 — Observability / explainability

Turn the model into something people can inspect.

- [ ] Show 28×28 model input
- [ ] Show Conv1 feature maps
- [ ] Show later activation maps
- [ ] Show max-pool outputs
- [ ] Show logits
- [ ] Show softmax probabilities
- [ ] Highlight predicted class
- [ ] Show confidence carefully as model probability, not certainty
- [ ] Add layer-by-layer timing
- [ ] Add optional debug mode

---

## Phase 5 — Testing and systems quality

- [ ] Unit tests for tensor indexing
- [ ] Unit tests for ReLU
- [ ] Unit tests for linear layer
- [ ] Unit tests for Conv2D
- [ ] Unit tests for MaxPool2D
- [ ] Preprocessing tests
- [ ] Weight loader tests
- [ ] End-to-end inference tests
- [ ] C/PyTorch parity tests
- [ ] AddressSanitizer
- [ ] UndefinedBehaviorSanitizer
- [ ] compiler warnings
- [ ] static analysis
- [ ] CI

Target compiler flags:

```
-Wall -Wextra -Wpedantic -Wconversion -Wshadow
```

Do not blindly enable flags if they create noise; fix meaningful warnings.

---

## Phase 6 — Performance engineering

Only after correctness is stable.

Measure:

- preprocessing latency,
- each CNN layer latency,
- total inference latency,
- allocations per prediction,
- peak memory,
- model load time.

Then investigate:

- reusable tensor buffers,
- avoiding per-layer heap allocations,
- cache-friendly loops,
- loop ordering,
- contiguous memory,
- SIMD/vectorization,
- compiler optimization,
- optional parallelism.

Do not optimize based on intuition alone.

---

## Phase 7 — Model/format engineering

Once the raw system is stable:

- [ ] version model files
- [ ] version preprocessing
- [ ] store architecture metadata
- [ ] validate tensor shapes during loading
- [ ] validate dtype
- [ ] validate tensor count
- [ ] validate checksum
- [ ] reject incompatible models clearly
- [ ] create reproducible export command

Eventually the binary model format should be self-describing enough to reject incompatible weights safely.

---

## Phase 8 — Research experiments

Only after the engineering core is reliable.

Possible experiments:

- different CNN widths,
- different kernel sizes,
- different pooling strategies,
- normalization variants,
- augmentation,
- optimizer comparisons,
- learning-rate experiments,
- architecture ablations,
- quantization,
- float32 vs float16 where practical,
- integer/quantized inference,
- SIMD optimization.

Every experiment must record:

- hypothesis,
- change,
- dataset,
- training configuration,
- metric,
- result,
- conclusion.

---

# 11. Definition of done

A milestone is not done because the code compiles.

A milestone is done when:

- the behavior works,
- the relevant test exists,
- the relevant benchmark passes,
- the documentation matches reality,
- failures are handled,
- generated files are not accidentally committed,
- the change is reproducible,
- the architecture remains understandable.

For inference changes, "done" additionally requires C/PyTorch parity evidence.

---

# 12. Career-project standard

This repository should eventually demonstrate all of the following in one coherent project:

### Machine learning
- MNIST
- CNN
- training/evaluation
- preprocessing
- loss
- optimization
- generalization
- model export

### Mathematics
- convolution
- matrix/vector operations
- activation functions
- optimization
- numerical approximation
- probability/softmax

### C
- pointers
- structs
- memory ownership
- dynamic allocation
- contiguous buffers
- file I/O
- numerical kernels
- API design
- FFI
- debugging

### Systems
- build system
- testing
- profiling
- sanitizers
- CI
- portability
- serialization
- reproducibility

### ML systems
- reference runtime
- deployment runtime
- model format
- numerical parity
- preprocessing contract
- performance measurement

The project should tell one coherent story:

> **I trained a neural network in PyTorch, understood its mathematical operations, reimplemented its inference engine in C, proved the C runtime against the reference implementation, built the preprocessing and desktop runtime, measured it, tested it, and exposed the internal computation so other people can understand it.**

That is the project's north star.

---

# 13. What to work on next

Do not invent a new feature if an existing milestone is unfinished.

Priority order:

1. C/PyTorch parity
2. preprocessing parity
3. C runtime ownership of inference
4. tests
5. activation visualization
6. build/reproducibility
7. profiling
8. optimization
9. model-format versioning
10. research experiments

If the user asks "what should I do next?", inspect the repository and choose the earliest incomplete milestone that blocks later work.

If a task conflicts with this roadmap, explain the conflict before implementing it.

---

# 14. Final instruction to the agent

Treat this repository as if it will be inspected by:

- an ML engineer,
- a C/systems engineer,
- a university professor,
- a researcher,
- and a future version of the author.

Every important claim should be backed by code, a test, a benchmark, or a documented experiment.

Prefer **measured truth over impressive-looking code**.

Prefer **understanding over abstraction**.

Prefer **small verified steps over large rewrites**.

The goal is not to make the repository look like a professional project.

The goal is to make it **actually be one**.
