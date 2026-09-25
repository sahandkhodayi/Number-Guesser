# Number Guesser — Continuation Guidebook Prompt

You are a technical writer, ML mentor, C systems engineer, and code-reviewer.
Repository: https://github.com/sahandkhodayi/Number-Guesser

Your ONLY task is to inspect the repository and write PROJECT_GUIDE.md.
Do NOT modify source code.
Do NOT invent completed work.
Do NOT produce a beginner course from zero.

IMPORTANT: This project already has a trained CNN, exported weights, a C inference implementation, preprocessing, a Raylib/C UI, benchmark/reference tooling, model loading, and CMake. The guide must start from THIS STATE and explain what we do NEXT.

## Goal

Write a serious long-term engineering guidebook that combines:
- continuation roadmap
- implementation manual
- mathematics connected to the existing code
- testing and verification manual
- debugging manual
- performance roadmap
- ML experiment roadmap
- definitions of done

The north-star system is:

PyTorch training/reference -> exported weights -> C model loader -> C tensors/CNN -> C preprocessing -> Raylib UI -> prediction/probabilities -> numerical parity -> tests/sanitizers/CI -> profiling/optimization -> reproducible deployment.

Python is for training, evaluation, export, reference calculations, and experiments.
C/Raylib is the main application runtime.

## 1. Inspect the repository first

Read the actual current versions of:
- README.md
- CMakeLists.txt
- c/README.md
- c/include/nn.h
- c/include/ui.h
- c/src/nn.c
- c/src/ui.c
- c/src/main.c
- c/tools/verify.c
- benchmark/*
- python/model.py
- python/dataset.py
- python/train.py
- python/evaluate.py
- python/export.py
- python/helper_functions.py
- models/*
- tests/*
- .gitignore
- relevant recent Git history

Treat code as the primary source of truth. If README and code disagree, explicitly report the disagreement and document the real state.

Start PROJECT_GUIDE.md with a factual 'Current State -> Target State' table:

| Subsystem | Current state | Evidence/file | Verification status | Next action |
|---|---|---|---|---|

Do not call something complete merely because code exists.

## 2. Start from the current implementation

Do NOT teach 'what is C', 'what is a CNN', or how to write a MaxPool2D from scratch unless a short explanation is necessary to understand an existing implementation.

Example: MaxPool2D already exists. Document:
- where it is implemented
- its current algorithm
- its tensor shapes
- why -INFINITY is required for an all-negative pooling window
- how it corresponds to PyTorch MaxPool2d(2,2)
- what unit tests should verify it
- how parity should be measured
- what future optimization is possible
- then move to the next milestone.

Apply this rule to every existing subsystem.

## 3. Immediate milestone: make the existing system trustworthy

The first objective is NOT adding flashy features.

The first objective is:
BUILD -> RUN -> LOAD MODEL -> RUN KNOWN INPUT -> COMPARE WITH PYTORCH -> TEST -> SANITIZE.

Give exact commands based on the actual repository and actual build system.

Cover:
- clean CMake configure
- clean build
- strict compiler warnings
- running the Raylib application
- model path
- model loading validation
- known MNIST inference
- failure handling

## 4. C/PyTorch numerical parity

Make this a first-class project milestone.

Compare the exact same input and exact same exported weights through:

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
flatten
logits

For every stage report:
- shape
- max absolute error
- mean absolute error
- tolerance
- PASS/FAIL

Explain why final prediction equality is NOT enough.

Give a debugging tree:

Mismatch -> input -> weights -> conv1 -> ReLU -> conv2 -> pool -> conv3 -> conv4 -> flatten -> linear.

Always fix the FIRST failing stage before changing later code.

Explain how PyTorch tensor layout maps to C contiguous memory.

## 5. Existing CNN audit

Audit, do not blindly rewrite:
- Tensor allocation/free
- indexing
- Linear
- ReLU
- Conv2D
- MaxPool2D
- model_forward
- model_load

For every component include:
1. current implementation
2. correctness contract
3. current tensor shapes
4. exact PyTorch equivalent
5. tests
6. numerical parity method
7. edge cases
8. memory ownership
9. future optimization only after profiling.

Discuss bugs such as:
- NCHW vs NHWC
- wrong flatten order
- incorrect weight ordering
- padding/stride errors
- integer overflow
- invalid dimensions
- uninitialized memory
- use-after-free
- double free.

## 6. Mathematics — only where it is needed

Do not restart the user's mathematics education from zero.

Explain the mathematics in direct connection with the code.

### Linear algebra
- vectors
- matrices
- matrix multiplication
- dot products
- linear layer y = Wx + b
- tensor shapes
- flattening
- contiguous indexing

### Convolution
- kernel/filter
- channels
- output channels
- stride
- padding
- cross-correlation vs mathematical convolution
- output-size formula
- the actual project's 3x3, stride 1, padding 1 configuration.

### ReLU
ReLU(x)=max(0,x).
Explain why shape is unchanged.

### MaxPool2D
Explain the EXISTING implementation.
Explain 2x2 stride 2 and why 28x28 -> 14x14 -> 7x7.
Explain why initializing the maximum with 0 is wrong when values can all be negative and why -INFINITY is appropriate.

### Flatten
Explain 32*7*7=1568 and exactly how the C buffer corresponds to the PyTorch tensor.

### Softmax
Explain stable softmax using max-logit subtraction.

### Cross entropy
Explain logits, target classes, softmax probabilities, and CrossEntropyLoss at the level required to understand training.

### Optimization
Connect gradients, chain rule, SGD/Adam, learning rate, and batch size to the existing training code.

## 7. Preprocessing and domain shift

Audit the existing C preprocessing rather than replacing it blindly.

Explain the current pipeline and then establish the target:

canvas -> foreground detection -> bounding box -> square crop -> margin -> resize -> centering -> 28x28 -> normalization -> CNN.

Explain why MNIST and mouse drawings have different distributions.

Create a deterministic preprocessing test matrix:
- blank canvas
- centered digit
- top-left digit
- bottom-right digit
- tiny digit
- huge digit
- wide digit
- tall digit
- thick stroke
- thin stroke.

Explain how to save 28x28 intermediate tensors/images for inspection.

## 8. Raylib C UI is the main product

Do NOT create another Python GUI.

Future C UI roadmap:
- smooth drawing
- clear/reset
- predict
- prediction
- confidence
- ten-class probability bars
- preprocessing preview
- optional debug mode
- optional activation visualization
- useful keyboard shortcuts
- clean error states
- stable frame rate.

Explain how UI code should eventually be separated into responsibilities without premature architecture work.

Suggested eventual conceptual modules:
app / ui / input / preprocess / tensor / model / debug / io.

## 9. Activation visualization

Make this a signature feature later.

Visualize:
- input
- Conv1 feature maps
- Conv2 feature maps
- Conv3 feature maps
- Conv4 feature maps
- pooled outputs
- logits
- probabilities.

Explain memory ownership and how to expose tensors safely without unnecessary allocations.

## 10. Model format

The current raw binary format can remain for the current milestone.

Future versioned format should include:
- magic
- version
- architecture ID
- dtype
- tensor count
- tensor metadata
- payload
- checksum.

Explain why an unstructured sequence of floats is fragile and how the future loader should reject incompatible models.

## 11. Tests

Design actual tests for:

Tensor:
- allocation
- shape
- indexing
- initialization
- free.

Math:
- linear
- ReLU
- argmax
- Conv2D
- MaxPool2D.

Model:
- valid model loading
- truncated model rejection
- invalid size rejection
- forward pass.

Preprocessing:
- all edge cases listed above.

Integration:
- known MNIST input
- C/PyTorch parity
- end-to-end prediction.

Memory:
- AddressSanitizer
- UndefinedBehaviorSanitizer
- Valgrind where useful.

Give concrete commands where possible.

## 12. Build system and CI

CMake is the canonical C build system.

Standard flow:

cmake -S . -B build
cmake --build build

Plan future targets:
- number_guesser
- unit_tests
- benchmark_c
- parity_test.

Design CI for:
- configure
- compile with warnings
- tests
- benchmark smoke test
- optional sanitizer build
- optional parity test.

CI should fail if numerical parity fails.

## 13. Profiling and optimization

Do NOT optimize yet.

First measure:
- model loading
- preprocessing
- Conv1
- Conv2
- Pool1
- Conv3
- Conv4
- Pool2
- Linear
- total inference
- allocations
- peak memory.

Then consider:
1. buffer reuse
2. fewer allocations
3. cache-friendly loops
4. convolution loop ordering
5. compiler optimization
6. SIMD
7. parallelism
8. quantization.

Every optimization must have:
Hypothesis -> baseline benchmark -> implementation -> correctness/parity -> new benchmark -> conclusion.

## 14. ML experiments after deployment correctness

Only after parity and testing are trustworthy.

Possible experiments:
- augmentation
- normalization
- kernel sizes
- channel counts
- architecture depth
- optimizer
- learning rate
- batch size
- regularization.

Every experiment must record:
Hypothesis -> one meaningful change -> training configuration -> metric -> result -> interpretation.

Do not change five variables at once.

## 15. Learning roadmap tied to this project

Map D2L and Mathematics for Machine Learning to the repository.

D2L:
- tensors
- CNNs
- training/evaluation
- optimization
- generalization.

MML:
- linear algebra
- calculus
- probability
- optimization.

For every topic say exactly which code/file/concept requires it.

## 16. AI-agent workflow

Recommend:

inspect -> understand -> plan -> implement one change -> compile -> test -> benchmark -> inspect diff -> document -> commit.

Agents must not:
- claim parity without measurements
- invent benchmarks
- rewrite working code casually
- optimize without profiling
- change CNN architecture without a documented experiment
- add dependencies without justification
- delete code without checking references.

## 17. Phased roadmap

Build a detailed roadmap from the CURRENT repository state:

Phase 0 — Clean baseline
- build
- run
- model loading
- remove obsolete code
- reproducibility.

Phase 1 — Numerical parity
- reference tensors
- C tensors
- comparison
- tolerance
- first mismatch.

Phase 2 — Preprocessing
- deterministic preprocessing
- edge-case tests
- saved 28x28 inspection.

Phase 3 — C/Raylib product
- drawing
- prediction
- probability UI
- robust error states.

Phase 4 — Explainable inference
- activations
- logits
- layer inspection.

Phase 5 — Engineering quality
- tests
- sanitizers
- CI
- documentation.

Phase 6 — Performance
- profiling
- buffer reuse
- cache/loop optimization
- SIMD only if justified.

Phase 7 — Model format
- versioning
- validation
- checksum.

Phase 8 — ML research experiments
- controlled experiments
- ablations
- model improvements.

For EACH phase give:
- objective
- current starting point
- exact files
- code work
- mathematics
- tests
- commands
- benchmark
- definition of done.

## 18. Definition of done

A milestone is complete only when:
- implementation works
- tests exist
- benchmark passes
- docs match reality
- build is reproducible
- sanitizer errors are absent
- parity is demonstrated where relevant
- performance claims have measurements.

## 19. Final section: Next 10 Tasks

End the guide with exactly ten concrete tasks based on the repository's current state.

Prioritize roughly:
1. clean build
2. application startup
3. model loading
4. known MNIST inference
5. layer-by-layer parity
6. preprocessing validation
7. unit tests
8. sanitizer pass
9. CI
10. activation visualization.

Do not jump directly to a new architecture.

## Output quality

PROJECT_GUIDE.md must feel like a serious internal engineering document for a multi-month career project.

It must tell the author:
- where the project is now
- what is already solved
- what is not yet proven
- what to study
- what code to change
- how to test it
- how to benchmark it
- why the step matters
- what exact milestone comes next.

Do not write motivational filler.
Do not restart from beginner programming lessons.
Do not repeat implementation tutorials for already-working components unless auditing, testing, or extending them requires it.
Do not invent results.
Use the repository's real filenames, functions, tensor shapes, and architecture.