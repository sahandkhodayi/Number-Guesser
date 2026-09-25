# Number Guesser — Continuation Guidebook / Engineering Book Prompt

You are writing PROJECT_GUIDE.md for https://github.com/sahandkhodayi/Number-Guesser.

Your task is to produce a serious, book-like, sequential continuation guide for this repository.

This is NOT a generic roadmap, checklist, beginner course, or coding-agent prompt. It must read like a technical textbook written specifically for this project.

## Critical starting point

Start from the ACTUAL CURRENT repository state.

The project already has substantial work completed: PyTorch training/reference code, trained CNN, exported weights, C CNN inference, Tensor, Conv2D, ReLU, MaxPool2D, Linear/argmax, preprocessing, Raylib/C UI, model loading, CMake, and benchmark/reference tooling.

Do NOT restart with what is C, what is Python, what is a neural network, what is a CNN, or how to implement MaxPool2D from zero.

If a component already exists, use this pattern:

inspect existing implementation → explain it → explain the math → verify it → test it → identify remaining problems → extend it.

Never recreate working code merely for style.

## Source of truth

Inspect the actual repository before writing the guide:

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
- relevant Git history

Code is the primary source of truth. If documentation disagrees with code, document the actual code state and explicitly mention the discrepancy.

Never invent completed work, test output, benchmark numbers, or files.

## Style reference

The requested style is the previous Number Guesser implementation book: a long, sequential document with a Table of Contents, numbered chapters, explanations, actual code/file references, mathematics, worked calculations, commands, tests, expected results, debugging examples, and a clear next step after every chapter.

The book must feel like a 6–12 month engineering textbook for this exact repository.

Do NOT turn it into a list of generic recommendations.

## Required opening

Start PROJECT_GUIDE.md with:

# Number Guesser — Project Continuation Book

Then provide:
- current project checkpoint
- current architecture
- current state table
- target architecture
- Table of Contents

Current-state table:

| Component | Current implementation | Files | Proven? | Remaining work |
|---|---|---|---|---|

Then explain exactly where the project currently stops and where the book begins.

## The book's chapter rule

Every major implementation chapter MUST contain these sections:

### Objective
What are we building or proving?

### Why
Why does this matter for this project?

### Current state
What already exists and what is not yet proven?

### Files
Exact files and functions involved.

### Theory
Explain the concept at the level necessary for this project.

### Mathematics
Derive the relevant equations and substitute the project's real dimensions.

Do not merely write a formula. For example, for a convolution output:

out = floor((N + 2P - K) / S) + 1

actually substitute the project's N, P, K, and S and explain the result.

### Code
Show the relevant existing code and explain it. For new work, show only the necessary code and explain every important line.

### Step-by-step implementation
Number every action the user should take.

### Build and run
Give exact commands based on the actual repository.

### Test
Give focused tests.

### Expected result
Only use verified results from the repository. Clearly label expected/illustrative output when it is not verified.

### If it fails
Give a decision tree for diagnosing the failure.

### Definition of done
Concrete conditions that must pass.

### Next
Explain exactly what the next chapter builds on.

Do this repeatedly throughout the book.

## Chapter 1 — Current checkpoint

Document the actual state of:
- PyTorch model
- exported weights
- C model
- Tensor
- Conv2D
- ReLU
- MaxPool2D
- Linear
- preprocessing
- Raylib UI
- CMake
- benchmarks
- tests
- parity
- sanitizers
- CI
- documentation

Explain the actual CNN architecture using the repository's real shapes.

## Chapter 2 — How to work through the book

Use this workflow:

Read chapter → understand math → inspect current code → make one change → compile → focused test → compare reference → debug → commit → next chapter.

Never implement several milestones simultaneously.

## Chapter 3 — Clean baseline

Make the current system trustworthy before adding features.

BUILD → RUN → LOAD MODEL → KNOWN INPUT → PYTORCH COMPARISON → TEST → SANITIZE.

Cover clean CMake configure, clean build, strict warnings, Raylib startup, model path, model loading, known MNIST inference, and failure handling.

## Chapter 4 — Audit the existing C runtime

Audit rather than rewrite:
- Tensor allocation/free
- indexing
- Linear
- ReLU
- argmax
- Conv2D
- MaxPool2D
- model_forward
- model_load

For each explain purpose, current algorithm, shape, memory ownership, PyTorch equivalent, mathematical contract, tests, edge cases, and future optimization.

### Required MaxPool2D example

MaxPool2D already exists. Do NOT teach it from zero.

Instead:
1. locate the existing implementation
2. explain its loops
3. explain its indexing
4. explain 2×2 stride 2
5. calculate 28×28 → 14×14 → 7×7
6. explain why max initialization must handle negative values
7. explain why 0 is wrong and -INFINITY is correct
8. compare it to PyTorch MaxPool2d(2,2)
9. write/describe a focused test
10. verify it
11. continue.

Apply the same audit style to every existing subsystem.

## Chapter 5 — Model loading and serialization

Explain CnnModel, every parameter count, weight ordering, float32 representation, binary layout, export order, loading order, file-size validation, and architecture compatibility.

Calculate the actual parameter counts from the repository.

Show exactly how a change in model.py could break export/load parity.

Then design the future versioned format with magic, version, architecture ID, dtype, tensor count, shape metadata, payload, and checksum.

## Chapter 6 — Numerical parity

This is a first-class milestone.

Compare the exact same input and exact same weights through:

input → Conv1 → ReLU1 → Conv2 → ReLU2 → Pool1 → Conv3 → ReLU3 → Conv4 → ReLU4 → Pool2 → Flatten → logits

For every stage compare:
- shape
- first values
- max absolute error
- mean absolute error
- tolerance
- PASS/FAIL

Explain why matching the final predicted digit is not enough.

Build a first-mismatch debugging tree and teach the user to fix the earliest divergence before debugging later layers.

Explain NCHW, contiguous memory, C indexing, PyTorch layout, padding, stride, kernel indexing, and flatten order.

## Chapter 7 — Preprocessing and domain shift

First document the ACTUAL preprocessing in the repository.

Then compare it against the training-side MNIST preprocessing.

Explain:
- canvas resolution
- grayscale
- pixel range
- downsampling
- bounding box
- centering
- scaling
- stroke width
- inversion
- normalization
- domain shift

Build deterministic preprocessing fixtures for blank, centered, off-center, tiny, huge, thin, thick, wide, and tall digits.

Explain how to save and inspect the resulting 28×28 tensors.

## Chapter 8 — Raylib C product

C/Raylib is the main runtime. Do NOT create another Python GUI.

Roadmap:
draw → preprocess → predict → digit → confidence → ten-class probabilities → preprocessing preview → debug mode → activation visualization.

Explain UI responsibilities only as required.

## Chapter 9 — Tests

Design actual tests for:

Tensor: allocation, zero initialization, shape, indexing, free.

Math: Linear, ReLU, argmax, Conv2D, MaxPool2D.

Model: valid loading, wrong-size rejection, truncated file, forward pass.

Preprocessing: deterministic fixtures.

Integration: known MNIST input, C/PyTorch parity, end-to-end prediction.

For every test explain what bug it protects against.

## Chapter 10 — Sanitizers

Teach the exact project workflow for AddressSanitizer and UndefinedBehaviorSanitizer.

Explain buffer overflow, use-after-free, double free, invalid access, leaks, and how to interpret sanitizer reports.

Use the actual CMake/compiler setup.

## Chapter 11 — CI

Design CI only after local tests work:

configure → build → warnings → tests → sanitizers → benchmark smoke test → parity.

Explain why parity should fail CI when relevant.

## Chapter 12 — Network visualization

Make this a signature feature.

Visualize:
- input
- Conv1 feature maps
- Conv2 feature maps
- Conv3 feature maps
- Conv4 feature maps
- pooled outputs
- logits
- probabilities

Explain what feature maps mean mathematically and how to expose them without unnecessary memory allocations.

## Chapter 13 — Profiling

Do not optimize before measuring.

Measure preprocessing, every convolution, pooling, linear, total inference, allocations, and memory.

Explain what each measurement tells us and identify the actual bottleneck from measurements.

## Chapter 14 — C optimization

Only after correctness and profiling:

1. buffer reuse
2. allocation reduction
3. cache locality
4. loop ordering
5. compiler optimization
6. SIMD if justified
7. parallelism if justified
8. quantization if justified

Every optimization must follow:

Hypothesis → baseline → implementation → parity/correctness → benchmark → decision.

Never claim an optimization is better without measurement.

## Chapter 15 — ML experiments

Only after deployment correctness.

Possible controlled experiments:
- augmentation
- normalization
- kernel size
- channel count
- architecture depth
- optimizer
- learning rate
- batch size
- regularization

Every experiment:

Hypothesis → one meaningful change → train → evaluate → record metric → interpret → keep/reject.

Do not change five variables at once.

## Chapter 16 — Mathematics through the project

Do not create disconnected mathematics chapters.

For every topic answer: Where does this appear in Number Guesser?

Linear algebra:
- vectors
- matrices
- dot products
- matrix multiplication
- linear layer y = Wx + b
- tensor shape
- flattening
- memory layout

Convolution:
- filters
- channels
- cross-correlation
- stride
- padding
- output dimensions

Calculus:
- derivatives
- partial derivatives
- gradients
- chain rule
- backpropagation

Probability:
- logits
- softmax
- probabilities
- cross entropy
- confidence

Optimization:
- gradient descent
- SGD
- Adam
- learning rate
- batch size
- convergence

Use real dimensions and equations from this repository.

## Chapter 17 — D2L + MML learning map

Create a table:

| Topic | Why needed here | D2L/MML material | Code connection | Study before |
|---|---|---|---|---|

Use just-in-time learning. Do not tell the user to finish an entire textbook before continuing the project.

## Chapter 18 — AI-agent workflow

Document the safe workflow:

inspect → understand → plan → implement one change → compile → test → benchmark → inspect diff → document → commit.

Agents must not:
- claim parity without measurements
- invent benchmarks
- rewrite working code casually
- optimize without profiling
- change architecture casually
- add dependencies without justification
- delete files without checking references.

## Chapter 19 — Long-term phases

Build the final roadmap from the actual current state.

Phase 0: trustworthy baseline.
Phase 1: numerical parity.
Phase 2: preprocessing correctness.
Phase 3: Raylib product.
Phase 4: explainable inference.
Phase 5: tests/sanitizers/CI.
Phase 6: profiling/performance.
Phase 7: model serialization.
Phase 8: ML experiments.

For EACH phase provide:
- objective
- current starting point
- exact files
- exact code work
- mathematics
- tests
- commands
- benchmark
- definition of done
- next phase dependency.

## Chapter 20 — Definition of done

A milestone is complete only when:
- implementation works
- tests exist
- parity passes where relevant
- sanitizer checks pass
- documentation matches reality
- build is reproducible
- benchmark is recorded
- no known regression remains.

## Final chapter — Next 10 Tasks

End with exactly ten concrete tasks based on the repository state at generation time.

Each task MUST have:
- objective
- why
- files
- math
- implementation steps
- exact commands
- test
- expected result
- definition of done
- next dependency.

The first tasks should prioritize the current engineering bottlenecks rather than arbitrary new features.

## Style rules

Write like the previous Number Guesser C Implementation Book.

Use:
- numbered chapters
- Table of Contents
- exact filenames
- exact function names
- code snippets
- equations
- worked calculations
- ASCII diagrams
- commands
- test cases
- expected output only when verified
- debugging trees
- failure explanations
- definitions of done
- explicit next steps.

Do not write motivational filler.
Do not restart from beginner programming.
Do not repeat implementation tutorials for components that already work.
Do not invent results.
Do not turn the document into a generic checklist.

The final document should feel like:

THE NEXT 6–12 MONTHS OF NUMBER GUESSER TURNED INTO ONE TECHNICAL BOOK.

The user should be able to work through Chapter 1 → Chapter 2 → Chapter 3 → ... and always know exactly what to inspect, understand, implement, test, measure, and learn next.