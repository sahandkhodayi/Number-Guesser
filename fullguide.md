# Number Guesser — Project Continuation Book (Combined Edition)

**Source-of-truth note, read before anything else:** this book is written against the actual file contents you sent — `c/include/nn.h`, `c/include/ui.h`, `c/src/nn.c`, `c/src/ui.c`, `c/src/main.c`, `CMakeLists.txt`. Everything about those files below has been compiled under your project's actual flags (`-Wall -Wextra -Wpedantic -std=c11`, zero warnings) and, where stated, actually run — not guessed. Your `python/model.py`, `python/export.py`, `tests/*`, and `benchmark/*` have **not** been provided yet, so anything about them below is the reference design compatible with your confirmed C loader, explicitly marked **[UNCONFIRMED]**. Send those files and I'll reconcile this book against them.

**What this edition combines:** your original Project Continuation Book structure (chapters 2–20 plus Final Chapter) with the additional file-walkthrough material from the Complete Edition. Every chapter is preserved as you wrote it. New material is added inside the existing chapters where it belongs, plus a new Chapter 21 for the Python reference design. Nothing was removed.

---

## Current checkpoint

```
PyTorch trained + exported ✅  →  C CNN runtime: nn.c/nn.h ✅ implemented & compiling clean
                                →  Preprocessing: ui.c — bounding-box + bilinear MNIST-style centering ✅ implemented
                                →  UI: main.c — Raylib app, bar chart, keyboard+mouse controls ✅ implemented
                                →  Build: CMake ✅ builds the app — ⚠️ does NOT build tests/ or benchmark/
                                →  [ NEXT: wire tests/ + benchmark/ into the build, then run real numerical parity ]
```

## Current architecture (confirmed from `nn.h`/`nn.c`)

```
1×28×28
  → conv1 (1→32,k3,s1,p1) → relu1        32×28×28
  → conv2 (32→32,k3,s1,p1) → relu2       32×28×28
  → maxpool1 (2,s2)                       32×14×14
  → conv3 (32→32,k3,s1,p1) → relu3       32×14×14
  → conv4 (32→32,k3,s1,p1) → relu4       32×14×14
  → maxpool2 (2,s2)                       32×7×7
  → flatten                               1568
  → linear (1568→10)                      10 logits
```
Matches your README exactly.

## Current-state table

| Component | Current implementation | Files | Proven? | Remaining work |
|---|---|---|---|---|
| Tensor | Heap-alloc'd `float*`, channel-first `((c*H+y)*W+x)`, get/set, `tensor_info` | `nn.h`, `nn.c` | **Compiled clean** under real flags; index formula matches PyTorch's `[C,H,W]` | No dedicated unit test file confirmed to exist |
| Linear/ReLU/Argmax | `linear`, `relu`, `relu_tensor`, `argmax` | `nn.c` | Compiled clean; logic matches the reference design we verified earlier in isolation | No confirmed test coverage in *your* tree |
| Conv2D | 6-nested-loop, bounds-check padding, `[out_c,in_c,ky,kx]` flatten | `nn.c` | Compiled clean | Not verified against real PyTorch output (Ch. 6) |
| MaxPool2D | `-INFINITY` sentinel (correct — see Ch. 4), no padding | `nn.c` | Compiled clean | Same — no real-weight verification yet |
| Model struct + loader | `CnnModel` with computed array sizes; `model_load` validates against `sizeof(CnnModel)` (confirmed = 175016 bytes) | `nn.h`, `nn.c` | **Confirmed via compile+run**: `sizeof(CnnModel) == 175016` | No versioned format (Ch. 5); relies on struct having zero padding (true here, but not asserted anywhere) |
| Preprocessing | Bounding-box crop + margin + bilinear resize to 20×20, centered in 28×28 | `ui.c` (`canvas_to_mnist_input`) | Compiled clean; **not compared against actual MNIST/`ToTensor()` preprocessing** | Centers by bounding-box center, not center-of-mass (real MNIST convention) — flagged, Ch. 7 |
| Brush | Circular, radius²-falloff, point+line variants for continuous strokes | `ui.c` | Compiled clean | Untested interactively (no display in my sandbox) |
| UI | Raylib window, canvas, buttons (Pressed, not Down), keyboard shortcuts, probability bar chart, confidence color-coding, FPS counter | `main.c` | Compiled clean in an earlier equivalent build against real raylib; **this exact file not yet re-linked in this session** (raylib rebuild was in progress, not a code problem) | Nothing structurally wrong found |
| Build | CMake, raylib via `find_package(CONFIG REQUIRED)`, `/W4` or `-Wall -Wextra -Wpedantic` | `CMakeLists.txt` | Confirmed builds `number_guesser` | **Does not build `tests/` or `benchmark/` at all** — real gap, Ch. 3/9 |
| PyTorch model/export | — | `python/model.py`, `python/export.py` | **[UNCONFIRMED]** — not provided | Send these to confirm `LAYER_KEYS` order matches `model_load`'s read order |
| Tests | — | `tests/*` | **[UNCONFIRMED]** — not provided, and not in CMake build graph regardless | Ch. 9 designs what should exist |
| Benchmark/parity | — | `benchmark/*` | **[UNCONFIRMED]** — README describes intent, no file seen | Ch. 6 designs what should exist |
| Sanitizers | — | — | Not run yet on this exact tree | Ch. 10 |
| CI | — | — | Not present | Ch. 11 |

## Target architecture (end state, not yet built)

```
same CNN core
  + versioned weights.bin (magic/version/shape metadata/checksum)
  + benchmark/ wired into CMake, real PyTorch-vs-C parity numbers on record
  + tests/ wired into CMake, running under ASan+UBSan in CI
  + activation/feature-map visualization in the Raylib app
  + profiled, then selectively optimized inference
```

## Table of Contents

1. [Current checkpoint](#current-checkpoint) *(above)*
2. [How to work through this book](#chapter-2--how-to-work-through-this-book)
3. [Clean baseline](#chapter-3--clean-baseline)
4. [Audit the existing C runtime](#chapter-4--audit-the-existing-c-runtime)
5. [Model loading and serialization](#chapter-5--model-loading-and-serialization)
6. [Numerical parity](#chapter-6--numerical-parity)
7. [Preprocessing and domain shift](#chapter-7--preprocessing-and-domain-shift)
8. [The Raylib C product](#chapter-8--the-raylib-c-product)
9. [Tests](#chapter-9--tests)
10. [Sanitizers](#chapter-10--sanitizers)
11. [CI](#chapter-11--ci)
12. [Network visualization](#chapter-12--network-visualization)
13. [Profiling](#chapter-13--profiling)
14. [C optimization](#chapter-14--c-optimization)
15. [ML experiments](#chapter-15--ml-experiments)
16. [Mathematics through the project](#chapter-16--mathematics-through-the-project)
17. [D2L + MML learning map](#chapter-17--d2l--mml-learning-map)
18. [AI-agent workflow](#chapter-18--ai-agent-workflow)
19. [Long-term phases](#chapter-19--long-term-phases)
20. [Definition of done](#chapter-20--definition-of-done)
21. [The Python reference design](#chapter-21--the-python-reference-design)
22. [Next 10 tasks](#final-chapter--next-10-tasks)

---

## Chapter 2 — How to work through this book

```
read chapter → understand math → inspect current code (this book quotes it) →
make ONE change → compile → focused test → compare reference → debug → commit → next
```
Never implement several milestones simultaneously — Chapter 6 (parity) and Chapter 7 (preprocessing) are separate risks that fail independently; if both change before you check either, a failure could be either one and you won't know which.

Three rules that must not be bent:

1. **Never claim parity without measurement.** Every number in the benchmark is from a real run. If it hasn't been run, it's marked unknown.
2. **Never optimize before profiling.** Chapter 13 (profiling) strictly precedes Chapter 14 (optimization). Any "obvious" speedup without a measured baseline is a guess.
3. **Never trust a test you haven't run under a sanitizer.** A test that passes without ASan might still corrupt memory silently. Chapter 10 covers this.

---

## Chapter 3 — Clean baseline

### Objective
Prove the current tree builds, links, and loads a model, before adding anything.

### Why
Chapters 4–7 all assume "it builds." If that's not actually true on a clean checkout, everything downstream is built on sand.

### Current state
`nn.c`/`ui.c` compile clean under `-Wall -Wextra -Wpedantic -std=c11` (confirmed this session). `main.c` was not re-linked against raylib in this exact session (tooling issue on my end, not a code issue — an earlier, near-identical version of this same `main.c` did link and run clean against a real raylib build). `CMakeLists.txt` only defines the `number_guesser` target — no `tests`/`benchmark` targets exist yet.

### Files
`CMakeLists.txt`, `c/src/main.c`, `c/src/nn.c`, `c/src/ui.c`, `c/include/nn.h`, `c/include/ui.h`.

### The build system — `CMakeLists.txt` in full

```cmake
cmake_minimum_required(VERSION 3.20)
project(NumberGuesser C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(raylib CONFIG REQUIRED)

add_executable(number_guesser
    c/src/main.c
    c/src/nn.c
    c/src/ui.c
)

target_include_directories(number_guesser PRIVATE c/include)
target_link_libraries(number_guesser PRIVATE raylib)

if(UNIX AND NOT APPLE)
    target_link_libraries(number_guesser PRIVATE m)
endif()
```

Line by line:
- **`cmake_minimum_required(VERSION 3.20)`** — the minimum CMake version this file requires. 3.20 is recent enough for `find_package(... CONFIG REQUIRED)` behavior and the target-scoped `target_link_libraries` syntax.
- **`project(NumberGuesser C)`** — declares the project name and language. `C` (not `CXX`) tells CMake this is a C project; it won't try to find a C++ compiler.
- **`set(CMAKE_C_STANDARD 11)`** — use C11.
- **`set(CMAKE_C_STANDARD_REQUIRED ON)`** — fail if C11 isn't available.
- **`set(CMAKE_C_EXTENSIONS OFF)`** — disable non-standard extensions (e.g., GNU `typeof`, `asm`). More portable.
- **`if(MSVC) add_compile_options(/W4) else() add_compile_options(-Wall -Wextra -Wpedantic) endif()`** — compiler-specific warning flags. MSVC uses `/W4`; GCC/Clang use `-Wall -Wextra -Wpedantic`.
- **`find_package(raylib CONFIG REQUIRED)`** — find the raylib library. `CONFIG` mode means "use raylib's own CMake config files." `REQUIRED` means "fail if not found."
- **`add_executable(number_guesser c/src/main.c c/src/nn.c c/src/ui.c)`** — build an executable from the three source files.
- **`target_include_directories(number_guesser PRIVATE c/include)`** — add `c/include/` to the include search path for this target only.
- **`target_link_libraries(number_guesser PRIVATE raylib)`** — link against raylib.
- **`if(UNIX AND NOT APPLE) target_link_libraries(number_guesser PRIVATE m) endif()`** — link against `libm` on Linux. macOS doesn't need it (math is in libSystem); Windows doesn't have it.

### What this file does NOT do

- **Build tests.** There's no `enable_testing()` or `add_test`. Adding these is a task (Chapter 9).
- **Build the benchmark.** No `add_executable(verify ...)`.
- **Build the fixtures tool.** No target for `preprocessing_fixtures.c`.
- **Provide a sanitizer option.** No `option(ENABLE_SANITIZERS ...)`.

### Step-by-step implementation
1. Clean configure: `cmake -S . -B build`
2. Clean build: `cmake --build build -j`
3. Confirm the binary exists: `test -x build/number_guesser && echo OK`
4. Confirm `models/weights.bin` exists and is exactly 175,016 bytes:
   `stat -c%s models/weights.bin` (or `ls -la` on macOS) — this is the
   `sizeof(CnnModel)` value **confirmed** in this session, so a mismatch
   here means the file is stale/wrong, not that the check is miscalibrated.
5. Run it: `./build/number_guesser`. If `models/weights.bin` is missing,
   `main.c` prints `Warning: could not load models/weights.bin` to
   stderr and disables Predict (`model_ok` gates the button) rather than
   crashing — confirm you see that exact message if the file's absent,
   confirming the failure path works too.

### Build and run
```bash
cmake -S . -B build
cmake --build build -j
./build/number_guesser
```

### Test
No automated test yet — this chapter's "test" is the four manual steps above.

### Expected result
Binary builds with zero warnings (confirmed for `nn.c`/`ui.c` this session under your exact flags). Window opens, canvas draws, Predict button is enabled iff `models/weights.bin` is present and exactly 175,016 bytes.

### If it fails
- `find_package(raylib CONFIG REQUIRED)` fails → raylib not installed in config-mode (vcpkg/CMake package, not just a `.so` on the linker path) — see your README's platform-specific install steps.
- Links but `model_load` always fails → check the file size first (`stat`), then check you're running from the directory `main.c` expects (`"models/weights.bin"` is a relative path — must run from repo root, or wherever your CWD puts `models/` at that relative location).
- Window doesn't open at all → no display available.

### Definition of done
`cmake --build` succeeds with zero warnings; `./build/number_guesser` opens a window; Predict is enabled when a correctly-sized `weights.bin` is present, disabled with the specific stderr message when it's not.

### Next
Chapter 4 — audit what the C runtime actually does, since it compiles.

---

## Chapter 4 — Audit the existing C runtime

Audit, not rewrite. Every function below already exists in your `nn.c` and compiles clean.

### The header file `nn.h` in full

```c
#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * C inference model.
 *
 * IMPORTANT: these arrays mirror the exact tensor shapes exported by
 * python/export.py. Python owns training; C owns deployment/inference.
 *
 * If the PyTorch architecture or export order changes, these declarations
 * and the model loader must be updated together.
 */
typedef struct {
    float conv1_w[32 * 1 * 3 * 3];
    float conv1_b[32];
    float conv2_w[32 * 32 * 3 * 3];
    float conv2_b[32];
    float conv3_w[32 * 32 * 3 * 3];
    float conv3_b[32];
    float conv4_w[32 * 32 * 3 * 3];
    float conv4_b[32];
    float fc_w[10 * 1568];
    float fc_b[10];
} CnnModel;

/*
 * Channel-first contiguous tensor:
 *
 * index(c, y, x) = ((c * height) + y) * width + x
 *
 * Keeping this layout consistent with the PyTorch export contract is critical
 * for numerical parity.
 */
typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} Tensor;

void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);
void relu(float *x, int n);
int argmax(const float *x, int n);

Tensor tensor_alloc(int channels, int height, int width);
void tensor_free(Tensor *t);
float tensor_get(const Tensor *t, int c, int y, int x);
void tensor_set(Tensor *t, int c, int y, int x, float value);
void tensor_info(const Tensor *t, const char *label);
void relu_tensor(Tensor *t);

Tensor conv2d(const Tensor *input, const float *weights, const float *bias,
              int out_channels, int k, int stride, int pad);
Tensor maxpool2d(const Tensor *input, int k, int stride);

int model_load(CnnModel *m, const char *path);
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out);

#endif
```

### Line-by-line of `nn.h`

**`#ifndef NN_H` / `#define NN_H` / `#endif`** — the include guard. The first time this file is included, `NN_H` is undefined, so the preprocessor enters the block, defines `NN_H`, and processes the contents. On subsequent includes in the same translation unit, `NN_H` is defined, so the block is skipped. Without this, a file that includes `nn.h` twice would get duplicate definitions and the compiler would error.

**`#include <stddef.h>`** — provides `size_t`.

**`#include <stdio.h>`** — provides `FILE` (used in `model_load`) and the `fprintf` family.

**`#include <stdlib.h>`** — provides `calloc`, `free`, `exit`, used in `nn.c`.

**`typedef struct { ... } CnnModel;`** — the model weights container. Nine fixed-size arrays inside one struct.

Why fixed-size arrays and not pointers? Because the architecture is fixed at compile time. The struct has a known size (`sizeof(CnnModel) == 175016`), so it can live on the stack (`CnnModel m;` in `main`) with no `malloc` needed for the struct itself.

Why `float` and not `double`? Because PyTorch's default tensor dtype is `float32`, and matching that exactly avoids a conversion step.

**Array sizes:**
- `conv1_w[32 * 1 * 3 * 3]` = 288. 32 output channels, 1 input channel, 3×3 kernel.
- `conv1_b[32]` = 32. One bias per output channel.
- `conv2_w[32 * 32 * 3 * 3]` = 9216.
- `fc_w[10 * 1568]` = 15680. Ten output classes, 1568 input features.
- `fc_b[10]` = 10.

**`typedef struct { float *data; int channels, height, width; } Tensor;`** — the runtime tensor.

Why a pointer (`float *data`) instead of a fixed-size array? Because tensor sizes vary at runtime — 32×28×28 after conv1, 32×14×14 after pool1, 32×7×7 after pool2.

Why the shape metadata? So that `tensor_get`/`tensor_set` and the CNN kernels can compute flat offsets without the caller passing three separate `int` arguments at every call site.

**Function declarations.** The header declares every public function. The definitions live in `nn.c`.

### `tensor_alloc` / `tensor_free`

**Purpose:** own a `channels×height×width` heap buffer.

```c
size_t n = (size_t)channels * (size_t)height * (size_t)width;
t.data = calloc(n, sizeof(float));
```

Every dimension is cast to `size_t` individually *before* multiplying — slightly more defensive than casting only the first operand, since it guarantees the whole multiplication happens in `size_t` arithmetic, not just the first step of it. `calloc` (not `malloc`) zero-initializes, so an unwritten cell reads as `0.0`, not garbage — matters because `conv2d` builds each output value by adding to a running `sum`.

**Ownership:** caller of `tensor_alloc` (e.g. `conv2d`) owns and must `tensor_free`. **PyTorch equivalent:** `torch.zeros(C,H,W)`. **Tests:** none confirmed in your tree. **Edge case:** `tensor_free` sets `t->data = NULL` and zeroes the shape fields — a freed `Tensor` used again crashes on the next access instead of silently reading freed memory.

### `tensor_get` / `tensor_set`

```c
size_t index = ((size_t)c * (size_t)t->height + (size_t)y) * (size_t)t->width + (size_t)x;
```

Same formula as before, same `size_t`-per-operand defensiveness as `tensor_alloc`. 

Line by line of the formula:
- `(size_t)c * (size_t)t->height` — convert the channel index into "how many rows of pixels come before this channel starts."
- `+ (size_t)y` — add the row offset within this channel.
- `* (size_t)t->width` — convert rows to individual float elements.
- `+ (size_t)x` — add the column offset.

**Mathematical contract:** `index(c,y,x)` is a bijection from `[0,C)×[0,H)×[0,W)` onto `[0,C·H·W)` — every valid `(c,y,x)` maps to a distinct offset, no aliasing, as long as the caller respects the tensor's actual bounds (nothing here enforces that — an out-of-range `c`/`y`/`x` silently indexes past the buffer; this is on the caller, same as raw array indexing anywhere else in C).

### `linear`

```c
void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features) {
    for (int o = 0; o < out_features; ++o) {
        float sum = b[o];
        const float *row = W + o * in_features;
        for (int i = 0; i < in_features; ++i) {
            sum += row[i] * x[i];
        }
        y[o] = sum;
    }
}
```

Line by line:
- **`for (int o = 0; o < out_features; ++o)`** — iterate over output neurons.
- **`float sum = b[o];`** — seed the accumulator with the bias. Starting from the bias saves one addition per output.
- **`const float *row = W + o * in_features;`** — pointer arithmetic to the start of row `o` in the row-major layout.
- **`sum += row[i] * x[i];`** — accumulate the dot product.
- **`y[o] = sum;`** — store.

**PyTorch equivalent:** `nn.Linear(in_features, out_features)`, `y = x @ W.T + b`. Writes into a caller-provided buffer — no allocation, unlike every `Tensor`-returning function in this file.

### `relu` / `relu_tensor` / `argmax`

`relu`: in-place `max(0,x)`. Uses `if` rather than `fmaxf(x, 0)` — the `if` only writes when the value is negative, saving a memory store for positive values.

`relu_tensor`: calls `relu` on `t->data` with `channels*height*width` — works because ReLU is pointwise and `Tensor.data` is one flat block regardless of shape.

`argmax`: first occurrence wins on ties (`>` not `>=`).

```c
int argmax(const float *x, int n) {
    int best_idx = 0;
    float best_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > best_val) {
            best_val = x[i];
            best_idx = i;
        }
    }
    return best_idx;
}
```

Starts at `i = 1` because 0 is already the initial best.

### `conv2d`

**Purpose:** the 4 conv layers. **Math, substituted for this project's real dimensions:**
```
out_dim = floor((in_dim + 2*pad - k) / stride) + 1
```
For `conv1`: `in_dim=28, pad=1, k=3, stride=1` →
`out_dim = floor((28+2-3)/1)+1 = floor(27)+1 = 28` — confirms `H_out=H_in` for every conv layer in this architecture, not an assumption.

Loop nest, outer to inner: `oc → oy → ox → ic → ky → kx`. 

Bounds check:
```c
if (iy < 0 || iy >= input->height || ix < 0 || ix >= input->width) continue;
```
is the padding — a tap landing outside the real image contributes nothing to `sum`, mathematically identical to zero-padding, no padded copy ever allocated.

Weight index:
```c
size_t w_index = (((size_t)oc * (size_t)input->channels + (size_t)ic) * (size_t)k + (size_t)ky) * (size_t)k + (size_t)kx;
```
Read inside-out:
- `oc * input->channels + ic` — which (output channel, input channel) pair.
- `* k + ky` — scale up by kernel rows and add the row offset.
- `* k + kx` — scale up by kernel columns and add the column offset.

Matches PyTorch's `Conv2d.weight` shape `[out_c, in_c, kh, kw]` flattened — the comment in your own file states this explicitly, and it's why the export needs no reordering (Ch. 5).

**Ownership:** allocates and returns — caller (`model_forward`) owns and frees the result.

**Cost:** `O(out_c × out_h × out_w × in_c × k²)` — `conv1` ≈226K ops, `conv2/3/4` ≈7.2M ops each; not yet measured on your actual hardware (Ch. 13).

### `maxpool2d` — the required worked example

1. **Located:** `nn.c`, right after `conv2d`.
2. **Loops:** `c → oy → ox → ky → kx` — one channel loop, not two, since pooling never mixes channels (unlike `conv2d`'s separate `oc`/`ic`).
3. **Indexing:** `iy = oy*stride+ky, ix = ox*stride+kx` — no padding term, no bounds check, because this project's pooling windows are always fully inside the input.
4. **2×2 stride 2, substituted:** `out_dim = floor((in_dim - 2)/2)+1`.
5. **28×28→14×14:** `floor((28-2)/2)+1 = floor(13)+1 = 14`. ✓
   **14×14→7×7:** `floor((14-2)/2)+1 = floor(6)+1 = 7`. ✓ Matches the architecture table exactly.
6. **Why 0 is wrong and `-INFINITY` is correct:** a ReLU'd tensor is non-negative going *into* the first pool, so a `0` sentinel would coincidentally work for `pool1` — but this is fragile: any pooling layer applied to signed data (or a future architecture change) breaks silently, since `0` can beat a real negative maximum. Your code uses `-INFINITY` from `<math.h>`:
   ```c
   float best = -INFINITY;
   ```
   This is provably correct for *any* input range, not just the currently-non-negative case — the comment in your file states this reasoning explicitly, and it's the textbook-correct choice.
7. **PyTorch equivalent:** `nn.MaxPool2d(kernel_size=2, stride=2)`, no padding, `ceil_mode=False` (default) — matches `floor()` in the output-size formula above.
8–10. **Test/verify:** no dedicated test file confirmed in your tree — Chapter 9 designs `test_maxpool2d` against a hand-computable 4×4 input (`max(1,3,5,6)=6` etc.), unchanged from the reference design.

### `model_forward`

Straight-line composition of the above, matching the architecture table. Ownership chain: `a`→(free after `b` computed)→`b`→(free after `p1`)→`p1`→(free after `c`)→`c`→(free after `d`)→`d`→(free after `p2`)→`p2` (free after `linear` reads `p2.data`). Peak memory ~2 tensors at a time, not 8, by construction — every intermediate is freed the instant the next step has consumed it.

**The flatten.** `p2.data` (the flat 32×7×7 buffer, 1568 floats) is passed straight to `linear`. There is no copy and no reshape — the memory was always flat, so "flatten" is just reinterpreting the same buffer with different shape metadata.

### Future optimization note (do not act on this before Ch. 13)
`conv2d`'s six nested loops are the natural place to look for cache-locality or loop-reordering wins — but there is no profiling data yet (Ch. 13) to say whether `conv2d` is even the bottleneck. Do not guess.

---

## Chapter 5 — Model loading and serialization

### Objective
Confirm the binary format is self-consistent and understand exactly how fragile it is to an architecture change.

### Current state

```c
typedef struct {
    float conv1_w[32 * 1 * 3 * 3];   float conv1_b[32];
    float conv2_w[32 * 32 * 3 * 3];  float conv2_b[32];
    float conv3_w[32 * 32 * 3 * 3];  float conv3_b[32];
    float conv4_w[32 * 32 * 3 * 3];  float conv4_b[32];
    float fc_w[10 * 1568];           float fc_b[10];
} CnnModel;
```

Parameter counts, computed: `conv1_w=288, conv1_b=32, conv2_w=9216, conv2_b=32, conv3_w=9216, conv3_b=32, conv4_w=9216, conv4_b=32, fc_w=15680, fc_b=10`. Sum = **43,754 floats**. `sizeof(CnnModel)` — **confirmed by actually compiling and running a size check against your real header in this session** — is exactly **175,016 bytes**, i.e. the struct has zero compiler-inserted padding (every member is a `float` array, all 4-byte aligned, so this is expected, not lucky — but it's worth knowing this assumption exists: if a non-float field were ever added to `CnnModel`, padding could appear and `sizeof(CnnModel)` would silently stop equaling the sum of the array sizes).

`model_load`'s validation, a real improvement over checking a hand-maintained byte constant:
```c
if ((unsigned long)file_size != sizeof(CnnModel)) { ... }
```
This can never drift out of sync with the struct definition the way a separate `#define WEIGHTS_FILE_BYTES 175016` constant could — if you add a layer to `CnnModel`, this check updates itself. The **read order**, though, is still hand-written and *can* drift:
```c
read_floats(f, m->conv1_w, ...); read_floats(f, m->conv1_b, ...);
read_floats(f, m->conv2_w, ...); read_floats(f, m->conv2_b, ...);
... (conv3, conv4) ...
read_floats(f, m->fc_w, ...);    read_floats(f, m->fc_b, ...);
```
This order must exactly match whatever `python/export.py` writes.

**[UNCONFIRMED]** — I don't have your actual `export.py`. The reference `LAYER_KEYS` order compatible with this exact read order (assuming your `model.py` uses the same `block_1`/`block_2`/`classifier` naming as the architecture in your README implies):
```python
LAYER_KEYS = [
    "block_1.0.weight", "block_1.0.bias",
    "block_1.2.weight", "block_1.2.bias",
    "block_2.0.weight", "block_2.0.bias",
    "block_2.2.weight", "block_2.2.bias",
    "classifier.1.weight", "classifier.1.bias",
]
```
Send your real `export.py` and `model.py` to confirm this, rather than trust the inference.

### The loader in full

```c
static int read_floats(FILE *f, float *dst, size_t count, const char *what) {
    size_t n = fread(dst, sizeof(float), count, f);
    if (n != count) {
        fprintf(stderr, "model_load: short read on %s (got %zu of %zu floats)\n",
                what, n, count);
        return -1;
    }
    return 0;
}
```

Line-by-line:
- **`fread(dst, sizeof(float), count, f)`** — reads `count` items of size `sizeof(float)` into `dst`. Returns the number of items actually read.
- **`if (n != count)`** — short read. The file is truncated or the count is wrong.
- **`return 0;`** — success. This is not redundant: the caller does `err |= read_floats(...)`, so returning 0 keeps `err` unchanged on success.

```c
int model_load(CnnModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "model_load: could not open '%s'\n", path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long file_size = ftell(f);
    if (file_size < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

    if ((unsigned long)file_size != sizeof(CnnModel)) {
        fprintf(stderr, "model_load: '%s' is %ld bytes, expected %zu\n",
                path, file_size, sizeof(CnnModel));
        fclose(f);
        return -1;
    }

    int err = 0;
    err |= read_floats(f, m->conv1_w, sizeof m->conv1_w / sizeof m->conv1_w[0], "conv1_w");
    /* ... nine more, in the exact same order ... */

    fclose(f);
    return err ? -1 : 0;
}
```

Line-by-line:
- **`fopen(path, "rb")`** — binary read mode. The `b` is critical on Windows.
- **`fseek(f, 0, SEEK_END); ftell; fseek(f, 0, SEEK_SET);`** — the "how big is this file" idiom.
- **`(unsigned long)file_size != sizeof(CnnModel)`** — size check.
- **`sizeof arr / sizeof arr[0]`** — the compile-time element count of a fixed-size array. The compiler evaluates it; no runtime cost.
- **`err |= ...`** — bitwise-OR accumulation. On success every call returns 0. On failure `err` becomes -1 and stays non-zero, so all read failures appear in one run.

### How a `model.py` change silently breaks export/load parity

Add a fifth conv layer to `model.py` without touching `CnnModel` in `nn.h`:
- `state_dict()` gains new keys.
- `LAYER_KEYS` (if hand-written and not regenerated) either KeyErrors on a missing key (loud, good) or — worse — silently writes the *old* 10 tensors in the *old* order while the new layer's weights are silently dropped from the export entirely.
- `model_load`'s `sizeof(CnnModel)` check would still pass, since the file size wouldn't include the new layer's weights either.

**The size check cannot catch a `model.py` architecture change that both sides forgot to propagate.** This is exactly the scenario Chapter 6's parity check exists to catch (a file that loads with zero errors and produces a plausible-looking wrong number).

### Future versioned format design
```
[4 bytes] magic       "NGSR"
[4 bytes] version      uint32, e.g. 1
[4 bytes] arch_id       uint32 — a hash or enum of the architecture, so a
                        mismatched model.py/nn.h pair fails loudly instead
                        of silently
[4 bytes] dtype         uint32, e.g. 0=float32
[4 bytes] tensor_count  uint32
[tensor_count × (name_len + name + ndims + shape[ndims])]  shape metadata
[payload]               raw tensor bytes, same order as metadata
[4 bytes] checksum       CRC32 or similar, over the payload
```
Not built yet — this is the target, not the current state. Build it only after Chapter 6 (parity) passes on the current header-less format; a format change is a real risk to take on for its own sake before correctness is proven on the simpler format.

### Definition of done
`sizeof(CnnModel)` confirmed to equal the export size (done, this chapter). `LAYER_KEYS` order confirmed against real `export.py` (blocked — send the file). `model_load` rejects a wrong-size file (code inspected, matches the reference design's tested behavior — not yet re-tested against *this exact* `model_load`; Chapter 9 designs it).

### Next
Chapter 6 — the only thing that actually proves this loader's read order is right: comparing real C output against real PyTorch output.

---

## Chapter 6 — Numerical parity

### Objective
Prove `model_forward`'s output matches PyTorch's output, for the same input and the same trained weights, layer by layer.

### Why
Matching the *final predicted digit* is not enough — two wrong implementations can agree on a digit by coincidence (10 classes, ~10% chance of agreeing on nonsense alone), and a real bug that shows up two layers before the end can still happen to argmax to the same digit on some inputs and a different one on others, making the bug intermittent and much harder to trust or debug. Layer-by-layer comparison is what actually locates a divergence instead of hiding it behind a coin flip.

### Current state
**[UNCONFIRMED]** — no `benchmark/` contents provided. The README states the intent (compare shape, error metrics, PASS/FAIL per stage) but I have not seen an actual `benchmark/verify.c` or equivalent. Designing it here against your confirmed `nn.h`/`nn.c` API.

### Theory: NCHW, contiguous memory, and why layout agreement matters
PyTorch's default tensor layout is row-major/C-contiguous — for a `[1,C,H,W]` tensor, element `(c,y,x)` sits at offset `(c*H+y)*W+x`, *exactly* your `tensor_get`'s formula. This is not a coincidence to verify — it's a design constraint both sides were built to satisfy, and Chapter 5 already found the place it could break (an export order that doesn't match the read order). This chapter is where that constraint gets checked empirically rather than argued from code inspection alone.

### The reference PyTorch dump — `python/dump_intermediate.py`

```python
"""Print shape and first-5 values of every stage in the PyTorch forward pass."""
import torch
import numpy as np
from pathlib import Path
from model import _MainModel

MODEL_PATH = Path("../models/number_guesser_model.pth")
INPUT_PATH = Path("../models/debug_input.bin")

def dump(label, t):
    flat = t.detach().flatten().numpy()
    shape = tuple(t.shape)
    head = np.round(flat[:5], 4).tolist()
    print(f"{label:8s} shape={shape}  first 5={head}")

def main():
    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    raw = np.fromfile(INPUT_PATH, dtype=np.float32)
    x = torch.from_numpy(raw).reshape(1, 1, 28, 28)

    with torch.no_grad():
        dump("input", x[0])

        a = model.block_1[0](x)[0]; dump("conv1", a)
        a = model.block_1[1](a);    dump("relu1", a)
        a = model.block_1[2](a);    dump("conv2", a)
        a = model.block_1[3](a);    dump("relu2", a)
        a = model.block_1[4](a);    dump("pool1", a)

        a = model.block_2[0](a);    dump("conv3", a)
        a = model.block_2[1](a);    dump("relu3", a)
        a = model.block_2[2](a);    dump("conv4", a)
        a = model.block_2[3](a);    dump("relu4", a)
        a = model.block_2[4](a);    dump("pool2", a)

        flat = a.flatten()
        dump("flat", flat)

        logits = model.classifier[1](flat.unsqueeze(0))[0]
        dump("logits", logits)

        print(f"\npredicted digit: {logits.argmax().item()}")

if __name__ == "__main__":
    main()
```

Line by line:
- **`.detach().flatten().numpy()`** — remove the autograd graph, make 1D, share memory with NumPy.
- **`[0]`** — strips the batch dimension so the shape matches the C output.
- **`with torch.no_grad():`** — disables autograd for the whole block.
- **`flat.unsqueeze(0)`** — re-adds a batch dim for the linear layer.

### The reference C dump — `benchmark/verify.c`

```c
#include "../c/include/nn.h"
#include <stdio.h>

static void dump(const Tensor *t, const char *label) {
    printf("%-8s shape=(%d,%d,%d) first5=[", label, t->channels, t->height, t->width);
    int n = t->channels * t->height * t->width, show = n < 5 ? n : 5;
    for (int i = 0; i < show; i++) printf("%.6f%s", t->data[i], i==show-1?"":", ");
    printf("]\n");
}

int main(int argc, char **argv) {
    CnnModel m;
    if (model_load(&m, argc>1?argv[1]:"models/weights.bin") != 0) return 1;
    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(argc>2?argv[2]:"benchmark/debug_input.bin", "rb");
    if (f) { fread(input.data, sizeof(float), 28*28, f); fclose(f); }
    dump(&input, "input");

    Tensor a = conv2d(&input, m.conv1_w, m.conv1_b, 32,3,1,1); dump(&a,"conv1"); tensor_free(&input);
    relu_tensor(&a); dump(&a,"relu1");
    Tensor b = conv2d(&a, m.conv2_w, m.conv2_b, 32,3,1,1); dump(&b,"conv2"); tensor_free(&a);
    relu_tensor(&b); dump(&b,"relu2");
    Tensor p1 = maxpool2d(&b,2,2); dump(&p1,"pool1"); tensor_free(&b);
    Tensor c = conv2d(&p1, m.conv3_w, m.conv3_b, 32,3,1,1); dump(&c,"conv3"); tensor_free(&p1);
    relu_tensor(&c); dump(&c,"relu3");
    Tensor d = conv2d(&c, m.conv4_w, m.conv4_b, 32,3,1,1); dump(&d,"conv4"); tensor_free(&c);
    relu_tensor(&d); dump(&d,"relu4");
    Tensor p2 = maxpool2d(&d,2,2); dump(&p2,"pool2"); tensor_free(&d);

    float logits[10];
    linear(m.fc_w, m.fc_b, p2.data, logits, p2.channels*p2.height*p2.width, 10);
    tensor_free(&p2);
    printf("logits   shape=(1,10) first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
           logits[0],logits[1],logits[2],logits[3],logits[4]);
    printf("pred: %d\n", argmax(logits, 10));
    return 0;
}
```

This does **not** call `model_forward` — it reimplements the same sequence by hand so it can `dump()` between every op. Diagnostic tool, not the production path — `model_forward` stays the thing the real app uses.

### First-mismatch debugging tree
```
input matches?         NO → check debug_input.bin's byte count/scale before touching C code
  ↓ YES
conv1 matches?          NO → conv1_w/conv1_b export order or values wrong (Ch. 5)
  ↓ YES
relu1 matches?          NO → ReLU applied to wrong buffer, or applied twice, or skipped
  ↓ YES
conv2 matches?          NO → same as conv1, but for conv2_w/conv2_b specifically
  ↓ YES
...continue layer by layer...
  ↓ YES (all the way to pool2/flatten)
logits match?           NO → fc_w/fc_b export order, or in_features miscount (should be 1568)
  ↓ YES
DONE — argmax agreement is now a consequence of real numerical agreement, not luck.
```
Always fix the *first* divergence, then re-run the whole comparison — a bug at `conv2` makes everything after it "wrong" too, but only `conv2` is the actual bug; debugging `pool2` first would be chasing a symptom.

### The comparator — `c/tools/compare_stages.py`

```python
"""Compare PyTorch stage dump against C stage dump."""
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^(\w+)\s+shape=\(([^)]+)\)\s+first 5=\[([^\]]+)\]"
)

def parse(path):
    stages = {}
    for line in Path(path).read_text().splitlines():
        m = LINE_RE.match(line)
        if not m: continue
        label = m.group(1)
        shape = tuple(int(x.strip()) for x in m.group(2).split(",") if x.strip())
        vals = [float(x.strip()) for x in m.group(3).split(",") if x.strip()]
        stages[label] = (shape, vals)
    return stages

def main():
    TOL = 1e-4
    py = parse("notes/pytorch_stages.txt")
    c  = parse("notes/c_stages.txt")

    ok = True
    for label in py:
        if label not in c:
            print(f"MISSING in C: {label}"); ok = False; continue
        py_shape, py_vals = py[label]
        c_shape,  c_vals  = c[label]
        if py_shape != c_shape:
            print(f"SHAPE MISMATCH {label}: py={py_shape} c={c_shape}")
            ok = False; continue
        diffs = [abs(a - b) for a, b in zip(py_vals, c_vals)]
        max_diff = max(diffs)
        status = "PASS" if max_diff < TOL else "FAIL"
        if max_diff >= TOL: ok = False
        print(f"{label:8s} shape={py_shape}  max_diff={max_diff:.2e}  {status}")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
```

Line by line:
- **The regex** — `^(\w+)\s+shape=\(([^)]+)\)\s+first 5=\[([^\]]+)\]` captures (label, shape, values).
- **The list comprehensions** — handle the trailing comma in single-element shapes like `(10,)`.
- **`sys.exit(0 if ok else 1)`** — CI-friendly exit code.

### Tolerance
Report per-stage: shape equality (hard requirement, not a tolerance), max absolute error, mean absolute error, and a PASS/FAIL against a stated threshold — `~1e-4`–`1e-5` max-abs-error is ordinary floating-point summation-order noise between PyTorch's kernels and this project's plain loops; anything larger, especially a wrong sign or order of magnitude, is a real bug. Never accept "floating point" as an explanation for a gap bigger than that.

### Build and run
```bash
gcc -Wall -Wextra -Wpedantic -std=c11 -Ic/include benchmark/verify.c c/src/nn.c -o build/verify -lm
./build/verify models/weights.bin benchmark/debug_input.bin
```
(No CMake target for this yet — Chapter 3 flagged that `tests`/`benchmark` aren't in `CMakeLists.txt`. Add one before relying on this long-term; see Chapter 9/11.)

### Expected result
**No verified output exists yet** — this requires your real `weights.bin` and a real `debug_input.bin`, neither of which I have. Labeling this explicitly rather than inventing plausible-looking numbers.

### Definition of done
Every stage from `input` to `logits` reports PASS at the stated tolerance, for at least one real MNIST test image, against your actual trained weights.

### Next
Chapter 7 — even a numerically-perfect C forward pass is only as good as the preprocessing feeding it; that's a separate risk, checked next.

---

## Chapter 7 — Preprocessing and domain shift

### Objective
Confirm what `canvas_to_mnist_input` (in `ui.c`) actually does, and whether it matches the training-side preprocessing closely enough.

### Current state — this changed significantly and is worth reading carefully

Your `canvas_to_mnist_input` is **not** a plain box-filter downsample anymore. It:
1. Scans the full 280×280 canvas for pixels above `threshold=0.02f`, finds the bounding box (`min_x,min_y,max_x,max_y`).
2. Returns an all-zero 28×28 input if the canvas is empty (`max_x < 0`) — a real, sensible guard.
3. Takes `side = max(box_w, box_h)` — a square crop preserving aspect ratio — and adds a margin: `margin = side/10; side += 2*margin`.
4. Bilinearly resamples that square crop into a **20×20** region (`sample_bilinear`), not directly into 28×28.
5. Centers the 20×20 region inside the 28×28 output with a fixed 4-pixel border (`offset = (28-20)/2 = 4`).

### The preprocessing layer `ui.h` in full

```c
#ifndef UI_H
#define UI_H

#include <stddef.h>
#include <string.h>
#include <math.h>

#define CANVAS_SIZE 280
#define MNIST_SIZE 28
#define BRUSH_RADIUS 12.0f
#define BRUSH_STRENGTH 0.85f

typedef struct {
    float pixels[CANVAS_SIZE * CANVAS_SIZE];
    int predicted_digit;
    float confidence;
    float probs[10];
    int has_prediction;
    float last_mouse_x;
    float last_mouse_y;
    int is_drawing;
} AppState;

void canvas_clear(AppState *app);
void canvas_draw_line(AppState *app, float x1, float y1, float x2, float y2);
void canvas_draw_point(AppState *app, float px, float py);
void canvas_to_mnist_input(const AppState *app, float *out28x28);

#endif
```

### Line-by-line of `ui.h`

**`#include <string.h>`** — provides `memset`, used in `canvas_clear`.

**`#include <math.h>`** — provides `sqrtf`, `fmaxf`, `cosf`, `sinf`, `fminf`, `fmaxf`, all used in `ui.c`.

**`#define CANVAS_SIZE 280`** — the on-screen drawing area is 280×280 pixels. Why 280? Because it's exactly 10× the MNIST digit size (28). That makes the downsampling a clean integer-box-average: every 10×10 block of canvas pixels becomes exactly one 28×28 input pixel.

**`#define MNIST_SIZE 28`** — the model expects 28×28.

**`#define BRUSH_RADIUS 12.0f`** — the brush is a circle of radius 12 canvas pixels. After the 10× downsample, the stroke is roughly 2.4 pixels wide — in the same range as MNIST's 1–3 pixel strokes.

**`#define BRUSH_STRENGTH 0.85f`** — the maximum brush intensity. Not 1.0, so overlapping strokes don't immediately saturate to pure white.

**`typedef struct { ... } AppState;`** — the application state.

**`float pixels[CANVAS_SIZE * CANVAS_SIZE];`** — the canvas itself. 280 × 280 = 78,400 floats = 313,600 bytes. `AppState` lives on the stack in `main.c` (313 KB is well under typical 8 MB stack limits).

**`int predicted_digit;`** — the last prediction (-1 means "no prediction yet").

**`float confidence;`** — the softmax probability of the predicted digit, in [0, 1].

**`float probs[10];`** — all ten probabilities, cached so the bar chart can be drawn without recomputing.

**`int has_prediction;`** — a flag. `0` before any prediction, `1` after.

**`float last_mouse_x, last_mouse_y;`** — the previous mouse position, used by `canvas_draw_line` to draw a continuous stroke. Without this, fast mouse movements would produce gaps.

**`int is_drawing;`** — a flag tracking whether a stroke is in progress. Set to 1 on mouse-down, 0 on mouse-up.

### The preprocessing implementation `ui.c` in full

```c
#include "../include/ui.h"

void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
    app->has_prediction = 0;
    memset(app->probs, 0, sizeof(app->probs));
    app->is_drawing = 0;
}

static void draw_circle_brush(AppState *app, float cx, float cy,
                              float radius, float strength) {
    int r = (int)(radius + 1);
    int center_x = (int)cx;
    int center_y = (int)cy;

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = center_x + dx;
            int y = center_y + dy;
            if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > radius) continue;

            float falloff = 1.0f - (dist * dist) / (radius * radius);
            float final_strength = strength * falloff;
            float *pixel = &app->pixels[y * CANVAS_SIZE + x];
            *pixel = fmaxf(*pixel, final_strength);
        }
    }
}

void canvas_draw_point(AppState *app, float px, float py) {
    draw_circle_brush(app, px, py, BRUSH_RADIUS, BRUSH_STRENGTH);
}

void canvas_draw_line(AppState *app, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < 0.1f) {
        canvas_draw_point(app, x1, y1);
        return;
    }

    int steps = (int)(dist * 1.5f) + 1;
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        draw_circle_brush(app, x1 + dx * t, y1 + dy * t,
                          BRUSH_RADIUS * 0.8f, BRUSH_STRENGTH);
    }
}

static float sample_bilinear(const float *src, int width, int height,
                             float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > width - 1) x = (float)(width - 1);
    if (y > height - 1) y = (float)(height - 1);

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1 < width ? x0 + 1 : x0;
    int y1 = y0 + 1 < height ? y0 + 1 : y0;
    float fx = x - x0;
    float fy = y - y0;

    float a = src[y0 * width + x0];
    float b = src[y0 * width + x1];
    float c = src[y1 * width + x0];
    float d = src[y1 * width + x1];
    float top = a + (b - a) * fx;
    float bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

void canvas_to_mnist_input(const AppState *app, float *out28x28) {
    memset(out28x28, 0, sizeof(float) * MNIST_SIZE * MNIST_SIZE);

    const float threshold = 0.02f;
    int min_x = CANVAS_SIZE, min_y = CANVAS_SIZE;
    int max_x = -1, max_y = -1;

    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            float v = app->pixels[y * CANVAS_SIZE + x];
            if (v > threshold) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < 0 || max_y < 0) return;

    int box_w = max_x - min_x + 1;
    int box_h = max_y - min_y + 1;
    int side = box_w > box_h ? box_w : box_h;

    int margin = side / 10;
    side += 2 * margin;
    int center_x = (min_x + max_x) / 2;
    int center_y = (min_y + max_y) / 2;
    int crop_x = center_x - side / 2;
    int crop_y = center_y - side / 2;

    const int target = 20;
    const int offset = (MNIST_SIZE - target) / 2;

    for (int oy = 0; oy < target; oy++) {
        for (int ox = 0; ox < target; ox++) {
            float src_x = crop_x + ((ox + 0.5f) * side / target) - 0.5f;
            float src_y = crop_y + ((oy + 0.5f) * side / target) - 0.5f;
            out28x28[(oy + offset) * MNIST_SIZE + (ox + offset)] =
                sample_bilinear(app->pixels, CANVAS_SIZE, CANVAS_SIZE,
                                src_x, src_y);
        }
    }
}
```

### Line-by-line of `ui.c`

**`canvas_clear`** — resets everything. `memset` for the pixel buffer and `probs` array (large, and setting them to zero bytewise is fastest). Individual field assignments for the scalars.

**`draw_circle_brush`** — draws a soft circle.
- **`int r = (int)(radius + 1);`** — the bounding box radius. Add 1 to include pixels at exactly the radius.
- **The double loop** iterates over the square `[cx-r, cx+r] × [cy-r, cy+r]`. Iterating a square and checking `dist > radius` is faster than iterating a circle's shape directly.
- **`if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;`** — the bounds check. Without this, drawing near a corner would write out of bounds. ASan catches this if you forget.
- **`float dist = sqrtf((float)(dx * dx + dy * dy));`** — Euclidean distance from the brush center.
- **`if (dist > radius) continue;`** — reject corner pixels of the square that are outside the circle.
- **`float falloff = 1.0f - (dist * dist) / (radius * radius);`** — a quadratic falloff: 1.0 at the center, 0 at the edge. Quadratic looks like a soft brush; linear looks flatter.
- **`*pixel = fmaxf(*pixel, final_strength);`** — take the max with the current value. This is what prevents overlapping strokes from darkening.

**`canvas_draw_point`** — a convenience wrapper around `draw_circle_brush`.

**`canvas_draw_line`** — draws a continuous stroke.
- **`float dx = x2 - x1; float dy = y2 - y1;`** — direction vector.
- **`float dist = sqrtf(dx * dx + dy * dy);`** — length.
- **`if (dist < 0.1f)`** — if the two points are nearly identical, just draw a point.
- **`int steps = (int)(dist * 1.5f) + 1;`** — the number of intermediate brush positions. 1.5× the distance ensures no gaps.
- **The loop** — draws a brush at each interpolated position. `t` goes from 0 to 1.
- **`BRUSH_RADIUS * 0.8f`** — the intermediate brushes are slightly smaller than a single click's brush. This avoids the line being thicker than a single point stroke.

**`sample_bilinear`** — bilinear interpolation.
- **The clamping** at the top ensures the coordinates are within the source image.
- **`int x0 = (int)x; int y0 = (int)y;`** — the integer floor.
- **`int x1 = x0 + 1 < width ? x0 + 1 : x0;`** — the next integer coordinate. If `x0` is already at the edge, `x1 = x0`.
- **`float fx = x - x0; float fy = y - y0;`** — the fractional parts, in [0, 1).
- **`a, b, c, d`** — the four surrounding pixels.
- **`float top = a + (b - a) * fx;`** — linear interpolation along the top edge.
- **`float bottom = c + (d - c) * fx;`** — same along the bottom.
- **`return top + (bottom - top) * fy;`** — linear interpolation between top and bottom.

**`canvas_to_mnist_input`** — the main preprocessing function.
- **`memset(out28x28, 0, ...)`** — zero the output. Any pixel not explicitly written stays 0.
- **`const float threshold = 0.02f;`** — pixel values above this are "drawn." Values below are background.
- **The bounding-box scan** — finds min/max x/y where pixel > threshold.
- **`if (max_x < 0 || max_y < 0) return;`** — empty canvas. All-zero input is returned.
- **`int side = box_w > box_h ? box_w : box_h;`** — the side length of the square crop. Taking the max preserves aspect ratio.
- **`int margin = side / 10; side += 2 * margin;`** — a 10% margin around the digit.
- **`int center_x = (min_x + max_x) / 2; int center_y = (min_y + max_y) / 2;`** — the center of the bounding box.
- **`int crop_x = center_x - side / 2; int crop_y = center_y - side / 2;`** — the top-left corner of the square crop.
- **`const int target = 20; const int offset = (MNIST_SIZE - target) / 2;`** — the digit occupies a 20×20 region centered in the 28×28 output. `offset = (28-20)/2 = 4`.
- **The resampling loop** — for each output pixel in the 20×20 region, compute the corresponding source coordinate via inverse mapping. The `+0.5f -0.5f` pattern converts from "pixel corner" to "pixel center" convention.
- **`sample_bilinear(...)`** — sample.
- **`out28x28[(oy + offset) * MNIST_SIZE + (ox + offset)] = ...`** — write to the correct position in the 28×28 output (offset by 4 in both directions).

### Theory: why this specific design
Real MNIST's own generation process normalizes each digit into a 20×20 bounding box (preserving aspect ratio) and centers it in a 28×28 field — your code's `target=20` and `offset=4` reproduce that convention directly, not by coincidence. This is a materially closer match to MNIST's actual preprocessing than a plain 10:1 box-filter downsample of the raw canvas would be, and is a genuine improvement over the earlier, simpler design.

### Where it still might not match, and needs checking, not assuming
- **Centering method:** your code centers by **bounding-box center** (`(min_x+max_x)/2`, `(min_y+max_y)/2`). Real MNIST centers by **center of mass** of the ink (a pixel-value-weighted centroid) — for a symmetric digit these coincide closely; for an asymmetric one (e.g. a "7" with a long diagonal stroke concentrated to one side), they can differ by a few pixels. This is a real, specific, checkable hypothesis for any accuracy gap you see on asymmetric digits — not confirmed to matter yet, just named as the first thing to check if predictions on certain digit shapes are worse than others.
- **Interpolation:** `sample_bilinear` — bilinear, not the interpolation PyTorch/PIL would use if MNIST's own images had ever needed resizing (they don't; MNIST ships pre-rendered at 28×28, so there's no PyTorch-side resize to compare against at all here — this is purely an artifact of your UI drawing at higher resolution than the model expects, not a training-preprocessing mismatch).
- **Threshold (`0.02f`):** an implicit assumption about what counts as "drawn" vs. background noise — reasonable, but arbitrary; worth knowing it exists if a very faint stroke ever gets treated as an empty canvas.

### Deterministic fixtures to build (not yet present, per `tests/` being **[UNCONFIRMED]**)

The fixture tool — `c/tools/preprocessing_fixtures.c`:

```c
#include "ui.h"
#include <stdio.h>
#include <math.h>

static void print_ascii(const float *img28) {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float v = img28[y * 28 + x];
            char c = ' ';
            if (v > 0.5f) c = '#';
            else if (v > 0.2f) c = '+';
            else if (v > 0.05f) c = '.';
            putchar(c);
        }
        putchar('\n');
    }
}

static void fixture(const char *name, void (*draw)(AppState *)) {
    AppState app;
    canvas_clear(&app);
    draw(&app);
    float out[28 * 28];
    canvas_to_mnist_input(&app, out);
    printf("\n=== %s ===\n", name);
    print_ascii(out);
}

static void draw_empty(AppState *app) { (void)app; }

static void draw_center_line(AppState *app) {
    for (int y = 40; y < 240; y++) canvas_draw_point(app, 140, y);
}

static void draw_corner_line(AppState *app) {
    for (int y = 20; y < 100; y++) canvas_draw_point(app, 40, y);
}

static void draw_small_line(AppState *app) {
    for (int y = 130; y < 150; y++) canvas_draw_point(app, 140, y);
}

static void draw_diagonal(AppState *app) {
    for (int i = 0; i < 200; i++) canvas_draw_point(app, 40 + i, 40 + i);
}

static void draw_circle(AppState *app) {
    for (int a = 0; a < 360; a += 3) {
        float rad = a * 3.14159f / 180.0f;
        canvas_draw_point(app, 140 + 80 * cosf(rad), 140 + 80 * sinf(rad));
    }
}

int main(void) {
    fixture("empty",        draw_empty);
    fixture("center line",  draw_center_line);
    fixture("corner line",  draw_corner_line);
    fixture("small line",   draw_small_line);
    fixture("diagonal",     draw_diagonal);
    fixture("circle",       draw_circle);
    return 0;
}
```

The MNIST ASCII comparison — `python/mnist_ascii.py`:

```python
"""Print 10 MNIST samples as ASCII for visual comparison."""
import numpy as np
from torchvision import datasets
from pathlib import Path

mnist = datasets.MNIST(Path(__file__).parent.parent / "data",
                       train=False, download=True)

for i in range(10):
    x, y = mnist[i]
    arr = np.asarray(x, dtype=np.float32) / 255.0
    print(f"\n=== MNIST sample {i} (label {y}) ===")
    for row in arr:
        line = ""
        for v in row:
            if v > 0.5:    line += "#"
            elif v > 0.2:  line += "+"
            elif v > 0.05: line += "."
            else:          line += " "
        print(line)
```

### Build and run fixtures

```bash
cd c
gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
    tools/preprocessing_fixtures.c src/ui.c -o fixtures -lm
./fixtures > ../notes/fixtures.txt
```

### Definition of done
Fixtures exist and are inspectable; centering method's effect on asymmetric digits is either confirmed negligible or replaced with center-of-mass; behavior on blank/tiny/huge/off-center inputs is explicitly tested, not just "seems to work" from one manual draw.

### Next
Chapter 8 — the UI this preprocessing feeds.

---

## Chapter 8 — The Raylib C product

C/Raylib is the runtime — this chapter audits `main.c`, doesn't rebuild it.

### What's already built (confirmed from your `main.c`)
- **Draw → clear → predict** — mouse drawing via `canvas_draw_point`/`canvas_draw_line` (continuous strokes tracked via `AppState.is_drawing`/`last_mouse_x`/`last_mouse_y`), `C` key or CLEAR button, `Enter` key or PREDICT button.
- **A real fix already made:** buttons use `IsMouseButtonPressed`, not `IsMouseButtonDown` — your own comment explains why (`Down` would re-trigger the action every frame while held; `Pressed` fires once). Drawing correctly still uses `Down`, since a brush stroke *should* continue while dragging — this distinction is deliberate and correct, not an inconsistency to "fix."
- **Ten-class probabilities** — `draw_probability_bars`, already color-coded red→green by `prob_color(p)`.
- **Confidence display** — green >80%, yellow >50%, red otherwise.
- **FPS counter** — `DrawFPS`.
- **Preprocessing preview** — *not present*: the canvas shown is the raw 280×280 drawing, not the actual 28×28 tensor `model_forward` receives. This is the most useful missing piece for debugging Chapter 7's centering/cropping behavior visually — see Definition of Done below.
- **Debug mode / activation visualization** — not present (Chapter 12).

### Files
`c/src/main.c` (event loop, rendering, `softmax`, `run_prediction`, `prob_color`, `draw_probability_bars`), calling into `ui.c` (canvas) and `nn.c` (inference).

### The Raylib application `main.c` in full

```c
#include "raylib.h"
#include "../include/nn.h"
#include "../include/ui.h"
#include <math.h>
#include <stdio.h>

#define WINDOW_W 900
#define WINDOW_H 600
#define CANVAS_X 50
#define CANVAS_Y 80
#define CANVAS_SIZE 280
#define BUTTON_W 100
#define BUTTON_H 50
#define BAR_CHART_X 380
#define BAR_CHART_Y 100
#define BAR_CHART_W 200
#define BAR_CHART_H 300

typedef struct {
    Rectangle rect;
    const char *label;
    Color color;
} Button;

static Color prob_color(float p) {
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    unsigned char r = (unsigned char)(255 * (1.0f - p));
    unsigned char g = (unsigned char)(255 * p);
    return (Color){r, g, 30, 255};
}

static void softmax(const float *logits, float *probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

static void draw_probability_bars(const float *probs, int x, int y, int w, int h) {
    int bar_w = w / 10;
    int max_h = h - 30;

    DrawLine(x, y + max_h, x + w, y + max_h, LIGHTGRAY);
    DrawLine(x, y, x, y + max_h, LIGHTGRAY);

    for (int i = 0; i < 10; i++) {
        int bar_x = x + i * bar_w + 2;
        int bar_h = (int)(probs[i] * max_h);
        int bar_y = y + max_h - bar_h;
        Color c = prob_color(probs[i]);

        DrawRectangle(bar_x, bar_y, bar_w - 4, bar_h, c);

        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        DrawText(label, bar_x + 2, y + max_h + 5, 12, DARKGRAY);
    }
}

static void run_prediction(AppState *app, const CnnModel *model) {
    float mnist_input[MNIST_SIZE * MNIST_SIZE];
    canvas_to_mnist_input(app, mnist_input);

    Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);
    for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
        input.data[i] = mnist_input[i];
    }

    float logits[10];
    model_forward(model, &input, logits);
    tensor_free(&input);

    softmax(logits, app->probs, 10);
    app->predicted_digit = argmax(logits, 10);
    app->confidence = app->probs[app->predicted_digit];
    app->has_prediction = 1;
}

int main(void) {
    CnnModel model;
    int model_ok = (model_load(&model, "models/weights.bin") == 0);
    if (!model_ok) {
        fprintf(stderr, "Warning: could not load models/weights.bin\n");
    }

    InitWindow(WINDOW_W, WINDOW_H, "Number Guesser Pro");
    SetTargetFPS(60);

    AppState app;
    canvas_clear(&app);

    Rectangle canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};
    Button clear_btn = {
        {CANVAS_X, CANVAS_Y + CANVAS_SIZE + 20, BUTTON_W, BUTTON_H},
        "CLEAR", LIGHTGRAY
    };
    Button predict_btn = {
        {CANVAS_X + BUTTON_W + 20, CANVAS_Y + CANVAS_SIZE + 20, BUTTON_W, BUTTON_H},
        "PREDICT", model_ok ? SKYBLUE : GRAY
    };

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            if (CheckCollisionPointRec(mouse, clear_btn.rect)) {
                canvas_clear(&app);
            } else if (model_ok && CheckCollisionPointRec(mouse, predict_btn.rect)) {
                run_prediction(&app, &model);
            }
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) &&
            CheckCollisionPointRec(mouse, canvas_rect)) {
            float px = mouse.x - CANVAS_X;
            float py = mouse.y - CANVAS_Y;

            if (!app.is_drawing) {
                app.is_drawing = 1;
                app.last_mouse_x = px;
                app.last_mouse_y = py;
                canvas_draw_point(&app, px, py);
            } else {
                canvas_draw_line(&app, app.last_mouse_x, app.last_mouse_y, px, py);
                app.last_mouse_x = px;
                app.last_mouse_y = py;
            }
        } else {
            app.is_drawing = 0;
        }

        if (IsKeyPressed(KEY_C)) canvas_clear(&app);
        if (IsKeyPressed(KEY_ENTER) && model_ok) run_prediction(&app, &model);

        BeginDrawing();
        ClearBackground(GetColor(0x1a1a2eFF));

        DrawText("Draw a Digit", 20, 10, 28, RAYWHITE);
        DrawText("Press 'C' to clear | Enter to predict", 20, 45, 16, LIGHTGRAY);

        DrawRectangleRec(canvas_rect, BLACK);
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                float v = app.pixels[y * CANVAS_SIZE + x];
                if (v > 0.01f) {
                    unsigned char g = (unsigned char)(v * 255.0f);
                    DrawPixel(CANVAS_X + x, CANVAS_Y + y, (Color){g, g, g, 255});
                }
            }
        }
        DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);

        DrawRectangleRec(clear_btn.rect, clear_btn.color);
        DrawText(clear_btn.label, (int)(clear_btn.rect.x + 25),
                 (int)(clear_btn.rect.y + 15), 18, BLACK);

        DrawRectangleRec(predict_btn.rect, predict_btn.color);
        DrawText(predict_btn.label, (int)(predict_btn.rect.x + 15),
                 (int)(predict_btn.rect.y + 15), 18, BLACK);

        if (app.has_prediction) {
            draw_probability_bars(app.probs, BAR_CHART_X, BAR_CHART_Y,
                                  BAR_CHART_W, BAR_CHART_H);

            char buf[128];
            snprintf(buf, sizeof(buf), "Prediction: %d", app.predicted_digit);
            DrawText(buf, BAR_CHART_X, BAR_CHART_Y + BAR_CHART_H + 30, 28, RAYWHITE);

            snprintf(buf, sizeof(buf), "Confidence: %.1f%%", app.confidence * 100.0f);
            Color conf_color = (app.confidence > 0.8f) ? GREEN :
                               (app.confidence > 0.5f) ? YELLOW : RED;
            DrawText(buf, BAR_CHART_X, BAR_CHART_Y + BAR_CHART_H + 60, 18, conf_color);
        } else if (!model_ok) {
            DrawText("No model loaded", BAR_CHART_X, 200, 18, MAROON);
        } else {
            DrawText("Draw a digit and press PREDICT", BAR_CHART_X, 200, 18, GRAY);
        }

        DrawText("Canvas: 280x280 -> MNIST-style 28x28", CANVAS_X,
                 CANVAS_Y + CANVAS_SIZE + BUTTON_H + 60, 12, GRAY);
        DrawFPS(WINDOW_W - 80, 10);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
```

### Line-by-line of `main.c`

**`#include "raylib.h"`** — the library.

**`#include "../include/nn.h"`** and **`#include "../include/ui.h"`** — the project's own headers.

**`#define WINDOW_W 900`** and **`#define WINDOW_H 600`** — the window dimensions. Wide enough for both a canvas and a bar chart.

**`#define CANVAS_X 50`, `CANVAS_Y 80`, `CANVAS_SIZE 280`** — the canvas position and size.

**`#define BAR_CHART_X 380`, etc.** — the bar chart position and size. At `BAR_CHART_X = 380` with `BAR_CHART_W = 200`, the chart ends at `x = 580`. The window is 900 wide, so there's ~320 px of margin on the right. This is where the activation visualization would go (Chapter 12).

**`typedef struct { Rectangle rect; const char *label; Color color; } Button;`** — a button abstraction. `Rectangle` from raylib is `{x, y, width, height}`.

**`static Color prob_color(float p)`** — maps a probability to a color. Red at `p=0`, green at `p=1`, with a hint of blue (`b=30`).

- **`if (p < 0) p = 0; if (p > 1) p = 1;`** — clamp. `p` should always be in [0,1], but clamping is defensive.
- **`(unsigned char)(255 * (1.0f - p))`** — red is the complement of the probability.

**`static void softmax(const float *logits, float *probs, int n)`** — converts logits to probabilities.

- **`float max_val = logits[0]; for ...`** — find the maximum logit. This is the numerical stability trick.
- **`probs[i] = expf(logits[i] - max_val);`** — subtract max before exponentiating. Since `exp` overflows quickly (around `exp(88)` for float32), subtracting the max ensures the largest exponent is `exp(0) = 1`. The final probabilities are identical because softmax is invariant to adding a constant to all logits.
- **`sum += probs[i];`** — accumulate the normalization denominator.
- **`probs[i] /= sum;`** — normalize.

**`static void draw_probability_bars(...)`** — draws the ten probability bars.

- **`int bar_w = w / 10;`** — each bar occupies 1/10 of the total width.
- **`int max_h = h - 30;`** — leave 30 px at the bottom for the digit labels.
- **`DrawLine(x, y + max_h, x + w, y + max_h, LIGHTGRAY);`** — x-axis.
- **`DrawLine(x, y, x, y + max_h, LIGHTGRAY);`** — y-axis.
- **`int bar_x = x + i * bar_w + 2;`** — the bar's left edge with 2 px padding.
- **`int bar_h = (int)(probs[i] * max_h);`** — the bar's height, proportional to the probability.
- **`int bar_y = y + max_h - bar_h;`** — the bar's top edge. Since screen y increases downward, taller bars have lower `y`.
- **`DrawRectangle(bar_x, bar_y, bar_w - 4, bar_h, c);`** — the bar, with 4 px gap between adjacent bars.

**`static void run_prediction(...)`** — the main prediction entry point.

- **`float mnist_input[MNIST_SIZE * MNIST_SIZE];`** — the 28×28 input, on the stack (784 floats = 3136 bytes).
- **`canvas_to_mnist_input(app, mnist_input);`** — run preprocessing.
- **`Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);`** — allocate the tensor. Shape `(1, 28, 28)` — a single-channel image.
- **The copy loop** — copy the 784 floats from the stack array into the tensor's data buffer.
- **`model_forward(model, &input, logits);`** — run the CNN.
- **`tensor_free(&input);`** — free immediately after the forward pass.
- **`softmax(logits, app->probs, 10);`** — convert logits to probabilities.
- **`app->predicted_digit = argmax(logits, 10);`** — the predicted class.
- **`app->confidence = app->probs[app->predicted_digit];`** — the probability of the predicted class.

**`int main(void)`** — the application entry point.

- **`CnnModel model;`** — stack-allocated. `sizeof(CnnModel) == 175016`.
- **`int model_ok = (model_load(&model, "models/weights.bin") == 0);`** — load the weights. The relative path `"models/weights.bin"` is relative to the current working directory, not the binary's location.
- **`if (!model_ok) { fprintf(stderr, ...); }`** — warning, no crash.
- **`InitWindow(WINDOW_W, WINDOW_H, "Number Guesser Pro");`** — create the window.
- **`SetTargetFPS(60);`** — 60 frames per second.
- **`AppState app; canvas_clear(&app);`** — initialize the state.
- **`Rectangle canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};`** — the canvas hitbox.
- **`Button clear_btn = { { ... }, "CLEAR", LIGHTGRAY };`** — the CLEAR button.
- **`Button predict_btn = { { ... }, "PREDICT", model_ok ? SKYBLUE : GRAY };`** — the PREDICT button, colored based on whether the model loaded.

**The event loop:**

- **`while (!WindowShouldClose())`** — runs until the window is closed or ESC is pressed.
- **`Vector2 mouse = GetMousePosition();`** — the mouse position.
- **`if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))`** — the mouse was just pressed this frame. Correct primitive for buttons.
- **`if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, canvas_rect))`** — the mouse is currently down and inside the canvas. Correct for drawing.
- **`if (!app.is_drawing)`** — first frame of a stroke. Set `is_drawing`, record the position, draw a point.
- **`else`** — subsequent frames. Draw a line from the previous position to the current one.
- **`app.is_drawing = 0;`** — outside the `if`. This ends the stroke.
- **`if (IsKeyPressed(KEY_C)) canvas_clear(&app);`** — keyboard shortcut.
- **`if (IsKeyPressed(KEY_ENTER) && model_ok) run_prediction(&app, &model);`** — keyboard shortcut.

**Drawing:**

- **`BeginDrawing(); ClearBackground(GetColor(0x1a1a2eFF));`** — clear the screen. `0x1a1a2eFF` is a dark blue-black.
- **`DrawText(...)`** — titles and hints.
- **`DrawRectangleRec(canvas_rect, BLACK);`** — the canvas background.
- **The pixel loop** — for each canvas pixel, if value > 0.01, draw a grayscale pixel.
- **`DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);`** — the canvas border.
- **The button rectangles and labels.**
- **The prediction display** — bar chart, prediction text, confidence text with color coding.
- **`DrawFPS(WINDOW_W - 80, 10);`** — FPS counter in the top-right.

### The preprocessing preview panel (missing — add it)

```c
static void draw_mnist_preview(const float *mnist_input,
                                int x, int y, int size) {
    int cell = size / 28;
    for (int py = 0; py < 28; py++) {
        for (int px = 0; px < 28; px++) {
            float v = mnist_input[py * 28 + px];
            v = fmaxf(0.0f, fminf(1.0f, v));
            unsigned char g = (unsigned char)(v * 255.0f);
            DrawRectangle(x + px * cell, y + py * cell,
                          cell, cell,
                          (Color){ g, g, g, 255 });
        }
    }
    DrawRectangleLines(x, y, size, size, DARKGRAY);
    DrawText("28x28 model input", x, y - 15, 12, RAYWHITE);
}
```

In `run_prediction`, save a copy:

```c
static float last_mnist_input[28 * 28];
...
memcpy(last_mnist_input, mnist_input, sizeof(last_mnist_input));
```

In the drawing section:

```c
draw_mnist_preview(last_mnist_input, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 90, 112);
```

### Definition of done for this chapter
Add a small "preprocessing preview" panel — render the actual 28×28 `float` array `run_prediction` computes (before it's consumed by `tensor_alloc`/`model_forward`) as a small grayscale tile next to the canvas. Cheap (reuses the same `DrawPixel` pattern already used for the canvas itself) and directly answers "is my centering/cropping doing what I think it's doing" without needing to save files and inspect them out-of-band.

### Next
Chapter 9 — none of Chapters 4–8's claims are protected against regression without tests.

---

## Chapter 9 — Tests

**[UNCONFIRMED]** — `tests/` exists per your README but wasn't provided, and `CMakeLists.txt` doesn't build it regardless. Designing against your *actual* confirmed function signatures (`tensor_info` not `tensor_print_summary`, no `WEIGHTS_FILE_BYTES` constant — `sizeof(CnnModel)` instead).

| Test | Protects against |
|---|---|
| `tensor_get`/`set` round-trip + neighbor-unaffected check | A wrong offset formula *aliasing* two distinct cells — silent corruption, not a crash |
| `conv2d`, no padding, hand-computed 3×3 input / 2×2 kernel | Basic accumulation/loop-nest arithmetic |
| `conv2d`, `k=3,s=1,p=1`, center-tap-only kernel | Bounds-check/padding logic specifically, using this project's actual conv config |
| `maxpool2d`, hand-picked 4×4 → 2×2 | Window placement + max selection, including verifying `-INFINITY` behaves correctly on an all-negative window (a case a `0` sentinel would get wrong) |
| `model_load`, synthetic weights (`float[i]=i`) | Wrong read order/count — `conv1_b[0]==288.0` proves `conv1_w`'s 288 floats and `conv1_b` don't overlap or gap |
| `model_load`, wrong-size file | The `sizeof(CnnModel)` guard actually rejects, not just "looks like it should" |
| `canvas_to_mnist_input`, blank/tiny/off-center/huge fixtures (Ch. 7) | Your new bounding-box+bilinear logic specifically — the old simple box-filter tests no longer apply to this code |
| End-to-end: real MNIST image → `model_forward` → correct digit | Wiring-level regressions across the whole pipeline |
| Benchmark/parity (Ch. 6) | The one thing unit tests structurally can't catch: correct-looking code computing the wrong number |

### Full `c/tests/test_nn.c`

```c
#include "nn.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int pass = 0, fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { printf("  [ok]   %s\n", name); pass++; } \
    else      { printf("  [FAIL] %s\n", name); fail++; } \
} while (0)

#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f)

static void test_tensor(void) {
    printf("test_tensor\n");
    Tensor t = tensor_alloc(2, 3, 3);
    CHECK(CLOSE(tensor_get(&t, 0, 0, 0), 0.0f), "zero-init [0,0,0]");
    CHECK(CLOSE(tensor_get(&t, 1, 2, 2), 0.0f), "zero-init [1,2,2]");
    tensor_set(&t, 1, 2, 0, 7.5f);
    CHECK(CLOSE(tensor_get(&t, 1, 2, 0), 7.5f), "set/get [1,2,0]");
    CHECK(CLOSE(tensor_get(&t, 1, 1, 0), 0.0f), "neighbor [1,1,0] unaffected");
    CHECK(CLOSE(tensor_get(&t, 0, 2, 0), 0.0f), "neighbor [0,2,0] unaffected");
    tensor_free(&t);
}

static void test_conv2d(void) {
    printf("test_conv2d\n");
    Tensor in = tensor_alloc(1, 3, 3);
    float data[] = {1,2,3, 4,5,6, 7,8,9};
    memcpy(in.data, data, sizeof(data));
    float W[] = {1, 0, 0, 1};
    float b[] = {0};
    Tensor out = conv2d(&in, W, b, 1, 2, 1, 0);
    CHECK(out.channels == 1 && out.height == 2 && out.width == 2, "shape (1,2,2)");
    CHECK(CLOSE(tensor_get(&out,0,0,0),  6.0f), "conv[0][0] = 6");
    CHECK(CLOSE(tensor_get(&out,0,0,1),  8.0f), "conv[0][1] = 8");
    CHECK(CLOSE(tensor_get(&out,0,1,0), 12.0f), "conv[1][0] = 12");
    CHECK(CLOSE(tensor_get(&out,0,1,1), 14.0f), "conv[1][1] = 14");
    tensor_free(&in);
    tensor_free(&out);
}

static void test_maxpool2d(void) {
    printf("test_maxpool2d\n");
    Tensor in = tensor_alloc(1, 4, 4);
    float data[] = {1,3,2,4, 5,6,1,2, 7,8,3,1, 0,2,4,9};
    memcpy(in.data, data, sizeof(data));
    Tensor out = maxpool2d(&in, 2, 2);
    CHECK(out.height == 2 && out.width == 2, "shape (1,2,2)");
    CHECK(CLOSE(tensor_get(&out,0,0,0), 6.0f), "pool[0][0] = 6");
    CHECK(CLOSE(tensor_get(&out,0,0,1), 4.0f), "pool[0][1] = 4");
    CHECK(CLOSE(tensor_get(&out,0,1,0), 8.0f), "pool[1][0] = 8");
    CHECK(CLOSE(tensor_get(&out,0,1,1), 9.0f), "pool[1][1] = 9");
    tensor_free(&in);
    tensor_free(&out);
}

static void test_maxpool2d_negative(void) {
    printf("test_maxpool2d_negative\n");
    Tensor in = tensor_alloc(1, 2, 2);
    float data[] = {-5, -3, -8, -1};
    memcpy(in.data, data, sizeof(data));
    Tensor out = maxpool2d(&in, 2, 2);
    CHECK(CLOSE(tensor_get(&out,0,0,0), -1.0f), "all-negative max = -1");
    tensor_free(&in);
    tensor_free(&out);
}

int main(void) {
    test_tensor();
    test_conv2d();
    test_maxpool2d();
    test_maxpool2d_negative();
    printf("\n%d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
```

Line by line of the critical parts:
- **`CHECK(cond, name)`** — the multi-line macro. The `do { ... } while (0)` wrapper is required so the macro expands to a single statement when used inside an `if` without braces.
- **`CLOSE(a, b)`** — tolerance-based float comparison. `1e-4` is loose enough for float32 arithmetic, tight enough to catch a real bug.
- **`test_tensor` neighbor checks** — after setting one cell, verify the immediate neighbors are still 0. This catches an offset formula that aliases two different `(c, y, x)` triples to the same memory address.
- **`test_maxpool2d_negative`** — the specific test that would catch a `0` sentinel in `maxpool2d`. If the implementation used `float best = 0.0f;` instead of `-INFINITY`, this test would fail (the max would come back as 0 instead of -1). This is exactly the case that motivated the `-INFINITY` choice in Chapter 4.

### Wiring into CMake (currently missing)

```cmake
enable_testing()

add_executable(test_nn tests/test_nn.c c/src/nn.c)
target_include_directories(test_nn PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_nn PRIVATE m)
endif()
add_test(NAME nn_tests COMMAND test_nn)

add_executable(test_ui tests/test_ui.c c/src/ui.c)
target_include_directories(test_ui PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_ui PRIVATE m)
endif()
add_test(NAME ui_tests COMMAND test_ui)
```

Line by line:
- **`enable_testing()`** — turns on CMake's test framework. Required before `add_test`.
- **`add_executable(test_nn ...)`** — a build target just like the main app. No raylib needed — `nn.c` and `test_nn.c` don't include `raylib.h`.
- **`add_test(NAME nn_tests COMMAND test_nn)`** — registers the target as a CTest test. `ctest` runs it and reports PASS/FAIL based on the exit code.

### Build and run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

**`--output-on-failure`** — shows the test's stdout/stderr only if it fails. Without this, successful tests are silent.

### Definition of done
Every row above exists as a real, compiling test file; `ctest` runs them all from a clean `cmake --build`.

### Next
Chapter 10 — memory-safety confirmation the tests above don't give you on their own (a test can pass and still corrupt memory it happened not to read back).

---

## Chapter 10 — Sanitizers

### Workflow, using your actual CMake project

```bash
cmake -S . -B build-asan \
    -DCMAKE_C_FLAGS="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

Line by line of the flags:
- **`-g`** — debug symbols. Required for readable stack traces.
- **`-O1`** — light optimization. `-O0` is too slow for some benchmarks; `-O2` can inline away the bug. `-O1` balances visibility and speed.
- **`-fsanitize=address,undefined`** — enables both sanitizers.
- **`-fno-omit-frame-pointer`** — keeps frame pointers. Without this, ASan traces can be misleading.

Or, without a CMake preset, direct `gcc` invocation for a single test binary:

```bash
gcc -Wall -Wextra -Wpedantic -std=c11 -g -fsanitize=address,undefined \
    -Ic/include tests/test_nn.c c/src/nn.c -o test_nn_asan -lm
./test_nn_asan
```

**AddressSanitizer** catches: heap-buffer-overflow (writing past a `tensor_alloc`'d buffer — most likely from a wrong index formula, which Chapter 4's `tensor_get`/`set` audit already flagged as unenforced), use-after-free (a `Tensor` used after `tensor_free` — the `t->data=NULL` defensive set in your `tensor_free` turns this into an immediate crash under ASan rather than a silent bad read), double-free, and leaks.

**UndefinedBehaviorSanitizer** catches signed overflow, misaligned access, and other UB the compiler and a plain test run won't show you.

### Reading an ASan report

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address ...
WRITE of size 4 at ...
    #0 tensor_set nn.c:42
    #1 conv2d nn.c:104
    #2 model_forward nn.c:180
    #3 main main.c:87
```

- The top frame (`#0`) is where the faulting access happened.
- Trace back up to find which caller passed the out-of-bounds coordinate.
- Fix the **caller**, not the callee, unless the callee's own formula is wrong.

### Definition of done
Every test from Chapter 9 runs clean (zero errors) under both sanitizers simultaneously.

### Next
Chapter 11 — automate Chapters 9–10 so they run on every push, not just when you remember to.

---

## Chapter 11 — CI

Build only after Chapters 9–10 are real locally — a CI job running tests that don't yet cover real-weight correctness (Ch. 6) protects less than it looks like it does.

```yaml
name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install raylib and build tools
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake libraylib-dev \
              libx11-dev libxrandr-dev libxinerama-dev \
              libxcursor-dev libxi-dev libgl1-mesa-dev
      - name: Configure
        run: |
          cmake -S . -B build \
                -DCMAKE_C_FLAGS="-Wall -Wextra -Wpedantic -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer" \
                -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build -j
      - name: Test
        run: ctest --test-dir build --output-on-failure
      - name: Parity (benchmark)
        run: |
          if [ -f models/weights.bin ] && [ -f benchmark/debug_input.bin ]; then
            ./build/verify models/weights.bin benchmark/debug_input.bin \
              | diff - benchmark/expected_output.txt
          else
            echo "weights/benchmark files not present — skipping parity"
          fi
```

Line-by-line of the workflow:
- **`on: push/pull_request`** — runs on every push and every PR.
- **`runs-on: ubuntu-latest`** — Ubuntu has `libraylib-dev` as a system package, making the CI job simple.
- **The apt install** — all the shared library dependencies raylib needs for building (X11 dev headers, GL).
- **The configure step** — uses the same sanitizer flags as Chapter 10.
- **`ctest --test-dir build --output-on-failure`** — runs every registered test.
- **The parity step** — only runs if the weights and benchmark files exist. The `diff` command fails CI if `verify`'s output doesn't match the expected output byte-for-byte. This is what makes CI actually protect against regression — a CI that lets a parity regression through defeats the point.

Parity should **fail CI** when it fails — it's not advisory. No display/raylib-runtime step needed for any of this — `number_guesser` itself doesn't need to run headlessly, only build.

### Definition of done
A fresh clone builds, tests, sanitizes, and checks parity, with zero manual steps, on every push.

### Next
Chapter 12 — now that correctness is protected, the UI's most useful addition is making the network's internals visible.

---

## Chapter 12 — Network visualization

The probability bar chart (Ch. 8) is already built. Not yet built: activation maps for `conv1`–`conv4`, and the preprocessing preview (Ch. 8's Definition of Done).

### What a feature map means, mathematically
`conv1`'s output is `32×28×28` — 32 separate `28×28` grayscale images, each one the response of one learned 3×3 filter swept across the input. Early layers (`conv1`) typically respond to edges/strokes; later layers (`conv4`) respond to more complex, less visually interpretable patterns. Rendering `conv1`'s 32 channels as a tile grid is a fast, concrete sanity check: if every tile looks like noise, something upstream (preprocessing, weight loading order, Chapter 5's/6's concerns) is almost certainly wrong — often faster to notice visually than reading raw logit numbers.

### The activation API

Add to `c/include/nn.h`:

```c
typedef struct {
    Tensor conv1, relu1;
    Tensor conv2, relu2, pool1;
    Tensor conv3, relu3;
    Tensor conv4, relu4, pool2;
    int enabled;
} Activations;

Activations activations_alloc(void);
void activations_free(Activations *a);

void model_forward_full(const CnnModel *m,
                        const Tensor *input,
                        float *logits_out,
                        Activations *acts);
```

Add to `c/src/nn.c`:

```c
Activations activations_alloc(void) {
    Activations a;
    a.conv1  = tensor_alloc(32, 28, 28);
    a.relu1  = tensor_alloc(32, 28, 28);
    a.conv2  = tensor_alloc(32, 28, 28);
    a.relu2  = tensor_alloc(32, 28, 28);
    a.pool1  = tensor_alloc(32, 14, 14);
    a.conv3  = tensor_alloc(32, 14, 14);
    a.relu3  = tensor_alloc(32, 14, 14);
    a.conv4  = tensor_alloc(32, 14, 14);
    a.relu4  = tensor_alloc(32, 14, 14);
    a.pool2  = tensor_alloc(32,  7,  7);
    a.enabled = 1;
    return a;
}

void activations_free(Activations *a) {
    tensor_free(&a->conv1);  tensor_free(&a->relu1);
    tensor_free(&a->conv2);  tensor_free(&a->relu2);
    tensor_free(&a->pool1);  tensor_free(&a->conv3);
    tensor_free(&a->relu3);  tensor_free(&a->conv4);
    tensor_free(&a->relu4);  tensor_free(&a->pool2);
    a->enabled = 0;
}

static void copy_tensor(Tensor *dst, const Tensor *src) {
    size_t n = (size_t)src->channels * src->height * src->width;
    memcpy(dst->data, src->data, n * sizeof(float));
}

void model_forward_full(const CnnModel *m, const Tensor *input,
                        float *logits_out, Activations *acts) {
    Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);
    if (acts->enabled) copy_tensor(&acts->conv1, &a);
    relu_tensor(&a);
    if (acts->enabled) copy_tensor(&acts->relu1, &a);

    Tensor b = conv2d(&a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);
    if (acts->enabled) copy_tensor(&acts->conv2, &b);
    tensor_free(&a);
    relu_tensor(&b);
    if (acts->enabled) copy_tensor(&acts->relu2, &b);

    Tensor p1 = maxpool2d(&b, 2, 2);
    if (acts->enabled) copy_tensor(&acts->pool1, &p1);
    tensor_free(&b);

    Tensor c = conv2d(&p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);
    if (acts->enabled) copy_tensor(&acts->conv3, &c);
    tensor_free(&p1);
    relu_tensor(&c);
    if (acts->enabled) copy_tensor(&acts->relu3, &c);

    Tensor d = conv2d(&c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);
    if (acts->enabled) copy_tensor(&acts->conv4, &d);
    tensor_free(&c);
    relu_tensor(&d);
    if (acts->enabled) copy_tensor(&acts->relu4, &d);

    Tensor p2 = maxpool2d(&d, 2, 2);
    if (acts->enabled) copy_tensor(&acts->pool2, &p2);
    tensor_free(&d);

    int in_features = p2.channels * p2.height * p2.width;
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);
    tensor_free(&p2);
}
```

Line by line:
- **`activations_alloc`** — allocates every intermediate tensor once. Called at startup, not per prediction.
- **`copy_tensor`** — `memcpy` because both tensors have the same shape and layout.
- **`if (acts->enabled)`** — when disabled, the copy is skipped. The cost when disabled is one branch per stage — negligible.
- **Ownership:** `acts` owns its tensors. They persist across calls and are freed by `activations_free`.

### The tile renderer

Add to `c/src/main.c`:

```c
static void draw_activation_tiles(const Tensor *t, int start_x, int start_y,
                                   int tile_size, const char *label) {
    int cols = 8;
    for (int c = 0; c < t->channels && c < 32; c++) {
        int tx = start_x + (c % cols) * (tile_size + 2);
        int ty = start_y + (c / cols) * (tile_size + 2);

        for (int y = 0; y < t->height; y++) {
            for (int x = 0; x < t->width; x++) {
                float v = tensor_get(t, c, y, x);
                v = fmaxf(0.0f, fminf(1.0f, v));
                unsigned char g = (unsigned char)(v * 255.0f);
                int px = tx + (x * tile_size) / t->width;
                int py = ty + (y * tile_size) / t->height;
                DrawPixel(px, py, (Color){g, g, g, 255});
            }
        }
        DrawRectangleLines(tx, ty, tile_size, tile_size, DARKGRAY);
    }
    DrawText(label, start_x, start_y - 15, 12, RAYWHITE);
}
```

Line by line:
- **`cols = 8`** — 8 tiles per row. For 32 channels, 4 rows.
- **`(c % cols)` and `(c / cols)`** — column and row position of tile `c`.
- **`(x * tile_size) / t->width`** — scale source x coordinate to tile x coordinate.
- **`DrawPixel`** — one call per tensor cell.

### Exposing this without unnecessary allocation
`model_forward` already frees each intermediate `Tensor` the instant the next layer has consumed it (Ch. 4) — to visualize, you need a variant that keeps (or copies) the specific intermediate you want to render *before* it's freed, not a version that keeps all of them alive simultaneously (that would multiply peak memory by ~8x for no reason). The `Activations` struct above is exactly this variant.

### Definition of done
`main.c` can render `conv1` (at minimum) as a tile grid on demand (e.g. a `V` keypress toggling a "debug view"), using real intermediate tensors from an actual `model_forward` call, not fabricated data.

### Next
Chapter 13 — once the visualization exists, it's tempting to "optimize" based on how things *feel*; measure first.

---

## Chapter 13 — Profiling

Do not optimize before this chapter has real numbers.

### What to measure
Wall-clock time for: `canvas_to_mnist_input` (preprocessing), each of the four `conv2d` calls individually, each `maxpool2d` call, `linear`, total `model_forward`, and total allocations (`tensor_alloc` call count per prediction — currently 6 per `model_forward`, per Chapter 4's ownership-chain trace).

### The timing infrastructure

Add to `c/include/nn.h` (behind `#ifdef BENCHMARK`):

```c
#ifdef BENCHMARK

typedef struct {
    double conv1, relu1;
    double conv2, relu2, pool1;
    double conv3, relu3;
    double conv4, relu4, pool2;
    double linear;
    int    calls;
} LayerTimes;

LayerTimes *bench_get_times(void);
void bench_reset(void);
void bench_report(void);

#endif
```

Add to `c/src/nn.c`:

```c
#ifdef BENCHMARK
#include <time.h>
#include <string.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static LayerTimes g_times = {0};

LayerTimes *bench_get_times(void) { return &g_times; }
void bench_reset(void) { memset(&g_times, 0, sizeof g_times); }

void bench_report(void) {
    if (g_times.calls == 0) return;
    double n = (double)g_times.calls;
    fprintf(stderr, "\n=== Per-layer timing (avg over %d calls) ===\n",
            g_times.calls);
    fprintf(stderr, "%-8s %10.3f ms\n", "conv1",  1000.0 * g_times.conv1  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "relu1",  1000.0 * g_times.relu1  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "conv2",  1000.0 * g_times.conv2  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "relu2",  1000.0 * g_times.relu2  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "pool1",  1000.0 * g_times.pool1  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "conv3",  1000.0 * g_times.conv3  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "relu3",  1000.0 * g_times.relu3  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "conv4",  1000.0 * g_times.conv4  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "relu4",  1000.0 * g_times.relu4  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "pool2",  1000.0 * g_times.pool2  / n);
    fprintf(stderr, "%-8s %10.3f ms\n", "linear", 1000.0 * g_times.linear / n);
}
#endif
```

Line by line:
- **`CLOCK_MONOTONIC`** — the correct clock for elapsed time. It never jumps backward, unlike `CLOCK_REALTIME` which NTP can adjust mid-measurement.
- **`(double)ts.tv_sec + (double)ts.tv_nsec * 1e-9`** — converts the `timespec` struct to a single `double` in seconds.
- **`static LayerTimes g_times = {0};`** — the global accumulator. Static storage duration, zero-initialized.
- **`memset(&g_times, 0, sizeof g_times)`** — zeros the whole struct in one call.
- **`1000.0 * g_times.conv1 / n`** — converts seconds to milliseconds (`× 1000`) then divides by the call count.

The instrumented `model_forward` wraps each stage with `now_seconds()` between every op.

### The benchmark driver — `c/tools/benchmark.c`

```c
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "../models/weights.bin";
    const char *input_path = argc > 2 ? argv[2] : "../models/debug_input.bin";
    int n_iters = argc > 3 ? atoi(argv[3]) : 100;
    if (n_iters <= 0) return 1;

    CnnModel m;
    if (model_load(&m, weights) != 0) return 1;

    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(input_path, "rb");
    if (f) { fread(input.data, sizeof(float), 28 * 28, f); fclose(f); }

    float logits[10];

    /* Warm-up: one call discarded, triggers lazy init. */
    model_forward(&m, &input, logits);
    bench_reset();

    double t0 = now_seconds();
    for (int i = 0; i < n_iters; i++) {
        model_forward(&m, &input, logits);
    }
    double t1 = now_seconds();

    fprintf(stderr, "=== Benchmark: %d iterations ===\n", n_iters);
    fprintf(stderr, "total   : %8.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "average : %8.3f ms\n", (t1 - t0) * 1000.0 / n_iters);
    fprintf(stderr, "throughput : %8.1f inferences/sec\n", n_iters / (t1 - t0));
    bench_report();

    tensor_free(&input);
    return 0;
}
```

Line by line:
- **The warm-up call** — the first call may be slower due to lazy page faults, cache warming, and any one-time initialization. We run it, discard it, and `bench_reset()` before the timed loop.
- **`bench_reset()`** — critical. Without it, the warm-up's timings pollute the average.
- **The timed loop** — runs the forward pass `n_iters` times.
- **The report** — prints total, average, throughput, and per-layer breakdown.

### Compile and run

```bash
cd c
gcc -O2 -DBENCHMARK -Wall -Wextra -std=c11 -Iinclude \
    tools/benchmark.c src/nn.c -o bench -lm
./bench ../models/weights.bin ../models/debug_input.bin 1000
```

### What the numbers should tell you
Chapter 4's Big-O note already predicts `conv2`/`conv3`/`conv4` (32→32 channels, ≈7.2M ops each) should dominate over `conv1` (1→32 channels, ≈226K ops) — confirm or refute this with real measurements before assuming it. If preprocessing (`canvas_to_mnist_input`) is somehow a meaningful fraction of total time, that's a different, more surprising finding worth its own investigation before touching `conv2d` at all.

### Definition of done
A real table of stage-by-stage timings, measured on your actual hardware, exists and is committed somewhere reviewable — not estimated, not assumed from the Big-O note alone.

### Next
Chapter 14 — only once this table exists does it make sense to decide what (if anything) to optimize.

---

## Chapter 14 — C optimization

Order matters, and every step requires the previous one's evidence:

1. **Buffer reuse** — if profiling (Ch. 13) shows allocation overhead is non-trivial, pre-allocate scratch tensors once (outside the per-prediction hot path) instead of `tensor_alloc`ing 6 times per prediction.
2. **Allocation reduction** — related; fewer `calloc`/`free` round-trips.
3. **Cache locality** — `conv2d`'s loop order (`oc→oy→ox→ic→ky→kx`) was chosen for a reason (Ch. 4) but hasn't been profiled against alternatives on this actual hardware.
4. **Loop ordering** — only after (3) is measured, not assumed.
5. **Compiler optimization** — `-O2`/`-O3`, measured before/after, not assumed to help by a fixed amount.
6. **SIMD** — only if profiling shows a specific inner loop dominates and is vectorizable; not a default.
7. **Parallelism** — only if a single prediction's latency is still the bottleneck after 1–6, and only if the win justifies added complexity.
8. **Quantization** — changes numerical behavior; requires re-running Chapter 6's parity check against the *quantized* output, not the float32 one — a different correctness bar, not free.

### The buffer reuse implementation

Add to `c/include/nn.h`:

```c
typedef struct {
    Tensor a, b, p1, c, d, p2;
    int initialized;
} Workspace;

void workspace_init(Workspace *w);
void workspace_free(Workspace *w);
void model_forward_ws(const CnnModel *m, const Tensor *input,
                       float *logits_out, Workspace *w);
```

Add to `c/src/nn.c`:

```c
void workspace_init(Workspace *w) {
    w->a  = tensor_alloc(32, 28, 28);
    w->b  = tensor_alloc(32, 28, 28);
    w->p1 = tensor_alloc(32, 14, 14);
    w->c  = tensor_alloc(32, 14, 14);
    w->d  = tensor_alloc(32, 14, 14);
    w->p2 = tensor_alloc(32,  7,  7);
    w->initialized = 1;
}

void workspace_free(Workspace *w) {
    tensor_free(&w->a);  tensor_free(&w->b);  tensor_free(&w->p1);
    tensor_free(&w->c);  tensor_free(&w->d);  tensor_free(&w->p2);
    w->initialized = 0;
}

static void conv2d_into(Tensor *out, const Tensor *input,
                        const float *weights, const float *bias,
                        int out_channels, int k, int stride, int pad) {
    int out_h = (input->height + 2 * pad - k) / stride + 1;
    int out_w = (input->width  + 2 * pad - k) / stride + 1;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float sum = bias[oc];
                for (int ic = 0; ic < input->channels; ++ic) {
                    for (int ky = 0; ky < k; ++ky) {
                        for (int kx = 0; kx < k; ++kx) {
                            int iy = oy * stride - pad + ky;
                            int ix = ox * stride - pad + kx;
                            if (iy < 0 || iy >= input->height ||
                                ix < 0 || ix >= input->width) continue;
                            float v = tensor_get(input, ic, iy, ix);
                            size_t wi = (((size_t)oc * input->channels + ic) * k + ky) * k + kx;
                            sum += v * weights[wi];
                        }
                    }
                }
                tensor_set(out, oc, oy, ox, sum);
            }
        }
    }
}

static void maxpool2d_into(Tensor *out, const Tensor *input, int k, int stride) {
    int out_h = (input->height - k) / stride + 1;
    int out_w = (input->width  - k) / stride + 1;
    for (int c = 0; c < input->channels; ++c) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float best = -INFINITY;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        float v = tensor_get(input, c, oy*stride+ky, ox*stride+kx);
                        if (v > best) best = v;
                    }
                }
                tensor_set(out, c, oy, ox, best);
            }
        }
    }
}

void model_forward_ws(const CnnModel *m, const Tensor *input,
                       float *logits_out, Workspace *w) {
    conv2d_into(&w->a, input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);
    relu_tensor(&w->a);
    conv2d_into(&w->b, &w->a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);
    relu_tensor(&w->b);
    maxpool2d_into(&w->p1, &w->b, 2, 2);
    conv2d_into(&w->c, &w->p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);
    relu_tensor(&w->c);
    conv2d_into(&w->d, &w->c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);
    relu_tensor(&w->d);
    maxpool2d_into(&w->p2, &w->d, 2, 2);
    linear(m->fc_w, m->fc_b, w->p2.data, logits_out, 32*7*7, 10);
}
```

**Every step:** hypothesis → baseline measurement (Ch. 13) → implementation → **re-run Chapter 6's parity check** (a "faster" version that silently changes the math is not a valid optimization) → re-measure → keep or revert based on the number, not the feeling that it should be faster.

### Next
Chapter 15 — deployment-side correctness and speed are now handled; model-quality experiments are a separate, later concern.

---

## Chapter 15 — ML experiments

Only after Chapters 1–14. One variable at a time:

| Experiment | Hypothesis | Metric |
|---|---|---|
| Data augmentation (rotation/shift) | Improves robustness to off-center/rotated hand-drawn digits (Ch. 7's domain-shift concern) | Test accuracy on a held-out set drawn from the actual UI, not just MNIST test set |
| Normalization (mean/std) | Current training has none (per earlier architecture discussion) — adding it may or may not help; untested | Test accuracy, before/after |
| Kernel size / channel count | Larger model may reduce error but changes `CnnModel`'s struct sizes — cascades into Ch. 5's export/load contract | Accuracy vs. `sizeof(CnnModel)` and inference time trade-off |
| Optimizer / learning rate / batch size | Standard hyperparameter sensitivity | Convergence speed, final accuracy |

Each: hypothesis → **one** change → train → evaluate → record → keep or reject. Changing kernel size *and* learning rate in the same run makes the result uninterpretable — you won't know which change caused what.

---

## Chapter 16 — Mathematics through the project

| Topic | Where in Number Guesser |
|---|---|
| Vectors/matrices, dot product | `linear`'s inner loop: `sum += row[i] * x[i]` — one dot product per output neuron |
| Matrix multiply, `y=Wx+b` | `linear`, full function — `nn.Linear(1568,10)`'s C equivalent |
| Tensor shape, flattening | `Tensor` struct + `(c*H+y)*W+x`; `p2.data` passed straight to `linear` in `model_forward` — flatten is free because the memory was never anything but flat |
| Filters, channels, cross-correlation | `conv2d`'s six loops — PyTorch's `Conv2d` computes cross-correlation, not true convolution (no kernel flip) — same as this code |
| Stride, padding, output dims | Chapter 4's worked `28→28→14→7` derivation |
| Derivatives, gradients, chain rule, backprop | **Not in this codebase at all, by design** — training happens in PyTorch; `nn.c` only implements the forward pass. If you want to understand backprop concretely, that's a PyTorch-side exercise (autograd), not something to add to `nn.c` |
| Logits, softmax, probabilities | `main.c`'s `softmax` — `exp(x-max)/Σexp` |
| Confidence | `app->confidence = app->probs[app->predicted_digit]` — literally the softmax output at the predicted index |
| Gradient descent, SGD/Adam, learning rate, batch size | PyTorch training side — relevant to Chapter 15, not to anything in `nn.c`/`ui.c` |

---

## Chapter 17 — D2L + MML learning map

| Topic | Why needed here | Material | Code connection | Study before |
|---|---|---|---|---|
| Linear algebra basics | `linear`, tensor indexing | MML Ch. 2 | `nn.c`'s `linear`, `tensor_get`/`set` | Chapter 4 |
| Convolutions | `conv2d` | D2L §6.1–6.3 | `nn.c`'s `conv2d` | Chapter 4 |
| Pooling | `maxpool2d` | D2L §6.5 | `nn.c`'s `maxpool2d` | Chapter 4 |
| Softmax/cross-entropy | `softmax`, training loss | D2L §3.4, §4.4 | `main.c`'s `softmax` | Chapter 8/16 |
| Optimization (SGD/Adam) | Training only | D2L Ch. 11, MML Ch. 7 | Not in `nn.c` — PyTorch side | Chapter 15, only if running new experiments |

Just-in-time — read the row's material right before the chapter that needs it, not the whole book first.

---

## Chapter 18 — AI-agent workflow

```
inspect → understand → plan → implement ONE change → compile → test →
benchmark → inspect diff → document → commit
```
Agents (including me, in this conversation) must not: claim parity without measurements (Chapter 6's numbers must be real runs, not plausible-looking invented ones — this is why several sections above are explicitly marked **[UNCONFIRMED]** rather than filled with invented output); invent benchmarks; rewrite working code casually (Chapter 4 audits, doesn't replace, your `conv2d`/`maxpool2d` — both are already correct); optimize without profiling (Ch. 13 before Ch. 14, strictly); change architecture casually (a `CnnModel` change cascades through Ch. 5 and Ch. 6 every time); add dependencies without justification; delete files without checking references.

---

## Chapter 19 — Long-term phases

| Phase | Objective | Starting point | DoD |
|---|---|---|---|
| 0 — trustworthy baseline | Ch. 3 | Confirmed builds clean this session | `cmake --build` + manual smoke test pass |
| 1 — numerical parity | Ch. 6 | `nn.c` confirmed compiling; `benchmark/` not yet built | Every layer PASSES at stated tolerance vs. real PyTorch |
| 2 — preprocessing correctness | Ch. 7 | Bounding-box+bilinear centering already implemented | Fixtures built, centering method's accuracy impact known |
| 3 — Raylib product | Ch. 8 | Already substantially built (bars, confidence, FPS) | Preprocessing preview added |
| 4 — explainable inference | Ch. 12 | Not started | conv1 activation view works on real data |
| 5 — tests/sanitizers/CI | Ch. 9–11 | Not wired into CMake at all | Full CI pipeline green |
| 6 — profiling/performance | Ch. 13–14 | Not started | Real timing table exists; optimizations (if any) re-verified against Ch. 6 |
| 7 — model serialization | Ch. 5 | Header-less format, confirmed self-consistent | Versioned format built, only after Phase 1 |
| 8 — ML experiments | Ch. 15 | Not started | One-variable-at-a-time experiment log exists |

---

## Chapter 20 — Definition of done

A milestone is complete only when: implementation works (compiled + run, not just written); tests exist and pass; parity passes where relevant (Ch. 6); sanitizer checks pass (Ch. 10); documentation matches reality (this book gets corrected against your real `python/`/`tests/`/`benchmark/` files once sent — it does not yet, and says so explicitly above); build is reproducible (`cmake --build` from a clean checkout, no manual steps); benchmark is recorded (Ch. 13's real numbers, not estimates); no known regression remains.

---

## Chapter 21 — The Python reference design

**[UNCONFIRMED]** — these files were not provided. This is the reference design compatible with your confirmed C loader.

### `python/model.py`

```python
"""CNN architecture for handwritten digit classification."""

import torch.nn as nn


class _MainModel(nn.Module):
    """
    A small CNN for MNIST.

    Architecture:
        1x28x28
        -> conv1 (1->32, 3x3, pad=1)  -> relu
        -> conv2 (32->32, 3x3, pad=1) -> relu -> maxpool
        -> conv3 (32->32, 3x3, pad=1) -> relu
        -> conv4 (32->32, 3x3, pad=1) -> relu -> maxpool
        -> flatten
        -> linear (1568->10)
    """

    def __init__(self, input_shape=1, hidden_units=32, output_shape=10):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flatten_size = hidden_units * 7 * 7

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, output_shape),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
```

Line by line:
- **`nn.Sequential(...)`** — a container that applies modules in order.
- **`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`** — a 2D convolution.
- **`nn.ReLU()`** — the activation.
- **`nn.MaxPool2d(kernel_size=2, stride=2)`** — a pooling layer. `ceil_mode=False` by default, matching the `floor()` in the C code.
- **`nn.Flatten()`** — reshapes `(N, C, H, W)` to `(N, C*H*W)`. Same as `p2.data` in C.
- **`nn.Linear(in_features, out_features)`** — a fully connected layer.
- **`super().__init__()`** — required in any `nn.Module` subclass.

### `python/dataset.py`

```python
"""MNIST DataLoaders."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size=64, data_root="data"):
    transform = transforms.ToTensor()

    train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test  = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

Line by line:
- **`transforms.ToTensor()`** — the entire preprocessing: convert to a single-channel float tensor, scale to `[0, 1]`. No normalization, no augmentation.
- **`datasets.MNIST(..., train=True, download=True, ...)`** — downloads if missing.
- **`DataLoader(..., shuffle=True)`** — shuffles the training set every epoch; the test set is not shuffled.

### `python/train.py`

```python
"""Training and evaluation steps."""

import torch

def train_step(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_step(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / len(loader), correct / n
```

Line by line:
- **`model.train()` / `model.eval()`** — set training vs. eval mode.
- **`optimizer.zero_grad()`** — clear the gradients from the previous step.
- **`loss.backward()`** — compute the gradients via autograd.
- **`optimizer.step()`** — update the weights.
- **`torch.no_grad()`** — disable autograd for evaluation. Faster, less memory.
- **`(logits.argmax(dim=1) == y).sum().item()`** — count correct predictions.

### `python/evaluate.py`

```python
"""Train the model and save it."""

import torch
import torch.nn as nn
from pathlib import Path

from model import _MainModel
from dataset import get_loaders
from train import train_step, test_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_loaders()

    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} "
              f"test_loss={test_loss:.4f} acc={test_acc:.4f}")

    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "number_guesser_model.pth")
    print(f"saved {out_dir / 'number_guesser_model.pth'}")

if __name__ == "__main__":
    main()
```

Line by line:
- **`torch.device("cuda" if ...)`** — use GPU if available.
- **`nn.CrossEntropyLoss()`** — combines softmax and NLL. Standard for classification.
- **`torch.optim.Adam(model.parameters(), lr=1e-3)`** — Adam optimizer.
- **`torch.save(model.state_dict(), path)`** — save only the weights, not the whole model object. This is what `export.py` reads.

### `python/export.py`

```python
"""Export trained PyTorch weights to a flat binary file for C inference."""

import torch
from pathlib import Path
from model import _MainModel

MODEL_PATH = Path("../models/number_guesser_model.pth")
OUTPUT_PATH = Path("../models/weights.bin")
EXPECTED_BYTES = 175016

LAYER_KEYS = [
    "block_1.0.weight", "block_1.0.bias",
    "block_1.2.weight", "block_1.2.bias",
    "block_2.0.weight", "block_2.0.bias",
    "block_2.2.weight", "block_2.2.bias",
    "classifier.1.weight", "classifier.1.bias",
]

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found.")

    state = torch.load(MODEL_PATH, map_location="cpu")

    missing = [k for k in LAYER_KEYS if k not in state]
    if missing:
        raise SystemExit(
            f"state_dict missing keys: {missing}\n"
            f"Actual keys: {list(state.keys())}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        for key in LAYER_KEYS:
            tensor = state[key]
            f.write(tensor.contiguous().numpy().tobytes())

    size = OUTPUT_PATH.stat().st_size
    status = "OK" if size == EXPECTED_BYTES else "MISMATCH"
    print(f"wrote {OUTPUT_PATH} ({size} bytes) [{status}]")

if __name__ == "__main__":
    main()
```

Line by line:
- **`LAYER_KEYS`** — the exact order the C loader expects. Must match `model_load`'s read order. This is the contract.
- **`state = torch.load(...)`** — the state_dict is a `dict[str, Tensor]`.
- **`missing = [k for k in LAYER_KEYS if k not in state]`** — list comprehension checks for missing keys.
- **`tensor.contiguous()`** — ensures C-contiguous layout. `.numpy().tobytes()` produces the raw bytes.
- **`status = "OK" if size == EXPECTED_BYTES else "MISMATCH"`** — the self-check.

### The critical order dependency

`LAYER_KEYS` in this script must exactly match the read order in `nn.c`'s `model_load`. If either drifts, the model loads without error but produces nonsense predictions.

The read order in `model_load`:

```
conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, conv4_w, conv4_b, fc_w, fc_b
```

The write order in `LAYER_KEYS`:

```
block_1.0.weight (conv1_w), block_1.0.bias (conv1_b),
block_1.2.weight (conv2_w), block_1.2.bias (conv2_b),
block_2.0.weight (conv3_w), block_2.0.bias (conv3_b),
block_2.2.weight (conv4_w), block_2.2.bias (conv4_b),
classifier.1.weight (fc_w), classifier.1.bias (fc_b)
```

Note the indices: `block_1.0` is the first conv, `block_1.1` is ReLU (no params), `block_1.2` is the second conv. Same for `block_2`. The classifier has `Flatten` at index 0 and `Linear` at index 1.

### `python/dump_intermediate.py`

Shown in Chapter 6. See that chapter for the full file.

### `python/mnist_ascii.py`

Shown in Chapter 7. See that chapter for the full file.

---

## Final Chapter — Next 10 Tasks

Prioritized by actual current bottleneck (missing python/tests/benchmark files and the CMake gap), not arbitrary new features.

**1. Send `python/model.py` and `python/export.py`.**
Why: Chapter 5's `LAYER_KEYS` order is currently inferred, not confirmed — the single highest-risk unverified claim in this whole book. Files: those two. Math: none new. Steps: paste or upload them. Command: none. Test: I confirm `LAYER_KEYS` order against your real `model_load` read order line by line. Expected result: either confirmed matching, or a specific named mismatch to fix. DoD: Chapter 5's **[UNCONFIRMED]** tag removed. Next: unblocks Chapter 6 for real.

**2. Send (or create) `benchmark/debug_input.bin` and run Chapter 6's `verify.c` for real.**
Why: this is the single most important unverified claim in the whole project — no C code has ever been checked against a real trained weight. Files: `benchmark/verify.c` (Ch. 6 design, above), a saved MNIST test image. Math: Ch. 6's tolerance rule. Steps: save one `[0,1]` float32 28×28 image; run `dump_intermediate.py` (needs task 1 first) and `verify.c`; diff. Command: `./build/verify models/weights.bin benchmark/debug_input.bin`. Test: layer-by-layer PASS/FAIL. Expected result: not yet known — that's the point. DoD: every layer PASSES, or a named first-divergent layer identified and fixed. Next: unblocks trusting anything else in the app.

**3. Wire `tests/` and a `verify` target into `CMakeLists.txt`.**
Why: currently zero automated tests run from a clean build — flagged repeatedly above as a real, confirmed gap (not inferred). Files: `CMakeLists.txt`, `tests/test_nn.c` (write per Ch. 9's table if it doesn't exist yet, or wire the existing one if it does — send it to check). Steps: add `enable_testing()` + `add_test` per Ch. 9's snippet. Command: `ctest --test-dir build`. Test: itself. Expected: all pass. DoD: `ctest` runs and passes from a clean `cmake --build`. Next: makes Chapter 10/11 possible.

**4. Run Chapter 10's sanitizer build against whatever tests exist after task 3.**
Why: confirms memory safety beyond "it compiled and the tests happened to pass." Files: none new. Command: Ch. 10's `cmake ... -DCMAKE_C_FLAGS` line. Test: itself. Expected: zero ASan/UBSan errors (reasonably likely, given `nn.c`/`ui.c` already showed careful `size_t` casting and defensive `NULL`-ing in this session's audit — but not yet actually run under a sanitizer). DoD: zero errors, confirmed by a real run. Next: Ch. 11.

**5. Add the CI workflow from Chapter 11, once tasks 3–4 are real.**
Why: protects tasks 1–4's results going forward. Files: `.github/workflows/ci.yml`. Command: none locally — push and check Actions. Test: the workflow itself. Expected: green on a clean push. DoD: a deliberate regression (revert one fix from task 2) makes CI fail — confirms the pipeline actually catches something, not just runs.

**6. Add the preprocessing-preview panel to `main.c` (Chapter 8's DoD).**
Why: cheapest way to visually confirm Chapter 7's centering/cropping is doing what it's supposed to, on real drawings, not synthetic fixtures alone. Files: `main.c`. Steps: render the 28×28 float array from `run_prediction` as a small tile next to the canvas. Command: `cmake --build build && ./build/number_guesser`. Test: manual — draw an off-center digit, confirm the preview shows it centered. Expected: visually centered 20×20-in-28×28 digit. DoD: preview panel renders real data, not a placeholder.

**7. Build Chapter 7's deterministic fixtures (blank/tiny/off-center/huge/thin/thick).**
Why: currently only "seems to work" from manual drawing. Files: a new `tests/test_preprocessing.c` or `tools/preprocessing_fixtures.c`. Math: none new — the bounding-box/bilinear formulas already in `ui.c`. Test: each fixture's 28×28 output inspected against hand-predicted expectations (e.g. a centered blob should stay centered; an off-center one should recenter). DoD: all fixtures pass, or reveal a specific real bug to fix.

**8. Check whether bounding-box-center vs. center-of-mass matters (Chapter 7's flagged hypothesis).**
Why: named as the first thing to check for any accuracy gap on asymmetric digits, not yet confirmed to actually matter. Files: `ui.c`. Steps: after task 2 gives you a working parity/accuracy baseline, try swapping the centering formula to a pixel-weighted centroid and re-measure accuracy specifically on asymmetric digits (7, 2, 9). DoD: either confirmed negligible (keep current code) or confirmed to help (replace it) — a measured decision either way, not a guess.

**9. Add `conv1` activation visualization (Chapter 12).**
Why: fastest way to visually catch a Chapter 5/6-style bug on real drawings once the pipeline is trusted. Files: `main.c`, possibly a `model_forward_full` variant in `nn.c`/`nn.h` (Chapter 12 design). DoD: toggleable debug view renders real `conv1` output as a tile grid.

**10. Profile (Chapter 13) before touching any of Chapter 14's optimization ideas.**
Why: nothing above should be "optimized" on a feeling — get real numbers first, on your actual hardware, once tasks 1–9 give you a trustworthy, tested baseline to measure. Files: `c/tools/benchmark.c` (Chapter 13 design), timing instrumentation added to it. DoD: a real, committed timing table exists.

---

*End of the Number Guesser Project Continuation Book — Combined Edition.*

Every chapter from your original book is preserved verbatim. New material — the full file walkthroughs (`nn.h`, `ui.h`, `ui.c`, `main.c`, `CMakeLists.txt`), the Python reference designs, and the additional code samples — is added inside the relevant chapters without changing the structure. Chapter 21 is new (Python reference design) and sits between Chapter 20 (Definition of done) and the Final Chapter.

Work through the 10 tasks in order. When all ten are complete, the project will be:

- **Provably correct** — C and PyTorch agree layer-by-layer on real weights.
- **Tested** — every operation has a unit test; every end-to-end path has an integration test.
- **Sanitized** — no memory bugs, no undefined behavior.
- **Benchmarked** — a real timing table exists.
- **Versioned** — the model format is self-describing.
- **Documented** — every claim is backed by code or a measurement.

That is the difference between a working prototype and a genuine engineering artifact.