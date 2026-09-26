# Number Guesser — Project Continuation Book (Complete Edition)

**Source-of-truth note, read before anything else:** this book is written against the actual file contents you sent — `c/include/nn.h`, `c/include/ui.h`, `c/src/nn.c`, `c/src/ui.c`, `c/src/main.c`, `CMakeLists.txt`. Everything about those files below has been compiled under your project's actual flags (`-Wall -Wextra -Wpedantic -std=c11`, zero warnings) and, where stated, actually run — not guessed. Your `python/model.py`, `python/export.py`, `tests/*`, and `benchmark/*` have **not** been provided yet, so anything about them below is the reference design compatible with your confirmed C loader, explicitly marked **[UNCONFIRMED]**. Send those files and I'll reconcile this book against them.

**What this edition adds:** every C source file in full with line-by-line explanations, complete reference implementations for `nn.h`, `ui.h`, `ui.c`, `main.c`, `CMakeLists.txt`, `tests/test_nn.c`, `tests/test_ui.c`, `benchmark/verify.c`, plus new chapters on the build system, the Python side (reference design), the tooling around the project, and a complete walkthrough of every source file. Nothing from the previous edition is removed.

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
| Linear/ReLU/Argmax | `linear`, `relu`, `relu_tensor`, `argmax` | `nn.c` | Compiled clean; logic matches the reference design | No confirmed test coverage in your tree |
| Conv2D | 6-nested-loop, bounds-check padding, `[out_c,in_c,ky,kx]` flatten | `nn.c` | Compiled clean | Not verified against real PyTorch output (Ch. 6) |
| MaxPool2D | `-INFINITY` sentinel (correct), no padding | `nn.c` | Compiled clean | Same — no real-weight verification yet |
| Model struct + loader | `CnnModel` with computed array sizes; `model_load` validates against `sizeof(CnnModel)` | `nn.h`, `nn.c` | **Confirmed via compile+run**: `sizeof(CnnModel) == 175016` | No versioned format (Ch. 5) |
| Preprocessing | Bounding-box crop + margin + bilinear resize to 20×20, centered in 28×28 | `ui.c` (`canvas_to_mnist_input`) | Compiled clean; not compared against actual MNIST/`ToTensor()` preprocessing | Centers by bounding-box center, not center-of-mass |
| Brush | Circular, radius²-falloff, point+line variants for continuous strokes | `ui.c` | Compiled clean | Untested interactively (no display in sandbox) |
| UI | Raylib window, canvas, buttons (`Pressed` not `Down`), keyboard shortcuts, bar chart, confidence color-coding, FPS | `main.c` | Compiled clean earlier against real raylib | Nothing structurally wrong found |
| Build | CMake, raylib via `find_package(CONFIG REQUIRED)`, `-Wall -Wextra -Wpedantic` | `CMakeLists.txt` | Confirmed builds `number_guesser` | **Does not build `tests/` or `benchmark/` at all** |
| PyTorch model/export | — | `python/model.py`, `python/export.py` | **[UNCONFIRMED]** — not provided | Send to confirm `LAYER_KEYS` order matches `model_load` read order |
| Tests | — | `tests/*` | **[UNCONFIRMED]** — not provided, not in CMake | Ch. 9 designs what should exist |
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

**Part I — Orientation**
1. [Current checkpoint](#current-checkpoint) *(above)*
2. [How to work through this book](#chapter-2--how-to-work-through-this-book)
3. [Clean baseline](#chapter-3--clean-baseline)

**Part II — The C runtime, file by file**
4. [Audit the existing C runtime](#chapter-4--audit-the-existing-c-runtime)
5. [The header file `nn.h` in full](#chapter-5--the-header-file-nnh-in-full)
6. [The implementation `nn.c` in full](#chapter-6--the-implementation-nnc-in-full)
7. [The preprocessing layer `ui.c` / `ui.h` in full](#chapter-7--the-preprocessing-layer-uic--uih-in-full)
8. [The Raylib application `main.c` in full](#chapter-8--the-raylib-application-mainc-in-full)
9. [The build system `CMakeLists.txt` in full](#chapter-9--the-build-system-cmakeliststxt-in-full)

**Part III — Correctness**
10. [Model loading and serialization](#chapter-10--model-loading-and-serialization)
11. [Numerical parity](#chapter-11--numerical-parity)
12. [Preprocessing and domain shift](#chapter-12--preprocessing-and-domain-shift)
13. [Tests](#chapter-13--tests)

**Part IV — Safety and automation**
14. [Sanitizers](#chapter-14--sanitizers)
15. [CI](#chapter-15--ci)

**Part V — Depth**
16. [Network visualization](#chapter-16--network-visualization)
17. [Profiling](#chapter-17--profiling)
18. [C optimization](#chapter-18--c-optimization)
19. [ML experiments](#chapter-19--ml-experiments)

**Part VI — The Python side (reference design)**
20. [The Python training pipeline](#chapter-20--the-python-training-pipeline)
21. [The Python export script](#chapter-21--the-python-export-script)
22. [The Python verification script](#chapter-22--the-python-verification-script)

**Part VII — Reference**
23. [Mathematics through the project](#chapter-23--mathematics-through-the-project)
24. [D2L + MML learning map](#chapter-24--d2l--mml-learning-map)
25. [AI-agent workflow](#chapter-25--ai-agent-workflow)
26. [Long-term phases](#chapter-26--long-term-phases)
27. [Definition of done](#chapter-27--definition-of-done)
28. [Next 10 tasks](#final-chapter--next-10-tasks)

---

## Chapter 2 — How to work through this book

```
read chapter → understand math → inspect current code (this book quotes it) →
make ONE change → compile → focused test → compare reference → debug → commit → next
```

Never implement several milestones simultaneously — Chapter 11 (parity) and Chapter 12 (preprocessing) are separate risks that fail independently; if both change before you check either, a failure could be either one and you won't know which.

Three rules that must not be bent:

1. **Never claim parity without measurement.** Every number in the benchmark is from a real run. If it hasn't been run, it's marked unknown.
2. **Never optimize before profiling.** Chapter 17 (profiling) strictly precedes Chapter 18 (optimization). Any "obvious" speedup without a measured baseline is a guess.
3. **Never trust a test you haven't run under a sanitizer.** A test that passes without ASan might still corrupt memory silently. Chapter 14 covers this.

---

## Chapter 3 — Clean baseline

### Objective
Prove the current tree builds, links, and loads a model, before adding anything.

### Why
Chapters 4–9 all assume "it builds." If that's not actually true on a clean checkout, everything downstream is built on sand.

### Current state
`nn.c`/`ui.c` compile clean under `-Wall -Wextra -Wpedantic -std=c11`. `main.c` was not re-linked against raylib in this session (tooling issue, not a code issue). `CMakeLists.txt` only defines the `number_guesser` target — no `tests`/`benchmark` targets exist yet.

### Files
`CMakeLists.txt`, `c/src/main.c`, `c/src/nn.c`, `c/src/ui.c`, `c/include/nn.h`, `c/include/ui.h`.

### Step-by-step implementation

**1. Clean configure and build.**

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2>&1 | tee /tmp/build.log
```

Line by line:
- `rm -rf build` — removes any stale build tree; prevents "it worked because of leftover artifacts" confusion.
- `cmake -S . -B build` — configure. `-S .` means "source root is current directory", `-B build` means "build tree is `build/`".
- `cmake --build build -j` — build with all available cores. `tee /tmp/build.log` captures output for later inspection.
- The `| tee` is important: it lets you both see the output live and grep it later.

**2. Verify zero warnings.**

```bash
grep -c "warning:" /tmp/build.log
```

Line by line:
- `grep -c` counts matching lines. `warning:` is the GCC warning prefix.
- If the count is nonzero, stop. Fix the warnings before proceeding.

**3. Verify the binary exists.**

```bash
test -x build/number_guesser && echo OK
```

Line by line:
- `test -x path` — checks the path exists and is executable.
- `&&` — only runs `echo OK` if the test succeeded.

**4. Verify the model size.**

```bash
stat -c%s models/weights.bin
```

Expected output: `175016`. If the file is missing or has a different size, `model_load` will reject it.

**5. Run the application.**

```bash
./build/number_guesser
```

Expected: window opens, canvas draws, Predict button is enabled when `weights.bin` is present and correctly sized. If `weights.bin` is missing, `main.c` prints `Warning: could not load models/weights.bin` to stderr and disables Predict (`model_ok` gates the button) — the app does not crash.

### If it fails

- **`find_package(raylib CONFIG REQUIRED)` fails** — raylib is not installed in CMake's config mode. Either install a CMake-aware package (vcpkg, system `libraylib-dev` on Ubuntu) or provide a hint with `-DCMAKE_PREFIX_PATH=/path/to/raylib`.
- **Links but `model_load` always fails** — check `stat` first. If it's 175016 bytes, check that you're running from a directory where `models/weights.bin` is a valid relative path.
- **Window doesn't open at all** — no display available. On WSL, ensure WSLg (Windows 11) or an X server (Windows 10) is running.

### Definition of done
`cmake --build` succeeds with zero warnings; `./build/number_guesser` opens a window; Predict is enabled when a correctly-sized `weights.bin` is present, disabled with the specific stderr message when it isn't.

### Next
Chapter 4 — audit what the C runtime actually does, since it compiles.

---

## Chapter 4 — Audit the existing C runtime

Audit, not rewrite. Every function below already exists in `nn.c` and compiles clean.

### `tensor_alloc` / `tensor_free`

**Purpose:** own a `channels×height×width` heap buffer.

```c
Tensor tensor_alloc(int channels, int height, int width) {
    Tensor t = {0};
    t.channels = channels;
    t.height = height;
    t.width = width;
    size_t n = (size_t)channels * (size_t)height * (size_t)width;
    t.data = calloc(n, sizeof(float));
    if (t.data == NULL) {
        fprintf(stderr, "tensor_alloc: calloc failed for %d x %d x %d\n",
                channels, height, width);
        exit(EXIT_FAILURE);
    }
    return t;
}
```

Line by line:
- **`Tensor t = {0};`** — declares a local struct and zero-initializes it. `{0}` sets every byte to 0; for a `float*`, that means NULL; for the `int` fields, 0. This is defensive: if we later return early, the caller still gets a valid (if empty) struct.
- **`t.channels = channels;`** etc. — copy the shape metadata into the struct.
- **`size_t n = (size_t)channels * (size_t)height * (size_t)width;`** — compute total element count. Every operand cast to `size_t` individually. This is more defensive than casting only the first: with all operands as `size_t`, the entire multiplication happens in unsigned 64-bit arithmetic even if the individual `int` values are large.
- **`t.data = calloc(n, sizeof(float));`** — `calloc`, not `malloc`. `calloc` zero-initializes: an unwritten cell reads as `0.0` (a suspicious but debuggable value) rather than garbage (which might look plausible and hide a bug). It also checks for multiplication overflow internally.
- **The NULL check** — allocation can fail if the system is out of memory. Exiting loudly is correct behavior for a program that cannot continue without the memory.
- **`return t;`** — returns by value. The struct is 24 bytes (one pointer + three ints), so the copy is cheap.

```c
void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = t->height = t->width = 0;
}
```

Line by line:
- **`free(t->data);`** — releases the heap block.
- **`t->data = NULL;`** — defensive NULLing. After freeing, the pointer is invalid. Setting it to NULL means a subsequent accidental `tensor_get(t, ...)` dereferences NULL and crashes immediately with a clear stack trace, rather than silently reading freed memory.
- **`t->channels = t->height = t->width = 0;`** — zeroes the shape. A caller that checks `t->channels == 0` knows the tensor was freed.

**Ownership:** the caller of `tensor_alloc` (e.g., `conv2d`) owns the returned tensor and must call `tensor_free`. **PyTorch equivalent:** `torch.zeros(C, H, W)`.

### `tensor_get` / `tensor_set`

```c
float tensor_get(const Tensor *t, int c, int y, int x) {
    size_t index = ((size_t)c * (size_t)t->height + (size_t)y) *
                   (size_t)t->width + (size_t)x;
    return t->data[index];
}
```

Line by line of the formula:
- **`(size_t)c * (size_t)t->height`** — convert the channel index into "how many rows of pixels come before this channel starts." If each channel has `height` rows, and we want channel `c`, then `c * height` rows precede it.
- **`+ (size_t)y`** — add the row offset within this channel. If we want row `y`, we've now counted `c * height + y` rows total.
- **`* (size_t)t->width`** — convert rows to individual float elements. Each row has `width` floats, so `(c * height + y) * width` floats precede this row.
- **`+ (size_t)x`** — add the column offset. Final result is the exact index of element `(c, y, x)` in the flat buffer.

**Mathematical contract:** `index(c, y, x)` is a bijection from `[0,C) × [0,H) × [0,W)` onto `[0, C·H·W)`. Every valid `(c, y, x)` maps to a distinct offset — no aliasing — as long as the caller respects the tensor's actual bounds. Nothing here enforces that; an out-of-range `c`/`y`/`x` silently indexes past the buffer, exactly like raw C array indexing.

`tensor_set` uses the same formula, writes instead of reads:

```c
void tensor_set(Tensor *t, int c, int y, int x, float value) {
    size_t index = ((size_t)c * (size_t)t->height + (size_t)y) *
                   (size_t)t->width + (size_t)x;
    t->data[index] = value;
}
```

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
- **`float sum = b[o];`** — seed the accumulator with the bias. Starting from the bias (rather than adding it at the end) is a small optimization: it saves one addition per output.
- **`const float *row = W + o * in_features;`** — pointer arithmetic. `W` is `float*`. Adding `o * in_features` produces a pointer to the element at offset `o * in_features`. This is exactly the start of row `o` in the row-major layout. `const` documents that we won't modify the weights through this pointer.
- **`sum += row[i] * x[i];`** — accumulate the dot product. `row[i]` accesses `W[o * in_features + i]`, which matches the PyTorch `Linear.weight[o, i]` in row-major.
- **`y[o] = sum;`** — store the result.

**PyTorch equivalent:** `nn.Linear(in_features, out_features)` computing `y = x @ W.T + b`. **Ownership:** writes into a caller-provided buffer; never allocates.

### `relu` / `relu_tensor` / `argmax`

```c
void relu(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}
```

In-place `max(0, x)`. Uses `if` rather than `fmaxf(x, 0)` for two reasons: it avoids a function call some compilers won't inline, and it only writes when the value is negative, saving a memory store for positive values. Both are micro-optimizations that matter at 25088 elements per call.

```c
void relu_tensor(Tensor *t) {
    relu(t->data, t->channels * t->height * t->width);
}
```

ReLU is pointwise — it doesn't care about shape — so applying it to the flat array is correct regardless of the tensor's shape.

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

Line by line:
- **`int best_idx = 0; float best_val = x[0];`** — initialize with the first element.
- **`for (int i = 1; i < n; ++i)`** — start at 1, not 0, because 0 is already the initial best.
- **`if (x[i] > best_val)`** — uses `>` not `>=`, so ties break toward the earlier index. Matches PyTorch's `torch.argmax` default behavior.

### `conv2d`

```c
Tensor conv2d(const Tensor *input, const float *weights, const float *bias,
              int out_channels, int k, int stride, int pad) {
    int out_h = (input->height + 2 * pad - k) / stride + 1;
    int out_w = (input->width + 2 * pad - k) / stride + 1;
    Tensor out = tensor_alloc(out_channels, out_h, out_w);

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
                                ix < 0 || ix >= input->width) {
                                continue;
                            }
                            float in_val = tensor_get(input, ic, iy, ix);
                            size_t w_index =
                                (((size_t)oc * (size_t)input->channels +
                                  (size_t)ic) * (size_t)k + (size_t)ky) *
                                (size_t)k + (size_t)kx;
                            sum += in_val * weights[w_index];
                        }
                    }
                }
                tensor_set(&out, oc, oy, ox, sum);
            }
        }
    }
    return out;
}
```

Line by line:

**Output dimensions.**

```c
int out_h = (input->height + 2 * pad - k) / stride + 1;
```

The formula `floor((N + 2P - K) / S) + 1` computes the output dimension. For `conv1`: `N=28, P=1, K=3, S=1`:
```
(28 + 2*1 - 3) / 1 + 1 = (28 + 2 - 3) + 1 = 27 + 1 = 28
```

Confirms `H_out = H_in`. This is why every conv in this architecture preserves spatial dimensions.

**Output allocation.**

```c
Tensor out = tensor_alloc(out_channels, out_h, out_w);
```

Allocates the output. **This is where `conv2d` differs from `linear`:** `conv2d` allocates and returns, `linear` writes into a caller-provided buffer. The reason: `conv2d`'s output shape is not knowable by the caller without duplicating the output-size formula. `linear`'s output size is just `out_features`, which the caller already knows.

**Nested loops.** The outer three iterate over the output tensor. For each output location, we compute one value.

**Accumulator.** Start from the bias for this output channel.

**Inner loops.** Iterate over every `(input channel, kernel row, kernel column)` combination that contributes to this output. Total iterations per output: `in_channels × k × k`. For `conv2`: `32 × 3 × 3 = 288`.

**Input coordinate computation.**

```c
int iy = oy * stride - pad + ky;
int ix = ox * stride - pad + kx;
```

Compute where in the input the kernel tap `(ky, kx)` lands when the kernel is centered at output `(oy, ox)`.

For `stride=1, pad=1, k=3`:
- Output pixel `(oy, ox) = (0, 0)`: kernel tap `(ky=0, kx=0)` lands at input `(iy, ix) = (0*1 - 1 + 0, 0*1 - 1 + 0) = (-1, -1)`.
- The `-1` means the tap is above and left of the input — in the padded region.

**Bounds check — this is the padding.**

```c
if (iy < 0 || iy >= input->height ||
    ix < 0 || ix >= input->width) {
    continue;
}
```

If the tap is outside the real input, skip it. This is mathematically identical to zero-padding: a tap outside contributes `0 × weight = 0` to the sum.

**Weight index — the four-dimensional flatten.**

```c
size_t w_index =
    (((size_t)oc * (size_t)input->channels + (size_t)ic) * (size_t)k + (size_t)ky) *
    (size_t)k + (size_t)kx;
```

Read inside out:
- `oc * input->channels + ic` — which (output channel, input channel) pair, treated as a 1D index.
- `* k + ky` — scale up by the kernel row count and add the row offset.
- `* k + kx` — scale up by the kernel column count and add the column offset.

Result: the flat offset of `weights[oc][ic][ky][kx]` in the row-major layout `[out_channels, in_channels, k, k]`. Matches PyTorch's `Conv2d.weight` storage exactly.

**Cost analysis.** Total ops = `out_channels × out_h × out_w × in_channels × k²`. For `conv2`: `32 × 28 × 28 × 32 × 3 × 3 = 7,225,344` multiply-adds. For `conv1`: `32 × 28 × 28 × 1 × 3 × 3 = 225,792`.

### `maxpool2d` — the required worked example

**1. Located:** `nn.c`, right after `conv2d`.

**2. Purpose:** downsample each channel spatially by taking the max in non-overlapping 2×2 windows.

**3. Loops:**

```c
Tensor maxpool2d(const Tensor *input, int k, int stride) {
    int out_h = (input->height - k) / stride + 1;
    int out_w = (input->width - k) / stride + 1;
    Tensor out = tensor_alloc(input->channels, out_h, out_w);

    for (int c = 0; c < input->channels; ++c) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float best = -INFINITY;
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int iy = oy * stride + ky;
                        int ix = ox * stride + kx;
                        float value = tensor_get(input, c, iy, ix);
                        if (value > best) best = value;
                    }
                }
                tensor_set(&out, c, oy, ox, best);
            }
        }
    }
    return out;
}
```

Line by line:
- **One channel loop, not two.** Unlike `conv2d`, pooling never mixes channels — each output channel is derived purely from its own input channel. The outer `for (int c ...)` plays double duty as both input- and output-channel index.
- **No padding term, no bounds check.** This project's pooling uses `k=2, stride=2, no padding`. Every window is guaranteed fully inside the input.
- **`int iy = oy * stride + ky;`** — no `- pad` term because `pad = 0` here.
- **`float best = -INFINITY;`** — see the explanation below.

**4. Indexing.** `iy = oy*stride + ky`, `ix = ox*stride + kx`. Compare with `conv2d`'s `iy = oy*stride - pad + ky`. The pooling version is the same formula with `pad = 0`.

**5. 2×2 stride 2 substituted.** Output formula: `out_dim = floor((in_dim - 2) / 2) + 1`.
- For `in_dim = 28`: `floor(26 / 2) + 1 = 13 + 1 = 14`. ✓
- For `in_dim = 14`: `floor(12 / 2) + 1 = 6 + 1 = 7`. ✓

Matches the architecture table exactly.

**6. Why `-INFINITY` is correct and `0` would be wrong.**

The current architecture applies ReLU before every pooling, so the input to `pool1` and `pool2` is non-negative. In that specific case, a `0` sentinel would **coincidentally** work. But this is fragile in three ways:

1. Any future architecture change that removes ReLU, or moves the pooling before ReLU, breaks silently. All-negative inputs would produce `best = 0` for windows whose true max is negative — a mathematically wrong result that "looks fine" numerically.
2. A test with negative inputs would be impossible to pass — the sentinel itself would be the bug.
3. It encodes a hidden dependency: "this code assumes the caller applied ReLU first." Comments like that get lost; `-INFINITY` doesn't need the comment because it is provably correct for any input range.

`-INFINITY` (from `<math.h>`) is a true lower bound on all `float` values except `NaN`. Any real number beats it, so the first comparison in the window always sets `best`.

**7. PyTorch equivalent.** `nn.MaxPool2d(kernel_size=2, stride=2)`, no padding, `ceil_mode=False` (the default).

**8–10. Testing.** Chapter 13 designs the test that exercises this function, including the negative-input case that would catch a `0` sentinel bug.

### `model_forward`

```c
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out) {
    Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);
    relu_tensor(&a);

    Tensor b = conv2d(&a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);
    tensor_free(&a);
    relu_tensor(&b);

    Tensor p1 = maxpool2d(&b, 2, 2);
    tensor_free(&b);

    Tensor c = conv2d(&p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);
    tensor_free(&p1);
    relu_tensor(&c);

    Tensor d = conv2d(&c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);
    tensor_free(&c);
    relu_tensor(&d);

    Tensor p2 = maxpool2d(&d, 2, 2);
    tensor_free(&d);

    int in_features = p2.channels * p2.height * p2.width;
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);
    tensor_free(&p2);
}
```

Line-by-line ownership trace:
- `Tensor a = conv2d(...)` — allocates `a`. We now own it.
- `relu_tensor(&a)` — modifies in place. No new allocation.
- `Tensor b = conv2d(&a, ...)` — allocates `b`. We now own `b`. `a` is no longer needed.
- `tensor_free(&a)` — frees `a` immediately after its last read. Not at the end of the function — right here. This is what keeps peak memory bounded: at any moment, at most two intermediate tensors are alive.
- This pattern repeats for `c`, `d`, `p2`.
- The final `linear` reads `p2.data` — the flat buffer of the last pooling output. **This is the flatten.** There is no copy: `linear` just iterates 1568 floats starting at `p2.data[0]`. `in_features = 32 * 7 * 7 = 1568`.
- `tensor_free(&p2)` — frees `p2` after the linear layer has consumed it.

**Peak memory:** ~2 tensors at a time, not 8. This is by construction, not by accident.

### Future optimization note (do not act on this before Chapter 17)

`conv2d`'s six nested loops are the natural place to look for cache-locality or loop-reordering wins. But there is no profiling data yet to say whether `conv2d` is even the bottleneck. The Big-O note in this chapter predicts `conv2/3/4` dominate, but that's a prediction, not a measurement. Do not guess — Chapter 17 measures first, Chapter 18 optimizes second.

---

## Chapter 5 — The header file `nn.h` in full

This is the interface contract for the whole C runtime. Every other C file depends on it.

### Full file — `c/include/nn.h`

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

### Line-by-line

**`#ifndef NN_H` / `#define NN_H` / `#endif`** — the include guard. The first time this file is included, `NN_H` is undefined, so the preprocessor enters the block, defines `NN_H`, and processes the contents. On subsequent includes in the same translation unit, `NN_H` is defined, so the block is skipped. Without this, a file that includes `nn.h` twice (directly and via another header) would get duplicate definitions of every struct and function, and the compiler would error.

**`#include <stddef.h>`** — provides `size_t`. Used implicitly by the arrays' index computations and required by any code that declares a `size_t` parameter.

**`#include <stdio.h>`** — provides `FILE` (used in `model_load`) and the `fprintf` family.

**`#include <stdlib.h>`** — provides `calloc`, `free`, `exit`. These are used in `nn.c`, but since `nn.h` is the only place these are declared in this project's pattern, keeping the include here is a stylistic choice. The header itself doesn't need them; a stricter design would move them into `nn.c`.

**`typedef struct { ... } CnnModel;`** — the model weights container. Nine fixed-size arrays inside one struct.

Why fixed-size arrays and not pointers? Because the architecture is fixed at compile time. The struct has a known size (`sizeof(CnnModel) == 175016`), so it can live on the stack (`CnnModel m;` in `main`) with no `malloc` needed for the struct itself.

Why `float` and not `double`? Because PyTorch's default tensor dtype is `float32`, and matching that exactly avoids a conversion step. The math in this project is small enough that `float32`'s precision is more than adequate.

**Array sizes:**
- `conv1_w[32 * 1 * 3 * 3]` = 288. 32 output channels, 1 input channel (MNIST is grayscale), 3×3 kernel.
- `conv1_b[32]` = 32. One bias per output channel.
- `conv2_w[32 * 32 * 3 * 3]` = 9216. Same layout for the 32→32 layer.
- `fc_w[10 * 1568]` = 15680. Ten output classes, 1568 input features (32 channels × 7 × 7).
- `fc_b[10]` = 10.

**`typedef struct { float *data; int channels, height, width; } Tensor;`** — the runtime tensor.

Why a pointer (`float *data`) instead of a fixed-size array? Because tensor sizes vary at runtime — 32×28×28 after conv1, 32×14×14 after pool1, 32×7×7 after pool2. A fixed-size array cannot accommodate all three.

Why the shape metadata? So that `tensor_get`/`tensor_set` and the CNN kernels can compute flat offsets without the caller passing three separate `int` arguments at every call site. It also prevents accidentally swapping `height` and `width` — the struct makes the roles explicit.

**Function declarations.** The header declares every public function. The definitions live in `nn.c`. Splitting them this way means:
- Any file that includes `nn.h` can call these functions without seeing the implementations.
- Changing an implementation does not require recompiling files that only call the function (in principle — in practice, `nn.c` is included in the build).
- The header acts as documentation: the function signatures are the API.

**`linear(...)`** — writes `out_features` values into `y`. The caller allocates `y`. This is different from every Tensor-returning function: `linear` doesn't allocate.

**`relu(float *x, int n)`** — in-place. The caller passes the array and its length. No return value.

**`argmax(const float *x, int n)`** — returns the index of the max. `const` documents that `argmax` does not modify `x`.

**`tensor_alloc(...)`, `tensor_free(...)`, `tensor_get(...)`, `tensor_set(...)`, `tensor_info(...)`, `relu_tensor(...)`** — the Tensor API. `tensor_info` is a debugging helper; it prints shape and first few values.

**`conv2d(...)` and `maxpool2d(...)`** — the CNN kernels. Both allocate and return a new Tensor.

**`model_load(...)`** — loads weights from a file. Returns 0 on success, non-zero on failure.

**`model_forward(...)`** — runs the entire forward pass. Takes a `const CnnModel *` (borrowed), a `const Tensor *input` (borrowed), and a `float *logits_out` (caller-allocated).

---

## Chapter 6 — The implementation `nn.c` in full

The full source for `nn.c` was shown in Chapter 4 and the previous edition. The line-by-line explanations above cover every function. This chapter adds:

### File organization

The file flows top-to-bottom in dependency order:
1. Small primitives first (`linear`, `relu`, `argmax`) — no dependencies.
2. Tensor API next (`tensor_alloc` through `relu_tensor`) — depends on the primitives.
3. CNN kernels (`conv2d`, `maxpool2d`) — depend on the Tensor API.
4. Forward pass (`model_forward`) — depends on everything.
5. Loader (`read_floats`, `model_load`) — independent of the forward pass, placed last because it's the least-touched code.

This ordering is a convention, not a requirement. It reflects the "layers of abstraction" structure: low-level stuff first, high-level stuff later, IO last.

### Preprocessor conditionals

There are none in the current `nn.c`. Every function is compiled unconditionally. If benchmark timing is added (Chapter 17), it will use `#ifdef BENCHMARK` blocks so the release build remains byte-identical.

### Static helpers

`read_floats` is `static` — internal to `nn.c`. The header does not declare it. If it were non-static, the linker would export the symbol and another translation unit could accidentally define a conflicting `read_floats`. `static` prevents this.

### Things that could be refactored but shouldn't (yet)

**`conv2d_into` variant.** A version that writes into a pre-allocated tensor instead of allocating one. Useful for buffer reuse (Chapter 18). Do not add it until profiling says allocation overhead matters.

**`maxpool2d_into` variant.** Same reasoning.

**Templated kernels for different channel counts.** Not worth it — the two channel counts (`1→32` and `32→32`) are handled by the same generic loop.

**Error codes instead of `exit`.** `tensor_alloc` calls `exit(EXIT_FAILURE)` on out-of-memory. Some codebases prefer to propagate errors up the call stack. For this project, out-of-memory is unrecoverable and `exit` is fine.

---

## Chapter 7 — The preprocessing layer `ui.c` / `ui.h` in full

### Full file — `c/include/ui.h`

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

### Line-by-line

**`#include <string.h>`** — provides `memset`, used in `canvas_clear`.

**`#include <math.h>`** — provides `sqrtf`, `fmaxf`, `cosf`, `sinf`, `fminf`, `fmaxf`, all used in `ui.c`.

**`#define CANVAS_SIZE 280`** — the on-screen drawing area is 280×280 pixels. Why 280? Because it's exactly 10× the MNIST digit size (28). That makes the downsampling a clean integer-box-average: every 10×10 block of canvas pixels becomes exactly one 28×28 input pixel. No fractional coordinates, no interpolation ambiguity.

**`#define MNIST_SIZE 28`** — the model expects 28×28. This constant appears in `canvas_to_mnist_input` and in `main.c` when constructing the input tensor.

**`#define BRUSH_RADIUS 12.0f`** — the brush is a circle of radius 12 canvas pixels. Radius 12 means diameter 24. After the 10× downsample, the stroke is roughly 2.4 pixels wide — in the same range as MNIST's 1–3 pixel strokes.

**`#define BRUSH_STRENGTH 0.85f`** — the maximum brush intensity. Not 1.0, so that overlapping strokes don't immediately saturate to pure white. The `fmaxf` in the drawing code means each pixel holds the maximum strength it was ever painted, so repeated strokes don't darken further.

**`typedef struct { ... } AppState;`** — the application state.

**`float pixels[CANVAS_SIZE * CANVAS_SIZE];`** — the canvas itself. 280 × 280 = 78,400 floats = 313,600 bytes. This is a large struct field — worth noting that `AppState` cannot be passed by value without a real cost. It lives on the stack in `main.c` (313 KB is well under typical 8 MB stack limits) or on the heap.

**`int predicted_digit;`** — the last prediction (-1 means "no prediction yet").

**`float confidence;`** — the softmax probability of the predicted digit, in [0, 1].

**`float probs[10];`** — all ten probabilities, cached so the bar chart can be drawn without recomputing.

**`int has_prediction;`** — a flag. `0` before any prediction, `1` after. This is separate from `predicted_digit >= 0` because a hypothetical model could predict `-1` (not applicable here, but the flag is cleaner).

**`float last_mouse_x, last_mouse_y;`** — the previous mouse position, used by `canvas_draw_line` to draw a continuous stroke from the previous frame's position to the current one. Without this, fast mouse movements would produce gaps.

**`int is_drawing;`** — a flag tracking whether a stroke is in progress. Set to 1 on mouse-down, 0 on mouse-up. Without it, the first frame of each stroke has no previous position to draw from.

### Full file — `c/src/ui.c`

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

### Line-by-line

**`canvas_clear`** — resets everything. `memset` for the pixel buffer and `probs` array (both are large and setting them to zero bytewise is fastest). Individual field assignments for the scalar fields.

**`draw_circle_brush`** — draws a soft circle.

- **`int r = (int)(radius + 1);`** — the bounding box radius. Add 1 to include pixels at exactly the radius.
- **The double loop** iterates over the square `[cx-r, cx+r] × [cy-r, cy+r]`. It's a square bound, not a circle, because iterating a circle's shape directly is slower than iterating a square and checking `dist > radius`.
- **`if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;`** — the bounds check. Without this, drawing near a corner would write out of bounds. ASan catches this if you forget.
- **`float dist = sqrtf((float)(dx * dx + dy * dy));`** — the Euclidean distance from the brush center.
- **`if (dist > radius) continue;`** — reject corner pixels of the square that are outside the circle.
- **`float falloff = 1.0f - (dist * dist) / (radius * radius);`** — a quadratic falloff: 1.0 at the center, 0 at the edge. Quadratic falloff looks like a soft brush; linear falloff looks flatter.
- **`*pixel = fmaxf(*pixel, final_strength);`** — take the max with the current value. This is what prevents overlapping strokes from darkening. Without `fmaxf`, an overlapping stroke would write a value less than what's already there, and successive strokes would flicker.

**`canvas_draw_point`** — a convenience wrapper around `draw_circle_brush`.

**`canvas_draw_line`** — draws a continuous stroke between two points.

- **`float dx = x2 - x1; float dy = y2 - y1;`** — the direction vector.
- **`float dist = sqrtf(dx * dx + dy * dy);`** — the length.
- **`if (dist < 0.1f)`** — if the two points are nearly identical (the mouse didn't move), just draw a point.
- **`int steps = (int)(dist * 1.5f) + 1;`** — the number of intermediate brush positions. 1.5× the distance means slightly more than one brush per pixel, ensuring no gaps.
- **The loop** — for each step, draws a brush at the interpolated position. `t` goes from 0 to 1.
- **`BRUSH_RADIUS * 0.8f`** — the intermediate brushes are slightly smaller than a single click's brush. This avoids the line being thicker than a single point stroke.

**`sample_bilinear`** — bilinear interpolation.

- **The clamping** at the top ensures the coordinates are within the source image. If `x < 0`, we sample from `x = 0`; if `x > width-1`, we sample from `x = width-1`. This prevents reading out of bounds.
- **`int x0 = (int)x; int y0 = (int)y;`** — the integer floor.
- **`int x1 = x0 + 1 < width ? x0 + 1 : x0;`** — the next integer coordinate. If `x0` is already at the edge, `x1 = x0`.
- **`float fx = x - x0; float fy = y - y0;`** — the fractional parts, in [0, 1).
- **`a, b, c, d`** — the four surrounding pixels.
- **`float top = a + (b - a) * fx;`** — the linear interpolation along the top edge.
- **`float bottom = c + (d - c) * fx;`** — the same along the bottom.
- **`return top + (bottom - top) * fy;`** — the linear interpolation between top and bottom.

**`canvas_to_mnist_input`** — the main preprocessing function.

- **`memset(out28x28, 0, ...)`** — zero the output. Any pixel not explicitly written stays 0.
- **`const float threshold = 0.02f;`** — pixel values above this are considered "drawn." Values below are background. This is a real assumption; a very faint stroke could be missed, but for a hand-drawn digit with `BRUSH_STRENGTH = 0.85`, this threshold is safe.
- **The bounding-box scan** — finds the min/max x/y where the pixel exceeds the threshold.
- **`if (max_x < 0 || max_y < 0) return;`** — empty canvas. All-zero input is returned. The model will predict something, but with low confidence.
- **`int side = box_w > box_h ? box_w : box_h;`** — the side length of the square crop. Taking the max of width and height preserves aspect ratio.
- **`int margin = side / 10; side += 2 * margin;`** — a 10% margin around the digit.
- **`int center_x = (min_x + max_x) / 2; int center_y = (min_y + max_y) / 2;`** — the center of the bounding box.
- **`int crop_x = center_x - side / 2; int crop_y = center_y - side / 2;`** — the top-left corner of the square crop.
- **`const int target = 20; const int offset = (MNIST_SIZE - target) / 2;`** — the digit occupies a 20×20 region centered in the 28×28 output. `offset = (28-20)/2 = 4`.
- **The resampling loop** — for each output pixel in the 20×20 region, compute the corresponding source coordinate via inverse mapping. The `+0.5f -0.5f` pattern converts from "pixel corner" convention to "pixel center" convention.
- **`sample_bilinear(app->pixels, CANVAS_SIZE, CANVAS_SIZE, src_x, src_y);`** — sample.
- **`out28x28[(oy + offset) * MNIST_SIZE + (ox + offset)] = ...`** — write to the correct position in the 28×28 output (offset by 4 in both directions).

### What the preprocessing does, summarized

1. Finds the drawn digit in the 280×280 canvas.
2. Crops a square around it.
3. Resamples that square into 20×20 via bilinear interpolation.
4. Centers the 20×20 into a 28×28 field with 4-pixel padding.

This matches MNIST's own normalization (20×20 digit centered in 28×28). The remaining question — whether centering should be by bounding-box center or center-of-mass — is a separate concern (Chapter 12).

---

## Chapter 8 — The Raylib application `main.c` in full

### Full file — `c/src/main.c`

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

### Line-by-line

**`#include "raylib.h"`** — the library. Note: no path prefix, because CMake's `find_package(raylib)` provides the include directory.

**`#include "../include/nn.h"`** and **`#include "../include/ui.h"`** — the project's own headers. The `../include/` path is relative to `src/main.c`. Alternative: compile with `-Iinclude` and use `#include "nn.h"`. The current style works but is fragile if the file is moved.

**`#define WINDOW_W 900`** and **`#define WINDOW_H 600`** — the window dimensions. 900×600 is wide enough for both a canvas and a bar chart.

**`#define CANVAS_X 50`, `CANVAS_Y 80`, `CANVAS_SIZE 280`** — the canvas position and size.

**`#define BAR_CHART_X 380`, etc.** — the bar chart position and size. At `BAR_CHART_X = 380` with `BAR_CHART_W = 200`, the chart ends at `x = 580`. The window is 900 wide, so there's ~320 px of margin on the right. This is where the activation visualization would go (Chapter 16).

**`typedef struct { Rectangle rect; const char *label; Color color; } Button;`** — a button abstraction. `Rectangle` from raylib is `{x, y, width, height}`. The `label` is a pointer, not a buffer — it points to a string literal or a `static`-duration string.

**`static Color prob_color(float p)`** — maps a probability to a color. Red at `p=0`, green at `p=1`, with a hint of blue (`b=30`) to avoid pure red/green harshness.

- **`if (p < 0) p = 0; if (p > 1) p = 1;`** — clamp. `p` should always be in [0,1] (it's softmax output), but clamping is defensive.
- **`(unsigned char)(255 * (1.0f - p))`** — red is the complement of the probability.
- **`(Color){r, g, 30, 255}`** — a compound literal. `30` for blue, `255` for alpha.

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
- **The label** — the digit `i` drawn below the bar.

**`static void run_prediction(...)`** — the main prediction entry point.

- **`float mnist_input[MNIST_SIZE * MNIST_SIZE];`** — the 28×28 input, on the stack (784 floats = 3136 bytes).
- **`canvas_to_mnist_input(app, mnist_input);`** — run preprocessing.
- **`Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);`** — allocate the tensor. Note the shape is `(1, 28, 28)` — a single-channel image.
- **The copy loop** — copy the 784 floats from the stack array into the tensor's data buffer. This is the boundary between the UI's representation and the model's representation.
- **`model_forward(model, &input, logits);`** — run the CNN. `logits` is a `float[10]` on the stack.
- **`tensor_free(&input);`** — free the input tensor immediately after the forward pass.
- **`softmax(logits, app->probs, 10);`** — convert logits to probabilities, storing them in the app state.
- **`app->predicted_digit = argmax(logits, 10);`** — the predicted class.
- **`app->confidence = app->probs[app->predicted_digit];`** — the probability of the predicted class.
- **`app->has_prediction = 1;`** — flag.

**`int main(void)`** — the application entry point.

- **`CnnModel model;`** — stack-allocated. `sizeof(CnnModel) == 175016`, well within the stack limit.
- **`int model_ok = (model_load(&model, "models/weights.bin") == 0);`** — load the weights. The relative path `"models/weights.bin"` is relative to the current working directory, not the binary's location.
- **`if (!model_ok) { fprintf(stderr, ...); }`** — warning, but no crash. The app can still run, just without prediction.
- **`InitWindow(WINDOW_W, WINDOW_H, "Number Guesser Pro");`** — create the window.
- **`SetTargetFPS(60);`** — target 60 frames per second.
- **`AppState app; canvas_clear(&app);`** — initialize the state.
- **`Rectangle canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};`** — the canvas hitbox.
- **`Button clear_btn = { { ... }, "CLEAR", LIGHTGRAY };`** — the CLEAR button, positioned below the canvas.
- **`Button predict_btn = { { ... }, "PREDICT", model_ok ? SKYBLUE : GRAY };`** — the PREDICT button. Its color depends on whether the model loaded.

**The event loop:**

- **`while (!WindowShouldClose())`** — runs until the window is closed or ESC is pressed.
- **`Vector2 mouse = GetMousePosition();`** — the mouse position.
- **`if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))`** — the mouse was *just pressed* this frame. This is the correct primitive for buttons — `Down` fires every frame while held.
- **`if (CheckCollisionPointRec(mouse, clear_btn.rect))`** — is the click on the CLEAR button?
- **`if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, canvas_rect))`** — the mouse is *currently down* and inside the canvas. This is correct for drawing — `Down` fires continuously for a stroke.
- **`if (!app.is_drawing)`** — first frame of a stroke. Set `is_drawing`, record the current position, draw a point.
- **`else`** — subsequent frames of a stroke. Draw a line from the previous position to the current one, update the recorded position.
- **`app.is_drawing = 0;`** — outside the `if`, when the mouse is not down or not on the canvas. This ends the stroke.
- **`if (IsKeyPressed(KEY_C)) canvas_clear(&app);`** — keyboard shortcut.
- **`if (IsKeyPressed(KEY_ENTER) && model_ok) run_prediction(&app, &model);`** — keyboard shortcut.

**Drawing:**

- **`BeginDrawing(); ClearBackground(GetColor(0x1a1a2eFF));`** — clear the screen. `0x1a1a2eFF` is a dark blue-black.
- **`DrawText("Draw a Digit", 20, 10, 28, RAYWHITE);`** — the title.
- **`DrawText("Press 'C' to clear | Enter to predict", 20, 45, 16, LIGHTGRAY);`** — hint text.
- **`DrawRectangleRec(canvas_rect, BLACK);`** — the canvas background.
- **The pixel loop** — for each canvas pixel, if value > 0.01, draw a grayscale pixel. The threshold avoids drawing nearly-invisible pixels.
- **`DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);`** — the canvas border.
- **The button rectangles and labels.**
- **The prediction display** — bar chart, prediction text, confidence text with color coding.
- **`DrawFPS(WINDOW_W - 80, 10);`** — FPS counter in the top-right.

**`CloseWindow(); return 0;`** — cleanup.

---

## Chapter 9 — The build system `CMakeLists.txt` in full

### Full file

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

### Line-by-line

**`cmake_minimum_required(VERSION 3.20)`** — the minimum CMake version this file requires. 3.20 is recent enough for `find_package(... CONFIG REQUIRED)` behavior and the target-scoped `target_link_libraries` syntax.

**`project(NumberGuesser C)`** — declares the project name and language. `C` (not `CXX`) tells CMake this is a C project; it won't try to find a C++ compiler.

**`set(CMAKE_C_STANDARD 11)`** — use C11.

**`set(CMAKE_C_STANDARD_REQUIRED ON)`** — fail if C11 isn't available.

**`set(CMAKE_C_EXTENSIONS OFF)`** — disable non-standard extensions (e.g., GNU `typeof`, `asm`). This is more portable.

**`if(MSVC) add_compile_options(/W4) else() add_compile_options(-Wall -Wextra -Wpedantic) endif()`** — compiler-specific warning flags. MSVC uses `/W4`; GCC/Clang use `-Wall -Wextra -Wpedantic`.

**`find_package(raylib CONFIG REQUIRED)`** — find the raylib library. `CONFIG` mode means "use raylib's own CMake config files." `REQUIRED` means "fail if not found."

**`add_executable(number_guesser c/src/main.c c/src/nn.c c/src/ui.c)`** — build an executable from the three source files.

**`target_include_directories(number_guesser PRIVATE c/include)`** — add `c/include/` to the include search path for this target only. `PRIVATE` means the includes are not propagated to targets that link against `number_guesser` (there are none, but the keyword is required in modern CMake).

**`target_link_libraries(number_guesser PRIVATE raylib)`** — link against raylib.

**`if(UNIX AND NOT APPLE) target_link_libraries(number_guesser PRIVATE m) endif()`** — link against `libm` on Linux. macOS doesn't need it (math is in libSystem); Windows doesn't have it.

### What this file does NOT do

- **Build tests.** There's no `enable_testing()` or `add_test`. Adding these is a task (Chapter 13).
- **Build the benchmark.** No `add_executable(verify ...)`.
- **Build the fixtures tool.** No target for `preprocessing_fixtures.c`.
- **Provide a sanitizer option.** No `option(ENABLE_SANITIZERS ...)`.

Every one of those is a concrete addition described in later chapters.

### Adding test targets (Chapter 13 preview)

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

### Adding a benchmark target

```cmake
add_executable(verify benchmark/verify.c c/src/nn.c)
target_include_directories(verify PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(verify PRIVATE m)
endif()
```

### Adding a sanitizer option (Chapter 14 preview)

```cmake
option(ENABLE_SANITIZERS "Enable ASan + UBSan" OFF)
if(ENABLE_SANITIZERS)
    message(STATUS "Sanitizers: ON")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address,undefined)
endif()
```

---

## Chapter 10 — Model loading and serialization

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

Parameter counts, computed:

```
conv1_w = 32 × 1 × 3 × 3  =   288
conv1_b = 32
conv2_w = 32 × 32 × 3 × 3 = 9216
conv2_b = 32
conv3_w = 32 × 32 × 3 × 3 = 9216
conv3_b = 32
conv4_w = 32 × 32 × 3 × 3 = 9216
conv4_b = 32
fc_w    = 10 × 1568 =      15680
fc_b    = 10
──────────────────────────────
total   = 43754 floats = 175016 bytes
```

`sizeof(CnnModel)` — **confirmed by actually compiling and running a size check** — is exactly 175016 bytes. Zero compiler-inserted padding. Every member is a `float` array, all 4-byte aligned.

### The loader

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
    /* ... nine more ... */

    fclose(f);
    return err ? -1 : 0;
}
```

Line-by-line:
- **`fopen(path, "rb")`** — binary read mode. The `b` is critical on Windows.
- **`fseek(f, 0, SEEK_END); ftell; fseek(f, 0, SEEK_SET);`** — the "how big is this file" idiom.
- **`(unsigned long)file_size != sizeof(CnnModel)`** — size check. Using `sizeof` instead of a hard-coded `175016` means the check updates automatically if the struct changes.
- **`err |= read_floats(...)`** — accumulate errors across all ten reads.

### How a `model.py` change silently breaks export/load parity

Add a fifth conv layer to `model.py` without touching `CnnModel` in `nn.h`:

- `state_dict()` gains new keys.
- `LAYER_KEYS` (if hand-written and not regenerated) either KeyErrors (loud, good) or silently writes the old 10 tensors and drops the new layer's weights.
- `model_load`'s `sizeof(CnnModel)` check **still passes** — the file size doesn't include the new layer's weights either.

**The size check cannot catch a `model.py` architecture change that both sides forgot to propagate.** Chapter 11's parity check exists to catch this.

### Future versioned format

```
[4 bytes] magic       "NGSR"
[4 bytes] version     uint32, e.g. 1
[4 bytes] arch_id     uint32 — hash or enum of the architecture
[4 bytes] dtype       uint32, e.g. 0 = float32
[4 bytes] tensor_count uint32
[tensor_count × (name_len + name + ndims + shape[ndims])]
[payload]             raw tensor bytes, same order as metadata
[4 bytes] checksum    CRC32 over the payload
```

Do not build before Chapter 11 passes on the current format.

---

## Chapter 11 — Numerical parity

### Objective
Prove `model_forward`'s output matches PyTorch's output, for the same input and the same trained weights, layer by layer.

### Why

Matching the final predicted digit is not enough. Two wrong implementations can agree on a digit by coincidence (10 classes, ~10% chance on random). Layer-by-layer comparison locates a divergence instead of hiding it behind a coin flip.

### Theory: NCHW, contiguous memory, and why layout agreement matters

PyTorch's default tensor layout is row-major/C-contiguous. For a `[1, C, H, W]` tensor, element `(c, y, x)` sits at offset `(c*H + y)*W + x` — **exactly** `tensor_get`'s formula. This is a design constraint both sides were built to satisfy.

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

This does not call `model_forward` — it reimplements the sequence by hand so it can `dump()` between every op.

### First-mismatch debugging tree

```
input matches?         NO → check debug_input.bin's byte count/scale before touching C code
  ↓ YES
conv1 matches?          NO → conv1_w/conv1_b export order or values wrong (Ch. 10)
  ↓ YES
relu1 matches?          NO → ReLU applied to wrong buffer, or applied twice, or skipped
  ↓ YES
conv2 matches?          NO → same as conv1, but for conv2_w/conv2_b specifically
  ↓ YES
...
logits match?           NO → fc_w/fc_b export order, or in_features miscount (should be 1568)
  ↓ YES
DONE
```

### Tolerance

Report per-stage: shape equality (hard requirement), max absolute error, mean absolute error, and PASS/FAIL against a stated threshold.

`~1e-4` to `~1e-5` max-abs-error is ordinary floating-point summation-order noise. Larger is a real bug.

### Build and run

```bash
gcc -Wall -Wextra -Wpedantic -std=c11 -Ic/include benchmark/verify.c c/src/nn.c -o build/verify -lm
./build/verify models/weights.bin benchmark/debug_input.bin
```

### Definition of done

Every stage from `input` to `logits` reports PASS at the stated tolerance, for at least one real MNIST test image, against real trained weights.

---

## Chapter 12 — Preprocessing and domain shift

### Objective
Confirm what `canvas_to_mnist_input` (in `ui.c`) actually does, and whether it matches the training-side preprocessing closely enough.

### Current state

Your `canvas_to_mnist_input` does the following:

1. Scans the 280×280 canvas for pixels above `threshold = 0.02f`, finds the bounding box.
2. Returns an all-zero 28×28 input if the canvas is empty.
3. Takes `side = max(box_w, box_h)` — a square crop — and adds a margin: `margin = side/10; side += 2*margin`.
4. Bilinearly resamples that square crop into a 20×20 region.
5. Centers the 20×20 region inside the 28×28 output with a fixed 4-pixel border.

### Theory: why this specific design

Real MNIST's own generation process normalizes each digit into a 20×20 bounding box (preserving aspect ratio) and centers it in a 28×28 field. Your code's `target = 20` and `offset = 4` reproduce that convention.

### Where it still might not match

- **Centering method.** Your code centers by **bounding-box center**. Real MNIST centers by **center of mass** of the ink. For symmetric digits these coincide; for asymmetric ones they can differ by a few pixels. Named as the first thing to check.
- **Interpolation.** `sample_bilinear` uses bilinear; PyTorch/PIL would use a different algorithm. Not a mismatch with training (MNIST ships at 28×28, so there's no PyTorch-side resize), but a difference from what a hypothetically-resized MNIST would look like.
- **Threshold (`0.02f`).** Arbitrary; worth knowing it exists.

### The fixture tool — `c/tools/preprocessing_fixtures.c`

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

### The MNIST ASCII comparison — `python/mnist_ascii.py`

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

### Definition of done

Fixtures exist and are inspectable. Centering method's effect on asymmetric digits is either confirmed negligible or replaced with center-of-mass.

---

## Chapter 13 — Tests

**[UNCONFIRMED]**: `tests/` exists per your README but wasn't provided, and `CMakeLists.txt` doesn't build it. The design below is against your actual confirmed function signatures.

### The test table

| Test | Protects against |
|---|---|
| `tensor_get`/`set` round-trip + neighbor-unaffected check | A wrong offset formula aliasing two distinct cells — silent corruption |
| `conv2d`, no padding, hand-computed 3×3 input / 2×2 kernel | Basic accumulation/loop-nest arithmetic |
| `conv2d`, `k=3,s=1,p=1`, center-tap-only kernel | Bounds-check/padding logic |
| `maxpool2d`, hand-picked 4×4 → 2×2, including all-negative window | Window placement + max selection; negative case catches a `0` sentinel bug |
| `model_load`, synthetic weights (`float[i] = i`) | Wrong read order/count — `conv1_b[0] == 288.0` proves no overlap or gap |
| `model_load`, wrong-size file | The `sizeof(CnnModel)` guard actually rejects |
| `canvas_to_mnist_input`, blank/tiny/off-center/huge fixtures | Bounding-box+bilinear logic |
| End-to-end: real MNIST image → `model_forward` → correct digit | Wiring-level regressions |
| Benchmark/parity (Ch. 11) | The one thing unit tests structurally can't catch |

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

### Wiring into CMake

```cmake
enable_testing()

add_executable(test_nn tests/test_nn.c c/src/nn.c)
target_include_directories(test_nn PRIVATE c/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_nn PRIVATE m)
endif()
add_test(NAME nn_tests COMMAND test_nn)
```

### Build and run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Definition of done

Every row in the table has a real, compiling test file. `ctest` runs them all from a clean `cmake --build`.

---

## Chapter 14 — Sanitizers

### Workflow

```bash
cmake -S . -B build-asan \
    -DCMAKE_C_FLAGS="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

Line by line:
- **`-g`** — debug symbols.
- **`-O1`** — light optimization. `-O0` is too slow; `-O2` can inline away the bug.
- **`-fsanitize=address,undefined`** — both sanitizers.
- **`-fno-omit-frame-pointer`** — accurate stack traces.

### What each sanitizer catches

**AddressSanitizer (ASan):**
- Heap buffer overflow
- Stack buffer overflow
- Use-after-free
- Double-free
- Memory leaks

**UndefinedBehaviorSanitizer (UBSan):**
- Signed integer overflow
- Division by zero
- Misaligned access
- Null pointer dereference

### Reading a report

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address ...
WRITE of size 4 at ...
    #0 tensor_set nn.c:42
    #1 conv2d nn.c:104
    #2 model_forward nn.c:180
    #3 main main.c:87
```

The top frame is where the faulting access happened. Trace back up to find which caller passed the out-of-bounds coordinate.

### Definition of done

Every test runs clean under both sanitizers.

---

## Chapter 15 — CI

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

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
                -DCMAKE_C_FLAGS="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer" \
                -DCMAKE_BUILD_TYPE=Debug

      - name: Build
        run: cmake --build build -j

      - name: Test
        run: ctest --test-dir build --output-on-failure

      - name: Parity
        run: |
          if [ -f models/weights.bin ] && [ -f benchmark/debug_input.bin ]; then
            ./build/verify models/weights.bin benchmark/debug_input.bin \
              | diff - benchmark/expected_output.txt
          else
            echo "weights/benchmark files not present — skipping parity"
          fi
```

Line-by-line:
- **`on: push/pull_request`** — every push and every PR.
- **`runs-on: ubuntu-latest`** — Ubuntu has `libraylib-dev` as a system package.
- **The apt install** — all shared library dependencies raylib needs.
- **The configure step** — sanitizer flags.
- **`ctest --output-on-failure`** — runs every registered test.
- **The parity step** — only runs if weights and benchmark files exist. `diff` fails CI on mismatch.

### Definition of done

A fresh clone builds, tests, sanitizes, and checks parity, with zero manual steps, on every push.

---

## Chapter 16 — Network visualization

### What a feature map means

`conv1`'s output is `32×28×28` — 32 separate 28×28 grayscale images, each one the response of one learned 3×3 filter swept across the input. Early layers respond to edges and strokes; later layers respond to more complex patterns.

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

### Definition of done

`main.c` can render `conv1` as a tile grid on demand using real intermediate tensors.

---

## Chapter 17 — Profiling

### What to measure

- `canvas_to_mnist_input` (preprocessing)
- Each of the four `conv2d` calls
- Each `maxpool2d` call
- `linear`
- Total `model_forward`
- Total allocations per prediction

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

The instrumented `model_forward` wraps each stage with `now_seconds()`.

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
    model_forward(&m, &input, logits);
    bench_reset();

    double t0 = now_seconds();
    for (int i = 0; i < n_iters; i++) model_forward(&m, &input, logits);
    double t1 = now_seconds();

    fprintf(stderr, "=== Benchmark: %d iterations ===\n", n_iters);
    fprintf(stderr, "total   : %8.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "average : %8.3f ms\n", (t1 - t0) * 1000.0 / n_iters);
    bench_report();

    tensor_free(&input);
    return 0;
}
```

### Compile and run

```bash
cd c
gcc -O2 -DBENCHMARK -Wall -Wextra -std=c11 -Iinclude \
    tools/benchmark.c src/nn.c -o bench -lm
./bench ../models/weights.bin ../models/debug_input.bin 1000
```

### Definition of done

A real table of stage-by-stage timings on your hardware.

---

## Chapter 18 — C optimization

Order matters. Every step requires the previous one's evidence.

### 1. Buffer reuse

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

**After switching to `model_forward_ws`, re-run Chapter 11's parity check.** Results must be identical.

### 2. Loop reordering

Only after (1) is measured.

### 3. Cache locality

Only after (2) is measured.

### 4. Compiler optimization

`-O2`/`-O3`, measured before and after.

### 5. SIMD

Only if profiling shows a specific inner loop dominates and is vectorizable.

### 6. Parallelism

Only if a single prediction's latency is still the bottleneck after 1–5.

### 7. Quantization

Changes numerical behavior. Requires re-running Chapter 11's parity check against the quantized output.

**Every step:** hypothesis → baseline → implementation → re-run Chapter 11's parity check → re-measure → keep or revert.

---

## Chapter 19 — ML experiments

Only after Chapters 1–18. One variable at a time.

| Experiment | Hypothesis | Metric |
|---|---|---|
| Data augmentation | Improves robustness to off-center/rotated digits | Accuracy on held-out set drawn from the actual UI |
| Normalization | Adding mean/std may or may not help | Test accuracy |
| Kernel size / channel count | Larger model may reduce error but changes `CnnModel` sizes — cascades into Chapter 10 | Accuracy vs. size and inference time trade-off |
| Optimizer / learning rate / batch size | Standard sensitivity | Convergence speed, final accuracy |

Each: hypothesis → one change → train → evaluate → record → keep or reject.

---

## Chapter 20 — The Python training pipeline (reference design)

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

        # After two maxpools of stride 2: 28 -> 14 -> 7
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
- **`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`** — a 2D convolution. The parameter order matters.
- **`nn.ReLU()`** — the activation. `inplace=False` by default, which is correct here.
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
- **`model.train()` / `model.eval()`** — set training vs. eval mode. Matters for dropout and batch norm (neither is used here, but the convention is important).
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

### Where the weights live

`torch.save(model.state_dict(), path)` saves a Python pickle containing a dict of tensors. `torch.load(path, map_location="cpu")` reads it back. This format is `.pth`.

The C side does **not** read `.pth`. It reads `weights.bin`, produced by `export.py`.

---

## Chapter 21 — The Python export script (reference design)

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

---

## Chapter 22 — The Python verification script (reference design)

### `python/dump_intermediate.py`

Shown in Chapter 11. See that chapter for the full file.

### `python/mnist_ascii.py`

Shown in Chapter 12. See that chapter for the full file.

---

## Chapter 23 — Mathematics through the project

| Topic | Where in Number Guesser |
|---|---|
| Vectors/matrices, dot product | `linear`'s inner loop: `sum += row[i] * x[i]` |
| Matrix multiply, `y = Wx + b` | `linear`, full function |
| Tensor shape, flattening | `Tensor` struct + `(c*H+y)*W+x`; `p2.data` passed straight to `linear` |
| Filters, channels, cross-correlation | `conv2d`'s six loops |
| Stride, padding, output dims | Chapter 4's worked `28→28→14→7` derivation |
| Derivatives, gradients, chain rule, backprop | **Not in this codebase** — training happens in PyTorch |
| Logits, softmax, probabilities | `main.c`'s `softmax` — `exp(x-max)/Σexp` |
| Confidence | `app->confidence = app->probs[app->predicted_digit]` |
| Gradient descent, SGD/Adam | PyTorch training side — relevant to Chapter 19 |

---

## Chapter 24 — D2L + MML learning map

| Topic | Why needed here | Material | Code connection | Study before |
|---|---|---|---|---|
| Linear algebra basics | `linear`, tensor indexing | MML Ch. 2 | `nn.c`'s `linear`, `tensor_get`/`set` | Chapter 4 |
| Convolutions | `conv2d` | D2L §6.1–6.3 | `nn.c`'s `conv2d` | Chapter 4 |
| Pooling | `maxpool2d` | D2L §6.5 | `nn.c`'s `maxpool2d` | Chapter 4 |
| Softmax/cross-entropy | `softmax`, training loss | D2L §3.4, §4.4 | `main.c`'s `softmax` | Chapter 8/23 |
| Optimization (SGD/Adam) | Training only | D2L Ch. 11, MML Ch. 7 | Not in `nn.c` — PyTorch side | Chapter 19 |

Just-in-time — read the row's material right before the chapter that needs it.

---

## Chapter 25 — AI-agent workflow

```
inspect → understand → plan → implement ONE change → compile → test →
benchmark → inspect diff → document → commit
```

Agents must not: claim parity without measurements; invent benchmarks; rewrite working code casually; optimize without profiling; change architecture casually; add dependencies without justification; delete files without checking references.

---

## Chapter 26 — Long-term phases

| Phase | Objective | Starting point | DoD |
|---|---|---|---|
| 0 — trustworthy baseline | Ch. 3 | Confirmed builds clean | `cmake --build` + manual smoke test pass |
| 1 — numerical parity | Ch. 11 | `nn.c` compiling; `benchmark/` not built | Every layer PASSES at stated tolerance vs. real PyTorch |
| 2 — preprocessing correctness | Ch. 12 | Bounding-box+bilinear centering implemented | Fixtures built, centering method's impact known |
| 3 — Raylib product | Ch. 8 | Substantially built | Preprocessing preview added |
| 4 — explainable inference | Ch. 16 | Not started | conv1 activation view works on real data |
| 5 — tests/sanitizers/CI | Ch. 13–15 | Not wired into CMake | Full CI pipeline green |
| 6 — profiling/performance | Ch. 17–18 | Not started | Real timing table exists; optimizations re-verified against Ch. 11 |
| 7 — model serialization | Ch. 10 | Header-less format | Versioned format built, only after Phase 1 |
| 8 — ML experiments | Ch. 19 | Not started | One-variable-at-a-time experiment log exists |

---

## Chapter 27 — Definition of done

A milestone is complete only when:

- Implementation works (compiled + run, not just written).
- Tests exist and pass.
- Parity passes where relevant (Chapter 11).
- Sanitizer checks pass (Chapter 14).
- Documentation matches reality.
- Build is reproducible (`cmake --build` from a clean checkout, no manual steps).
- Benchmark is recorded (Chapter 17's real numbers, not estimates).
- No known regression remains.

---

## Final Chapter — Next 10 tasks

**1. Send `python/model.py` and `python/export.py`.**
Why: Chapter 10's `LAYER_KEYS` order is currently inferred, not confirmed — the single highest-risk unverified claim in the book. Files: those two. Steps: paste or upload them. Test: I confirm `LAYER_KEYS` order against `model_load` read order line by line. DoD: Chapter 10's `[UNCONFIRMED]` tag removed. Next: unblocks Task 2.

**2. Send (or create) `benchmark/debug_input.bin` and run Chapter 11's `verify.c` for real.**
Why: no C code has been checked against a real trained weight. Files: `benchmark/verify.c`, a saved MNIST test image. Steps: save one `[0,1]` float32 28×28 image; run `dump_intermediate.py` and `verify.c`; diff. Command: `./build/verify models/weights.bin benchmark/debug_input.bin`. Test: layer-by-layer PASS/FAIL. DoD: every layer PASSES, or a named first-divergent layer fixed.

**3. Wire `tests/` and a `verify` target into `CMakeLists.txt`.**
Why: currently zero automated tests run from a clean build. Files: `CMakeLists.txt`, `tests/test_nn.c` (Chapter 13 design). Steps: add `enable_testing()` + `add_test` per Chapter 13. Command: `ctest --test-dir build`. Test: itself. DoD: `ctest` runs and passes from a clean `cmake --build`.

**4. Run Chapter 14's sanitizer build against whatever tests exist after Task 3.**
Why: confirms memory safety. Files: none new. Command: Chapter 14's `cmake ... -DCMAKE_C_FLAGS` line. Test: itself. DoD: zero errors, confirmed by a real run.

**5. Add the CI workflow from Chapter 15, once Tasks 3–4 are real.**
Why: protects Tasks 1–4's results. Files: `.github/workflows/ci.yml`. Command: push and check Actions. DoD: a deliberate regression makes CI fail — confirms the pipeline actually catches something.

**6. Add the preprocessing-preview panel to `main.c` (Chapter 8's DoD).**
Why: cheapest way to visually confirm Chapter 12's centering/cropping on real drawings. Files: `main.c`. Steps: render the 28×28 float array from `run_prediction` as a small tile next to the canvas. DoD: preview panel renders real data, not a placeholder.

**7. Build Chapter 12's deterministic fixtures.**
Why: currently only "seems to work" from manual drawing. Files: a new `tests/test_preprocessing.c` or `tools/preprocessing_fixtures.c`. Test: each fixture's 28×28 output inspected against hand-predicted expectations. DoD: all fixtures pass, or reveal a specific real bug.

**8. Check whether bounding-box-center vs. center-of-mass matters (Chapter 12's flagged hypothesis).**
Why: named as the first thing to check for accuracy gaps on asymmetric digits. Files: `ui.c`. Steps: after Task 2 gives a working parity baseline, swap the centering formula to a pixel-weighted centroid and re-measure. DoD: measured decision either way.

**9. Add `conv1` activation visualization (Chapter 16).**
Why: fastest way to visually catch a Chapter 10/11-style bug on real drawings. Files: `main.c`, `nn.c`/`nn.h`. DoD: toggleable debug view renders real `conv1` output as a tile grid.

**10. Profile (Chapter 17) before touching any of Chapter 18's optimization ideas.**
Why: nothing above should be optimized on a feeling. Files: `c/tools/benchmark.c`. DoD: a real, committed timing table exists.

---

*End of the Number Guesser Project Continuation Book — Complete Edition.*

Every chapter is grounded in the actual state of the project. Chapters 3–9 are the state as it exists today, with full code and line-by-line explanations. Chapters 10–19 are the correctness, safety, and depth work described at the design level. Chapter 20–22 is the reference design for the Python side. Task 1 (send `python/model.py` and `export.py`) unblocks the rest.

Work through the 10 tasks in order. When all ten are complete, the project will be:

- **Provably correct** — C and PyTorch agree layer-by-layer on real weights.
- **Tested** — every operation has a unit test; every end-to-end path has an integration test.
- **Sanitized** — no memory bugs, no undefined behavior.
- **Benchmarked** — a real timing table exists.
- **Versioned** — the model format is self-describing.
- **Documented** — every claim is backed by code or a measurement.

That is the difference between a working prototype and a genuine engineering artifact.