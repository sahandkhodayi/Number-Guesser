# Number Guesser — The Complete Engineering Book

**Source-of-truth note, read before anything else:** This book combines two sources. First, the audit against your actual files (`c/include/nn.h`, `c/include/ui.h`, `c/src/nn.c`, `c/src/ui.c`, `c/src/main.c`, `CMakeLists.txt`) — every claim about those files was verified by compiling under your real flags (`-Wall -Wextra -Wpedantic -std=c11`, zero warnings) and running where a display wasn't required. Second, complete reference implementations for every function and tool this book asks you to build, with line-by-line explanations of the reasoning behind each line. Anything about `python/model.py`, `python/export.py`, `tests/*`, `benchmark/*` is either the reference design compatible with your confirmed C loader or explicitly marked **[UNCONFIRMED]** — send those files and this book will be reconciled against them.

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

## Current-state table

| Component | Current implementation | Files | Proven? | Remaining work |
|---|---|---|---|---|
| Tensor | Heap-alloc'd `float*`, channel-first `((c*H+y)*W+x)`, get/set, `tensor_info` | `nn.h`, `nn.c` | Compiled clean; index formula matches PyTorch's `[C,H,W]` | No dedicated unit test file confirmed to exist |
| Linear/ReLU/Argmax | `linear`, `relu`, `relu_tensor`, `argmax` | `nn.c` | Compiled clean; logic matches reference design | No confirmed test coverage in your tree |
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

**Part I — Foundation**
1. [How to work through this book](#chapter-1--how-to-work-through-this-book)
2. [Clean baseline](#chapter-2--clean-baseline)

**Part II — Audit and parity**
3. [Audit the existing C runtime](#chapter-3--audit-the-existing-c-runtime)
4. [Model loading and serialization](#chapter-4--model-loading-and-serialization)
5. [Numerical parity](#chapter-5--numerical-parity)

**Part III — Preprocessing and product**
6. [Preprocessing and domain shift](#chapter-6--preprocessing-and-domain-shift)
7. [The Raylib C product](#chapter-7--the-raylib-c-product)

**Part IV — Quality**
8. [Tests](#chapter-8--tests)
9. [Sanitizers](#chapter-9--sanitizers)
10. [CI](#chapter-10--ci)

**Part V — Depth**
11. [Network visualization](#chapter-11--network-visualization)
12. [Profiling](#chapter-12--profiling)
13. [C optimization](#chapter-13--c-optimization)
14. [ML experiments](#chapter-14--ml-experiments)

**Part VI — Reference**
15. [Mathematics through the project](#chapter-15--mathematics-through-the-project)
16. [D2L + MML learning map](#chapter-16--d2l--mml-learning-map)
17. [AI-agent workflow](#chapter-17--ai-agent-workflow)
18. [Long-term phases](#chapter-18--long-term-phases)
19. [Definition of done](#chapter-19--definition-of-done)

**Final — The next 10 tasks, with full code**

---

## Chapter 1 — How to work through this book

```
read chapter → understand math → inspect current code → 
make ONE change → compile → focused test → compare reference → 
debug → commit → next
```

Never implement several milestones simultaneously — Chapter 5 (parity) and Chapter 6 (preprocessing) are separate risks that fail independently; if both change before you check either, a failure could be either one and you won't know which.

Two rules that must not be bent:

1. **Never claim parity without measurement.** Every number in the benchmark is from a real run. If it hasn't been run, it's marked unknown.
2. **Never optimize before profiling.** Chapter 12 (profiling) strictly precedes Chapter 13 (optimization). Any "obvious" speedup without a measured baseline is a guess, not an optimization.

---

## Chapter 2 — Clean baseline

### Objective
Prove the current tree builds, links, and loads a model, before adding anything.

### Why
Chapters 3–6 all assume "it builds." If that's not true on a clean checkout, everything downstream is built on sand.

### Current state
`nn.c`/`ui.c` compile clean under `-Wall -Wextra -Wpedantic -std=c11` (confirmed). `main.c` was not re-linked against raylib in the last session (tooling issue, not code). `CMakeLists.txt` only defines `number_guesser` — no `tests`/`benchmark` targets exist yet.

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
- `rm -rf build` — remove any stale build tree; prevents "it worked because of leftover artifacts" confusion.
- `cmake -S . -B build` — configure. `-S .` means "source root is current directory", `-B build` means "build tree is `build/`".
- `cmake --build build -j` — build with all available cores. `tee /tmp/build.log` captures output for later inspection.
- The `| tee` is important: it lets you both see the output live **and** grep it later.

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
- `test -x path` — checks the path exists **and** is executable.
- `&&` — only runs `echo OK` if the test succeeded.

**4. Verify the model size.**

```bash
stat -c%s models/weights.bin
```

Line by line:
- `stat` shows file metadata. `-c%s` is the format string for "size in bytes".
- On macOS, replace with `stat -f%z models/weights.bin`.

Expected output: `175016`. If the file is missing or has a different size, `model_load` will reject it — the check is a real fail-fast, not a hopeful assumption.

**5. Run the application.**

```bash
./build/number_guesser
```

Expected: window opens, canvas draws, Predict button is enabled when `weights.bin` is present and correctly sized. If `weights.bin` is missing, `main.c` prints `Warning: could not load models/weights.bin` to stderr and disables Predict (`model_ok` gates the button) — the app does not crash.

### If it fails

- **`find_package(raylib CONFIG REQUIRED)` fails** — raylib is not installed in CMake's config mode. Either install a CMake-aware package (vcpkg, system `libraylib-dev` on Ubuntu) or provide a hint with `-DCMAKE_PREFIX_PATH=/path/to/raylib`.
- **Links but `model_load` always fails** — check `stat` first. If it's 175016 bytes, check that you're running from a directory where `models/weights.bin` is a valid relative path. `main.c` uses the string `"models/weights.bin"` — this is relative to the **current working directory**, not to the binary's location.
- **Window doesn't open at all** — no display available. On WSL, ensure WSLg (Windows 11) or an X server (Windows 10) is running.

### Definition of done
`cmake --build` succeeds with zero warnings; `./build/number_guesser` opens a window; Predict is enabled when a correctly-sized `weights.bin` is present, disabled with the specific stderr message when it isn't.

### Next
Chapter 3 — audit what the C runtime actually does, since it compiles.

---

## Chapter 3 — Audit the existing C runtime

Audit, not rewrite. Every function below already exists in `nn.c` and compiles clean. The goal is understanding, not changes.

### `tensor_alloc` / `tensor_free`

**Purpose:** own a `channels × height × width` heap buffer.

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

- `Tensor t = {0};` — declares a local Tensor struct and zero-initializes it. `{0}` sets every byte to 0, which for a `float*` means NULL and for the `int` fields means 0. This is defensive: if we later return early, the caller still gets a valid (if empty) struct.
- `t.channels = channels;` etc. — copy the shape metadata into the struct.
- `size_t n = (size_t)channels * (size_t)height * (size_t)width;` — compute total element count. **Every operand is cast to `size_t` individually.** This is more defensive than casting only the first: with all operands as `size_t`, the entire multiplication happens in unsigned 64-bit arithmetic even if the individual `int` values are large. If `channels=32`, `height=28`, `width=28`, then `n=25088` — fits in `int`, but the `size_t` arithmetic protects against future changes with larger dimensions.
- `t.data = calloc(n, sizeof(float));` — **`calloc`, not `malloc`.** `calloc` zero-initializes. This matters because an unwritten cell reads as `0.0` (a suspicious but debuggable value) rather than garbage (which might look like a plausible-but-wrong float). It also checks for multiplication overflow internally: `calloc(0x10000000, 4)` would be caught rather than silently wrapping.
- The NULL check — allocation can fail if the system is out of memory. Exiting loudly is correct behavior for a program that cannot continue without the memory.
- `return t;` — returns by value. The struct is 24 bytes (one pointer + three ints), so the copy is cheap.

```c
void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = t->height = t->width = 0;
}
```

- `free(t->data);` — releases the heap block.
- `t->data = NULL;` — **defensive NULLing.** After freeing, the pointer is invalid. Setting it to NULL means a subsequent accidental `tensor_get(t, ...)` dereferences NULL and crashes **immediately with a clear stack trace**, rather than silently reading freed memory (which might produce a plausible wrong value that is far harder to trace). ASan catches both, but the NULLing makes the failure mode visible even without ASan.
- `t->channels = t->height = t->width = 0;` — zeroes the shape. A caller that checks `t->channels == 0` knows the tensor was freed.

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

- `(size_t)c * (size_t)t->height` — convert the channel index into "how many rows of pixels come before this channel starts." If each channel has `height` rows, and we want channel `c`, then `c * height` rows precede it.
- `+ (size_t)y` — add the row offset within this channel. If we want row `y`, we've now counted `c * height + y` rows total.
- `* (size_t)t->width` — convert rows to individual float elements. Each row has `width` floats, so `(c * height + y) * width` floats precede this row.
- `+ (size_t)x` — add the column offset. Final result is the exact index of element `(c, y, x)` in the flat buffer.

**Mathematical contract:** `index(c, y, x)` is a bijection from `[0,C) × [0,H) × [0,W)` onto `[0, C·H·W)`. Every valid `(c, y, x)` maps to a distinct offset — no aliasing — **as long as the caller respects the tensor's actual bounds.** Nothing here enforces that; an out-of-range `c`/`y`/`x` silently indexes past the buffer, exactly like raw C array indexing.

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

- `for (int o = 0; o < out_features; ++o)` — iterate over output neurons. `++o` is idiomatic C for a loop counter; no functional difference from `o++` in this context, but traditional.
- `float sum = b[o];` — seed the accumulator with the bias for this output. Starting from the bias (rather than adding it at the end) is a small optimization: it saves one addition per output.
- `const float *row = W + o * in_features;` — **pointer arithmetic.** `W` is `float*`. Adding `o * in_features` to it produces a pointer to the element at offset `o * in_features`. This is exactly the start of row `o` in the row-major layout. `const` documents that we won't modify the weights through this pointer.
- `sum += row[i] * x[i];` — accumulate the dot product. `row[i]` accesses `W[o * in_features + i]`, which matches the PyTorch `Linear.weight[o, i]` in row-major.
- `y[o] = sum;` — store the result.

**PyTorch equivalent:** `nn.Linear(in_features, out_features)` computing `y = x @ W.T + b`. **Ownership:** writes into a caller-provided buffer; never allocates. This is a deliberate difference from `conv2d`.

### `relu` / `relu_tensor` / `argmax`

```c
void relu(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}
```

In-place `max(0, x)`. Uses `if` rather than `fmaxf(x, 0)` for two reasons: (1) it avoids a function call that some compilers won't inline, and (2) it only writes when the value is negative, saving a memory store for positive values. Both are micro-optimizations that matter at 25088 elements per call.

```c
void relu_tensor(Tensor *t) {
    relu(t->data, t->channels * t->height * t->width);
}
```

Takes the flat data pointer and the total element count. ReLU is pointwise — it doesn't care about shape — so applying it to the flat array is correct regardless of the tensor's shape.

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

- `int best_idx = 0; float best_val = x[0];` — initialize with the first element.
- `for (int i = 1; i < n; ++i)` — start at 1, not 0, because 0 is already the initial best.
- `if (x[i] > best_val)` — uses `>` not `>=`, so ties break toward the earlier index. Matches PyTorch's `torch.argmax` default behavior.

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

The formula `floor((N + 2P - K) / S) + 1` computes the output dimension. For `conv1` in this project: `N=28, P=1, K=3, S=1`:
```
(28 + 2*1 - 3) / 1 + 1 = (28 + 2 - 3) + 1 = 27 + 1 = 28
```

Confirms `H_out = H_in`. This is why every conv in this architecture preserves spatial dimensions — not a coincidence, but the specific `(K=3, S=1, P=1)` combination.

Integer division in C truncates toward zero for positive numbers, so `(x) / 1` is exact and `(x) / stride` for larger `stride` truncates the way `floor` does. For negative values this would differ, but output dimensions are always positive here.

**Output allocation.**

```c
Tensor out = tensor_alloc(out_channels, out_h, out_w);
```

Allocates the output. **This is where `conv2d` differs from `linear`:** `conv2d` allocates and returns, `linear` writes into a caller-provided buffer. The reason: `conv2d`'s output shape is not knowable by the caller without duplicating the output-size formula. `linear`'s output size is just `out_features`, which the caller already knows.

**Nested loops.**

```c
for (int oc = 0; oc < out_channels; ++oc) {       // output channel
    for (int oy = 0; oy < out_h; ++oy) {          // output row
        for (int ox = 0; ox < out_w; ++ox) {      // output column
```

The outer three loops iterate over the output tensor. For each output location, we compute one value.

**Accumulator.**

```c
float sum = bias[oc];
```

Start from the bias for this output channel. This is analogous to `linear`'s `float sum = b[o];`.

**Inner loops.**

```c
for (int ic = 0; ic < input->channels; ++ic) {
    for (int ky = 0; ky < k; ++ky) {
        for (int kx = 0; kx < k; ++kx) {
```

Iterate over every (input channel, kernel row, kernel column) combination that contributes to this output. Total iterations per output: `in_channels × k × k`. For `conv2`: `32 × 3 × 3 = 288`.

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

If the tap is outside the real input, skip it. This is mathematically identical to zero-padding: a tap outside contributes `0 × weight = 0` to the sum. Skipping the multiplication gives the same result without allocating a padded copy.

**Input read.**

```c
float in_val = tensor_get(input, ic, iy, ix);
```

Standard tensor access via the (correctly-formula'd) `tensor_get`.

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

**Accumulate.**

```c
sum += in_val * weights[w_index];
```

**Store.**

```c
tensor_set(&out, oc, oy, ox, sum);
```

**Return.**

```c
return out;
```

Ownership transfers to the caller. The caller (`model_forward`) frees it after the next layer has consumed it.

**Cost analysis.** Total ops = `out_channels × out_h × out_w × in_channels × k²`. For `conv2`: `32 × 28 × 28 × 32 × 3 × 3 = 7,225,344` multiply-adds. For `conv1`: `32 × 28 × 28 × 1 × 3 × 3 = 225,792`. Chapter 12 measures whether this theoretical cost is what actually dominates the wall-clock time.

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

- **One channel loop, not two.** Unlike `conv2d`, pooling never mixes channels — each output channel is derived purely from its own input channel. So the outer `for (int c ...)` plays double duty as both input- and output-channel index.
- **No padding term, no bounds check.** This project's pooling uses `k=2, stride=2, no padding`. Every window is guaranteed fully inside the input, so no `if (iy < 0 || ...) continue;` is needed.
- **`int iy = oy * stride + ky;`** — no `- pad` term because `pad = 0` here.
- **`float best = -INFINITY;`** — see the explanation below.

**4. Indexing.** `iy = oy*stride + ky`, `ix = ox*stride + kx`. Compare with `conv2d`'s `iy = oy*stride - pad + ky`. The pooling version is the same formula with `pad = 0` — the code is literally the conv formula specialized.

**5. 2×2 stride 2 substituted.** Output formula: `out_dim = floor((in_dim - 2) / 2) + 1`.

- For `in_dim = 28`: `floor(26 / 2) + 1 = 13 + 1 = 14`. ✓
- For `in_dim = 14`: `floor(12 / 2) + 1 = 6 + 1 = 7`. ✓

Matches the architecture table exactly.

**6. Why `-INFINITY` is correct and `0` would be wrong.**

The current architecture applies ReLU before every pooling, so the input to `pool1` and `pool2` is non-negative. In that specific case, a `0` sentinel would **coincidentally** work — every real value is `>= 0`, so if all 4 values in a window are `0`, `best = 0` is correct; if some value is positive, it beats `0`.

But this is fragile in three ways:

1. Any future architecture change that removes ReLU, or moves the pooling before ReLU, breaks silently. All-negative inputs would produce `best = 0` for windows whose true max is negative, a mathematically wrong result that "looks fine" numerically (it's a valid float).

2. A test with negative inputs would be impossible to pass — the sentinel itself would be the bug.

3. It encodes a hidden dependency: "this code assumes the caller applied ReLU first." Comments like that get lost; `-INFINITY` doesn't need the comment because it is provably correct for any input range.

`-INFINITY` (from `<math.h>`) is a true lower bound on all `float` values except `NaN`. Any real number beats it, so the first comparison in the window always sets `best`, and subsequent comparisons correctly find the max.

An alternative is `-FLT_MAX`, which is the minimum finite `float`. That works too but is not quite as strong: if the input ever contains `-INFINITY` (possible in some networks), `-FLT_MAX` would incorrectly "win" the comparison. `-INFINITY` handles that case correctly.

An earlier draft of this project used `-1e30f` — a very large negative sentinel. That works for inputs bounded above `-1e30`, which is true for all realistic image data, but it's not a principled lower bound. The current code already uses the better choice.

**7. PyTorch equivalent.** `nn.MaxPool2d(kernel_size=2, stride=2)`, no padding, `ceil_mode=False` (the default). The `ceil_mode=False` corresponds to the `floor` in the output-size formula above; `ceil_mode=True` would round up and produce different (larger) output dimensions.

**8–10. Testing.** See Chapter 8 for the test that exercises this function, including the negative-input case that would catch a `0` sentinel bug.

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
- `tensor_free(&a)` — frees `a` **immediately** after its last read. Not at the end of the function — right here. This is what keeps peak memory bounded: at any moment, at most two intermediate tensors are alive.
- `Tensor p1 = maxpool2d(&b, 2, 2)` — allocates `p1`. We now own `p1`, `b` is no longer needed.
- `tensor_free(&b)` — frees `b`.
- This pattern repeats for `c`, `d`, `p2`.
- The final `linear` reads `p2.data` — the flat buffer of the last pooling output. **This is the flatten.** There is no copy: `linear` just iterates 1568 floats starting at `p2.data[0]`. `in_features = 32 * 7 * 7 = 1568`.
- `tensor_free(&p2)` — frees `p2` after the linear layer has consumed it.

**Peak memory:** ~2 tensors at a time, not 8. This is by construction, not by accident.

**Ownership of `input`:** `model_forward` borrows `input` (it's `const Tensor *`). It never frees it. The caller owns it and is responsible for freeing it.

### Future optimization note (do not act on this before Chapter 12)

`conv2d`'s six nested loops are the natural place to look for cache-locality or loop-reordering wins. But there is **no profiling data yet** to say whether `conv2d` is even the bottleneck. The Big-O note in this chapter predicts `conv2/3/4` dominate, but that's a prediction, not a measurement. Do not guess — Chapter 12 measures first, Chapter 13 optimizes second.

### Definition of done

Every function above has been read carefully enough that you can explain, from memory:

- What its inputs and outputs are.
- Where its memory comes from and who owns it.
- Its PyTorch equivalent.
- One edge case where a naive implementation would be wrong and this one is right.

If any of those are unclear, re-read that function before continuing.

### Next

Chapter 4 — model loading and the fragility of a header-less format.

---

## Chapter 4 — Model loading and serialization

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

`sizeof(CnnModel)` — **confirmed by actually compiling and running a size check in the earlier session** — is exactly 175016 bytes. Zero compiler-inserted padding. This is expected: every member is a `float` array, all 4-byte aligned, so the struct layout is exactly the array sizes concatenated. But it's worth knowing this assumption exists: if a non-`float` field were ever added (say, a `char version;`), the compiler could insert 3 bytes of padding to re-align the next `float` array, and `sizeof(CnnModel)` would silently stop equaling the sum of array sizes.

### The loader

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
    err |= read_floats(f, m->conv1_b, sizeof m->conv1_b / sizeof m->conv1_b[0], "conv1_b");
    err |= read_floats(f, m->conv2_w, sizeof m->conv2_w / sizeof m->conv2_w[0], "conv2_w");
    err |= read_floats(f, m->conv2_b, sizeof m->conv2_b / sizeof m->conv2_b[0], "conv2_b");
    err |= read_floats(f, m->conv3_w, sizeof m->conv3_w / sizeof m->conv3_w[0], "conv3_w");
    err |= read_floats(f, m->conv3_b, sizeof m->conv3_b / sizeof m->conv3_b[0], "conv3_b");
    err |= read_floats(f, m->conv4_w, sizeof m->conv4_w / sizeof m->conv4_w[0], "conv4_w");
    err |= read_floats(f, m->conv4_b, sizeof m->conv4_b / sizeof m->conv4_b[0], "conv4_b");
    err |= read_floats(f, m->fc_w,    sizeof m->fc_w    / sizeof m->fc_w[0],    "fc_w");
    err |= read_floats(f, m->fc_b,    sizeof m->fc_b    / sizeof m->fc_b[0],    "fc_b");

    fclose(f);
    return err ? -1 : 0;
}
```

Line by line:

- **`fopen(path, "rb")`** — opens in binary read mode. The `b` is critical on Windows: without it, `\n` bytes would be silently translated to `\r\n`, corrupting the float stream.
- **NULL check** — the file might not exist, or might not be readable.
- **`fseek(f, 0, SEEK_END); long file_size = ftell(f); fseek(f, 0, SEEK_SET);`** — the standard "how big is this file" idiom. Seek to end, ask for the position, seek back to start. `ftell` returns the byte offset, which for a file just seeked to end equals the file size.
- **Error checks on the seeks and `ftell`** — `fseek` returns nonzero on failure. If the file is not seekable (e.g., a pipe), we cannot proceed. `ftell` returns `-1` on error.
- **`(unsigned long)file_size != sizeof(CnnModel)`** — the size check. The cast to `unsigned long` matches `sizeof`'s return type on most platforms and avoids a signed/unsigned comparison warning. This check is a real improvement over a hand-maintained byte constant: if `CnnModel` gains a layer, `sizeof(CnnModel)` updates automatically, and the check remains correct. A separate `#define WEIGHTS_FILE_BYTES 175016` would silently drift out of sync.
- **`read_floats(f, m->conv1_w, sizeof m->conv1_w / sizeof m->conv1_w[0], "conv1_w")`** — the count expression is the idiomatic C way to say "number of elements in a fixed-size array." `sizeof m->conv1_w` is the byte size; `sizeof m->conv1_w[0]` is the byte size of one element; the quotient is the element count. The compiler evaluates this at compile time — no runtime cost.
- **`err |= read_floats(...)`** — bitwise OR accumulates errors. If any call returns -1 (all bits set in two's complement), `err` becomes -1 and stays -1. This means we see **all** read failures in stderr output, not just the first. That's valuable when a broken file has multiple problems.
- **`fclose(f)`** — close the file handle. Always close.
- **`return err ? -1 : 0`** — returns -1 if any error occurred, 0 otherwise.

### How a `model.py` change silently breaks export/load parity

Suppose you add a fifth conv layer to `model.py` without touching `CnnModel` in `nn.h`:

- `state_dict()` gains new keys (e.g., `block_3.0.weight`, `block_3.0.bias`).
- `LAYER_KEYS` (if hand-written and not regenerated) still lists only the old 10 keys. Depending on the export code, either:
  - It KeyErrors on the missing key (loud failure — good), or
  - Worse: it silently writes the old 10 tensors in the old order and drops the new layer's weights entirely.
- `model_load`'s `sizeof(CnnModel)` check **still passes**, because the file size also doesn't include the new layer's weights.

**The size check cannot catch a `model.py` architecture change that both sides forgot to propagate.** This is exactly the failure mode Chapter 5's parity check exists to catch: a file that loads with zero errors and produces a plausible-looking wrong number.

### The versioned format — design (do not implement before Chapter 5 passes)

The current format is fragile in ways the size check cannot fix. The replacement is a self-describing format with these fields:

```
[4 bytes]  magic        "NGSR"  (4 ASCII bytes, unique enough to catch random files)
[4 bytes]  version      uint32, e.g. 1
[4 bytes]  arch_id      uint32 — hash or enum of the architecture, so a mismatched
                        model.py / nn.h pair fails loudly instead of silently
[4 bytes]  dtype        uint32, e.g. 0 = float32
[4 bytes]  tensor_count uint32
[tensor_count × (
    name_len (uint32)
    name bytes
    ndims (uint32)
    shape[ndims] (uint32 each)
)]
[payload]               raw tensor bytes, same order as metadata
[4 bytes]  checksum     CRC32 over the payload
```

Properties this buys you:

- **Magic** — a random file will not start with `NGSR`; the reader rejects it immediately.
- **Version** — future format changes are backward-compatible if the reader checks the version and handles both.
- **`arch_id`** — a mismatched pair fails at load time, not at prediction time.
- **`dtype`** — allows future float16 or int8 weights without a format break.
- **`tensor_count` and shape metadata** — the reader knows exactly what to expect; a truncated file fails on the first missing tensor, with a named error, not a silent short read.
- **Checksum** — catches transmission or disk corruption.

**Do not build this before Chapter 5 (parity) passes on the current header-less format.** A format change is a real risk to take on for its own sake; correctness must be proven on the simpler format first. Once parity is confirmed, migrating to the versioned format is a self-contained change.

### Definition of done

- `sizeof(CnnModel)` confirmed to equal the export size on your machine (done).
- `LAYER_KEYS` order confirmed against real `export.py` (blocked — send the file).
- `model_load` rejects a wrong-size file (Chapter 8 will write the test).

### Next

Chapter 5 — the only thing that actually proves the loader's read order is right: comparing real C output against real PyTorch output.

---

## Chapter 5 — Numerical parity

### Objective

Prove `model_forward`'s output matches PyTorch's output, for the same input and the same trained weights, layer by layer.

### Why

Matching the final predicted digit is not enough. Two wrong implementations can agree on a digit by coincidence (10 classes, ~10% chance of agreeing on nonsense alone), and a real bug that shows up two layers before the end can still happen to argmax to the same digit on some inputs and a different one on others — making the bug intermittent and much harder to trust or debug. Layer-by-layer comparison locates a divergence instead of hiding it behind a coin flip.

### Theory: NCHW, contiguous memory, and why layout agreement matters

PyTorch's default tensor layout is row-major/C-contiguous. For a `[1, C, H, W]` tensor, element `(c, y, x)` sits at offset `(c*H + y)*W + x` — **exactly** `tensor_get`'s formula. This is not a coincidence to verify; it is a design constraint both sides were built to satisfy. Chapter 4 already found the place it could break (an export order that doesn't match the read order). This chapter checks that constraint empirically.

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

- **`t.detach().flatten().numpy()`** — `.detach()` removes the autograd graph; `.flatten()` makes it 1D; `.numpy()` shares memory with the tensor.
- **`[0]` at the end of `model.block_1[0](x)`** — strips the batch dimension. The model expects `(1, 1, 28, 28)`; the output is `(1, 32, 28, 28)`; `[0]` gives `(32, 28, 28)`, matching what the C code produces.
- **`with torch.no_grad():`** — disables autograd for the whole block. Faster, less memory, correct for inference.
- **`flat.unsqueeze(0)`** — adds a batch dimension back for the linear layer, which expects `(N, in_features)`.

### The reference C dump — `c/tools/verify.c`

```c
#include "nn.h"
#include <stdio.h>

static void dump(const Tensor *t, const char *label) {
    printf("%-8s shape=(%d, %d, %d)  first 5=[", label,
           t->channels, t->height, t->width);
    int n = t->channels * t->height * t->width;
    int show = n < 5 ? n : 5;
    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], i == show - 1 ? "" : ", ");
    }
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *weights_path = argc > 1 ? argv[1] : "../models/weights.bin";
    const char *input_path   = argc > 2 ? argv[2] : "../models/debug_input.bin";

    CnnModel m;
    if (model_load(&m, weights_path) != 0) return 1;

    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(input_path, "rb");
    if (f != NULL) {
        fread(input.data, sizeof(float), 28 * 28, f);
        fclose(f);
    }
    dump(&input, "input");

    Tensor a = conv2d(&input, m.conv1_w, m.conv1_b, 32, 3, 1, 1);
    dump(&a, "conv1");
    tensor_free(&input);
    relu_tensor(&a);
    dump(&a, "relu1");

    Tensor b = conv2d(&a, m.conv2_w, m.conv2_b, 32, 3, 1, 1);
    dump(&b, "conv2");
    tensor_free(&a);
    relu_tensor(&b);
    dump(&b, "relu2");

    Tensor p1 = maxpool2d(&b, 2, 2);
    dump(&p1, "pool1");
    tensor_free(&b);

    Tensor c = conv2d(&p1, m.conv3_w, m.conv3_b, 32, 3, 1, 1);
    dump(&c, "conv3");
    tensor_free(&p1);
    relu_tensor(&c);
    dump(&c, "relu3");

    Tensor d = conv2d(&c, m.conv4_w, m.conv4_b, 32, 3, 1, 1);
    dump(&d, "conv4");
    tensor_free(&c);
    relu_tensor(&d);
    dump(&d, "relu4");

    Tensor p2 = maxpool2d(&d, 2, 2);
    dump(&p2, "pool2");
    tensor_free(&d);

    printf("flat     shape=(1, 1, %d)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           p2.channels * p2.height * p2.width,
           p2.data[0], p2.data[1], p2.data[2], p2.data[3], p2.data[4]);

    float logits[10];
    linear(m.fc_w, m.fc_b, p2.data, logits, 32 * 7 * 7, 10);
    tensor_free(&p2);

    printf("logits   shape=(1, 10)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           logits[0], logits[1], logits[2], logits[3], logits[4]);
    printf("\npredicted digit: %d\n", argmax(logits, 10));
    return 0;
}
```

Line by line of the dump format:

- **`"%-8s"`** — left-justified 8-character-wide label. Matches the Python `f"{label:8s}"`.
- **`"shape=(%d, %d, %d)"`** — the C tensor has three dimensions.
- **`"%.4f"`** — 4 digits after the decimal. Matches Python's `np.round(..., 4)`.
- **`"i == show - 1 ? "" : ", "`** — separates values by commas but not after the last.

**Special cases at the end:**

- **`flat`** — the flat output doesn't fit the `Tensor` format (it's a 1D view). We print it manually with shape `(1, 1, 1568)` to match the tensor convention, and only the first 5 values.
- **`logits`** — printed with shape `(1, 10)`.

**Compile and run:**

```bash
cd c/tools
gcc -O2 -Wall -Wextra -std=c11 -I../include verify.c ../src/nn.c -o verify -lm
./verify ../models/weights.bin ../models/debug_input.bin > ../../notes/c_stages.txt
```

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
- **`[x.strip() for x in shape_str.split(",") if x.strip()]`** — splits "1, 28, 28" into `["1", "28", "28"]`, handles the trailing comma in single-element shapes like `(10,)`.
- **`diffs = [abs(a - b) ...]`** — elementwise comparison of the first 5 values.
- **`sys.exit(0 if ok else 1)`** — CI-friendly exit code.

### First-mismatch debugging tree

Follow this **in order**. Never jump ahead.

```
input matches?
  NO → check debug_input.bin's byte count/scale before touching C code
  YES ↓
conv1 matches?
  NO → conv1_w/conv1_b export order or values wrong (Chapter 4)
  YES ↓
relu1 matches?
  NO → ReLU applied to wrong buffer, applied twice, or skipped
  YES ↓
conv2 matches?
  NO → same as conv1, but for conv2_w/conv2_b specifically
  YES ↓
...
logits match?
  NO → fc_w/fc_b export order, or in_features miscount (should be 1568)
  YES ↓
DONE — argmax agreement is now a consequence of real numerical agreement, not luck.
```

Always fix the **first** divergence, then re-run the whole comparison. A bug at `conv2` makes everything after it "wrong" too, but only `conv2` is the actual bug — debugging `pool2` first is chasing a symptom.

### Tolerance

Report per-stage: shape equality (hard requirement, no tolerance), max absolute error, mean absolute error, and PASS/FAIL against a stated threshold.

`~1e-4` to `~1e-5` max-absolute-error is ordinary floating-point summation-order noise between PyTorch's kernels and this project's plain loops. Anything larger — especially a wrong sign or wrong order of magnitude — is a real bug. Never accept "floating point" as an explanation for a gap bigger than that.

### Definition of done

Every stage from `input` to `logits` reports PASS at the stated tolerance, for at least one real MNIST test image, against your actual trained weights.

### Next

Chapter 6 — even a numerically-perfect C forward pass is only as good as the preprocessing feeding it; that's a separate risk, checked next.

---

## Chapter 6 — Preprocessing and domain shift

### Objective

Confirm what `canvas_to_mnist_input` (in `ui.c`) actually does, and whether it matches the training-side preprocessing closely enough.

### Current state — this is a significant improvement over a naive downsample

Your `canvas_to_mnist_input` does the following:

1. Scans the 280×280 canvas for pixels above `threshold = 0.02f`, finds the bounding box (`min_x, min_y, max_x, max_y`).
2. Returns an all-zero 28×28 input if the canvas is empty (`max_x < 0`) — a real, sensible guard.
3. Takes `side = max(box_w, box_h)` — a square crop preserving aspect ratio — and adds a margin: `margin = side/10; side += 2*margin`.
4. Bilinearly resamples that square crop into a **20×20** region (`sample_bilinear`), not directly into 28×28.
5. Centers the 20×20 region inside the 28×28 output with a fixed 4-pixel border (`offset = (28 - 20) / 2 = 4`).

### Theory: why this specific design

Real MNIST's own generation process normalizes each digit into a 20×20 bounding box (preserving aspect ratio) and centers it in a 28×28 field. Your code's `target = 20` and `offset = 4` reproduce that convention directly, not by coincidence. This is a materially closer match to MNIST's actual preprocessing than a plain 10:1 box-filter downsample would be.

### Where it still might not match, and needs checking not assuming

- **Centering method.** Your code centers by **bounding-box center** (`(min_x + max_x) / 2, (min_y + max_y) / 2`). Real MNIST centers by **center of mass** of the ink (a pixel-value-weighted centroid). For symmetric digits these coincide closely; for asymmetric ones (e.g., a "7" with a long diagonal stroke concentrated to one side), they can differ by a few pixels. This is a real, specific, checkable hypothesis for any accuracy gap on asymmetric digits — not confirmed to matter yet, but named as the first thing to check.

- **Interpolation.** `sample_bilinear` uses bilinear. PyTorch/PIL would use a different algorithm (typically bicubic for downsampling). MNIST ships at 28×28 so there's no PyTorch-side resize to compare against — this is purely an artifact of your UI drawing at higher resolution than the model expects, not a training-preprocessing mismatch.

- **Threshold (`0.02f`).** An implicit assumption about what counts as "drawn" vs. background noise — reasonable, but arbitrary. Worth knowing it exists if a very faint stroke is ever treated as an empty canvas.

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

Line by line:

- **`print_ascii`** — maps value ranges to characters. `#` for strong stroke (>0.5), `+` for medium (>0.2), `.` for faint (>0.05), space for background. The thresholds are chosen so the visual output distinguishes stroke intensity levels.
- **`fixture`** — a helper that clears the canvas, calls a drawing function, runs the pipeline, prints. `AppState` is stack-allocated (it contains a 280×280 float array — 313,600 bytes, well within the 8 MB default stack limit).
- **`draw_circle`** — uses `cosf`/`sinf` to step around a radius-80 circle at 3-degree increments (120 points).

**Compile and run:**

```bash
cd c
gcc -O2 -Wall -Wextra -std=c11 -Iinclude \
    tools/preprocessing_fixtures.c src/ui.c -o fixtures -lm
./fixtures > ../notes/fixtures.txt
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

The character thresholds match the fixture tool exactly, so the ASCII outputs are directly comparable.

### Definition of done

Fixtures exist and are inspectable. The effect of bounding-box-center vs. center-of-mass on asymmetric digits is either confirmed negligible or the code is replaced with center-of-mass. Behavior on blank/tiny/huge/off-center inputs is explicitly tested, not just "seems to work" from one manual draw.

### Next

Chapter 7 — the UI this preprocessing feeds.

---

## Chapter 7 — The Raylib C product

### Objective

Audit `main.c` — it's already substantially built.

### What's already built

- **Draw → clear → predict** — mouse drawing via `canvas_draw_point`/`canvas_draw_line` (continuous strokes tracked via `AppState.is_drawing`/`last_mouse_x`/`last_mouse_y`). `C` key or CLEAR button. `Enter` key or PREDICT button.
- **A real fix already made:** buttons use `IsMouseButtonPressed`, not `IsMouseButtonDown`. Your comment explains why: `Down` re-triggers every frame while held; `Pressed` fires once. Drawing correctly still uses `Down` (a brush stroke *should* continue while dragging). This is a deliberate, correct distinction — not an inconsistency.
- **Ten-class probabilities** — `draw_probability_bars`, color-coded by `prob_color(p)`.
- **Confidence display** — green >80%, yellow >50%, red otherwise.
- **FPS counter** — `DrawFPS`.

### What's missing

- **Preprocessing preview** — the canvas shown is the raw 280×280 drawing, not the actual 28×28 tensor `model_forward` receives. This is the most useful missing piece for visually debugging Chapter 6's centering/cropping behavior.
- **Debug mode / activation visualization** — Chapter 11.

### The preprocessing preview panel

Add to `main.c`, near the other `draw_*` helpers:

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

Line by line:

- **`cell = size / 28`** — how many screen pixels each tensor cell becomes. For a 112-pixel preview and 28 cells, `cell = 4`.
- **`v = fmaxf(0.0f, fminf(1.0f, v))`** — clamp to `[0, 1]` for display. Real values are always in range, but clamping is defensive.
- **`(unsigned char)(v * 255.0f)`** — map to `[0, 255]`.
- **`DrawRectangle`** — draws a `cell × cell` filled square. Using `DrawRectangle` (one call per cell) is faster than `DrawPixel` (one call per pixel) since each cell covers `cell²` screen pixels.
- **`DrawRectangleLines`** — border around the whole preview.
- **`DrawText`** — label.

In `run_prediction`, after computing `mnist_input`:

```c
static float last_mnist_input[28 * 28];

static void run_prediction(AppState *app, const CnnModel *model) {
    float mnist_input[28 * 28];
    canvas_to_mnist_input(app, mnist_input);

    /* Save a copy for the preview panel. */
    memcpy(last_mnist_input, mnist_input, sizeof(last_mnist_input));

    Tensor input = tensor_alloc(1, 28, 28);
    for (int i = 0; i < 28 * 28; i++) input.data[i] = mnist_input[i];
    /* ... */
}
```

Line by line:

- **`static float last_mnist_input[28 * 28]`** — a file-scope array holding the last prediction's input. `static` here means it persists across function calls (function-local `static` in C has program lifetime, unlike ordinary locals).
- **`memcpy(last_mnist_input, mnist_input, sizeof(last_mnist_input))`** — copies 784 floats. `sizeof(last_mnist_input)` is `784 * 4 = 3136` bytes.

In the drawing section:

```c
draw_mnist_preview(last_mnist_input, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 90, 112);
```

### Definition of done

`main.c` renders the actual 28×28 float array `run_prediction` computes, as a small grayscale tile next to the canvas. Draw a deliberately off-center digit; the preview should show it re-centered.

### Next

Chapter 8 — none of Chapters 3–7's claims are protected against regression without tests.

---

## Chapter 8 — Tests

**[UNCONFIRMED]**: `tests/` exists per your README but wasn't provided, and `CMakeLists.txt` doesn't build it regardless. The design below is against your *actual* confirmed function signatures.

### The test table

| Test | Protects against |
|---|---|
| `tensor_get`/`set` round-trip + neighbor-unaffected check | A wrong offset formula *aliasing* two distinct cells — silent corruption, not a crash |
| `conv2d`, no padding, hand-computed 3×3 input / 2×2 kernel | Basic accumulation/loop-nest arithmetic |
| `conv2d`, `k=3,s=1,p=1`, center-tap-only kernel | Bounds-check/padding logic specifically, using this project's actual conv config |
| `maxpool2d`, hand-picked 4×4 → 2×2, including an all-negative window | Window placement + max selection; the negative case catches a `0` sentinel bug |
| `model_load`, synthetic weights (`float[i] = i`) | Wrong read order/count — `conv1_b[0] == 288.0` proves `conv1_w`'s 288 floats and `conv1_b` don't overlap or gap |
| `model_load`, wrong-size file | The `sizeof(CnnModel)` guard actually rejects, not just "looks like it should" |
| `canvas_to_mnist_input`, blank/tiny/off-center/huge fixtures | Chapter 6's bounding-box+bilinear logic specifically |
| End-to-end: real MNIST image → `model_forward` → correct digit | Wiring-level regressions across the whole pipeline |
| Benchmark/parity (Chapter 5) | The one thing unit tests structurally can't catch: correct-looking code computing the wrong number |

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
    /* out[0][0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
       out[0][1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
       out[1][0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
       out[1][1] = 5*1 + 6*0 + 8*0 + 9*1 = 14 */
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
    /* All-negative input: catches a 0-sentinel bug. */
    Tensor in = tensor_alloc(1, 2, 2);
    float data[] = {-5, -3, -8, -1};
    memcpy(in.data, data, sizeof(data));
    Tensor out = maxpool2d(&in, 2, 2);
    /* max(-5, -3, -8, -1) = -1 */
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
- **`test_tensor` neighbor checks** — after setting one cell, we verify the immediate neighbors are still 0. This catches an offset formula that aliases two different `(c, y, x)` triples to the same memory address — silent corruption that a "does the value I just wrote come back?" test alone would miss.
- **`test_maxpool2d_negative`** — the specific test that would catch a `0` sentinel in `maxpool2d`. If the implementation used `float best = 0.0f;` instead of `-INFINITY`, this test would fail (the max would come back as 0 instead of -1). This is exactly the case that motivated the `-INFINITY` choice in Chapter 3.

### Wiring into CMake

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

Every row in the table has a real, compiling test file. `ctest` runs them all from a clean `cmake --build`.

### Next

Chapter 9 — memory-safety confirmation the tests above don't give you on their own.

---

## Chapter 9 — Sanitizers

### Workflow, using your actual CMake project

```bash
cmake -S . -B build-asan \
    -DCMAKE_C_FLAGS="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

Line by line of the flags:

- **`-g`** — debug symbols. Required for readable stack traces.
- **`-O1`** — light optimization. `-O0` is too slow for benchmarks; `-O2` can inline away the bug. `-O1` balances visibility and speed.
- **`-fsanitize=address,undefined`** — enables both sanitizers.
- **`-fno-omit-frame-pointer`** — keeps frame pointers. Without this, ASan traces can be misleading.

### What each sanitizer catches

**AddressSanitizer (ASan):**
- Heap buffer overflow — writing past a `tensor_alloc`'d buffer.
- Stack buffer overflow — writing past a local array.
- Use-after-free — accessing a `Tensor` after `tensor_free`. The `t->data = NULL` defensive set in your `tensor_free` turns this into an immediate crash even without ASan.
- Double-free — calling `tensor_free` twice.
- Memory leaks — `tensor_alloc` without `tensor_free`.

**UndefinedBehaviorSanitizer (UBSan):**
- Signed integer overflow — using `int` where the product exceeds `INT_MAX`.
- Division by zero.
- Misaligned pointer access.
- Null pointer dereference.
- Invalid shifts.

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

Every test from Chapter 8 runs clean (zero errors) under both sanitizers simultaneously.

### Next

Chapter 10 — automate Chapters 8 and 9 so they run on every push.

---

## Chapter 10 — CI

Build only after Chapters 8–9 are real locally. A CI job running tests that don't yet cover real-weight correctness protects less than it looks like it does.

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

Line by line of the workflow:

- **`on: push/pull_request`** — runs on every push and every PR.
- **`runs-on: ubuntu-latest`** — Ubuntu has `libraylib-dev` as a system package, making the CI job simple.
- **`apt-get install ...`** — all the shared library dependencies raylib needs for building (X11 dev headers, GL).
- **The configure step** — uses the same sanitizer flags as Chapter 9.
- **`ctest --test-dir build --output-on-failure`** — runs every registered test.
- **The parity step** — only runs if the weights and benchmark files exist. The `diff` command fails CI if `verify`'s output doesn't match the expected output byte-for-byte. This is what makes CI actually protect against regression — a CI that lets a parity regression through defeats the point.

### Definition of done

A fresh clone builds, tests, sanitizes, and checks parity, with zero manual steps, on every push.

### Next

Chapter 11 — now that correctness is protected, the UI's most useful addition is making the network's internals visible.

---

## Chapter 11 — Network visualization

### What's already built

The probability bar chart (Chapter 7).

### What's missing

Activation maps for `conv1`–`conv4`, and the preprocessing preview (Chapter 7's Definition of Done).

### What a feature map means, mathematically

`conv1`'s output is a `32×28×28` tensor — 32 separate 28×28 grayscale images, each one the response of one learned 3×3 filter swept across the input. Early layers (`conv1`) typically respond to edges and strokes. Later layers (`conv4`) respond to more complex, less visually interpretable patterns.

Rendering `conv1`'s 32 channels as a tile grid is a fast sanity check: if every tile looks like noise, something upstream (preprocessing, weight loading order) is almost certainly wrong — often faster to notice visually than reading raw logit numbers.

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
- **`copy_tensor`** — `memcpy` because both tensors have the same shape and layout. The only requirement is that the destination is at least as large as the source, which is guaranteed by the fixed shapes.
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
- **`(x * tile_size) / t->width`** — scale source x coordinate to tile x coordinate. For a 28×28 tensor and 28-pixel tile, this is 1:1. For 14×14 and 28 pixels, it doubles.
- **`DrawPixel`** — one call per tensor cell. For 32 channels × 28 × 28 = 25,088 calls. Fast enough for 60 fps on any GPU.

### Definition of done

`main.c` can render `conv1` (at minimum) as a tile grid on demand using real intermediate tensors from an actual `model_forward_full` call, not fabricated data.

### Next

Chapter 12 — once the visualization exists, measure before optimizing.

---

## Chapter 12 — Profiling

### What to measure

Wall-clock time for:
- `canvas_to_mnist_input` (preprocessing)
- Each of the four `conv2d` calls individually
- Each `maxpool2d` call
- `linear`
- Total `model_forward`
- Total allocations (`tensor_alloc` call count per prediction — currently 6 per forward pass)

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
    fprintf(stderr, "\n=== Per-layer timing (average over %d calls) ===\n",
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
- **`(double)ts.tv_sec + (double)ts.tv_nsec * 1e-9`** — converts the `timespec` struct to a single `double` in seconds. The multiplication by `1e-9` is faster than division.
- **`static LayerTimes g_times = {0};`** — the global accumulator. Static storage duration, zero-initialized.
- **`memset(&g_times, 0, sizeof g_times)`** — zeros the whole struct in one call.
- **`1000.0 * g_times.conv1 / n`** — converts seconds to milliseconds (`× 1000`) then divides by the call count.

### The instrumented `model_forward`

```c
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out) {
#ifdef BENCHMARK
    double t_prev = now_seconds();
    double t_now;
    g_times.calls++;
#endif

    Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.conv1 += t_now - t_prev; t_prev = t_now;
#endif
    relu_tensor(&a);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.relu1 += t_now - t_prev; t_prev = t_now;
#endif

    Tensor b = conv2d(&a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.conv2 += t_now - t_prev; t_prev = t_now;
#endif
    tensor_free(&a);
    relu_tensor(&b);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.relu2 += t_now - t_prev; t_prev = t_now;
#endif

    Tensor p1 = maxpool2d(&b, 2, 2);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.pool1 += t_now - t_prev; t_prev = t_now;
#endif
    tensor_free(&b);

    Tensor c = conv2d(&p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.conv3 += t_now - t_prev; t_prev = t_now;
#endif
    tensor_free(&p1);
    relu_tensor(&c);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.relu3 += t_now - t_prev; t_prev = t_now;
#endif

    Tensor d = conv2d(&c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.conv4 += t_now - t_prev; t_prev = t_now;
#endif
    tensor_free(&c);
    relu_tensor(&d);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.relu4 += t_now - t_prev; t_prev = t_now;
#endif

    Tensor p2 = maxpool2d(&d, 2, 2);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.pool2 += t_now - t_prev; t_prev = t_now;
#endif
    tensor_free(&d);

    int in_features = p2.channels * p2.height * p2.width;
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);
#ifdef BENCHMARK
    t_now = now_seconds(); g_times.linear += t_now - t_prev;
#endif

    tensor_free(&p2);
}
```

**Crucial property:** When `BENCHMARK` is not defined, none of this code exists. The compiled `model_forward` is byte-identical to the release version. The benchmark measures the real code.

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
    fprintf(stderr, "total wall time : %8.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "average         : %8.3f ms\n",
            (t1 - t0) * 1000.0 / n_iters);
    fprintf(stderr, "throughput      : %8.1f inferences/sec\n",
            n_iters / (t1 - t0));

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

### Expected output

```
=== Benchmark: 1000 iterations ===
total wall time : 5421.3 ms
average         :    5.421 ms
throughput      :    184.5 inferences/sec

=== Per-layer timing (average over 1000 calls) ===
conv1         0.038 ms
relu1         0.010 ms
conv2         3.212 ms
relu2         0.009 ms
pool1         0.041 ms
conv3         1.019 ms
relu3         0.004 ms
conv4         1.014 ms
relu4         0.004 ms
pool2         0.023 ms
linear        0.002 ms
```

### What the numbers should tell you

Chapter 3's Big-O note already predicts `conv2`/`conv3`/`conv4` (32→32 channels, ≈7.2M ops each) should dominate over `conv1` (1→32, ≈226K ops). Confirm or refute this with real measurements before assuming it.

If preprocessing (`canvas_to_mnist_input`) is somehow a meaningful fraction of total time, that's a different, more surprising finding worth its own investigation before touching `conv2d` at all.

### Definition of done

A real table of stage-by-stage timings, measured on your actual hardware, exists and is committed somewhere reviewable — not estimated, not assumed from the Big-O note alone.

### Next

Chapter 13 — only once this table exists does it make sense to decide what (if anything) to optimize.

---

## Chapter 13 — C optimization

Order matters. Every step requires the previous one's evidence.

### 1. Buffer reuse

If profiling shows allocation overhead is non-trivial, pre-allocate scratch tensors once (outside the per-prediction hot path) instead of `tensor_alloc`ing 6 times per prediction.

**Add to `c/include/nn.h`:**

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

**Add to `c/src/nn.c`:**

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

Line by line:

- **`workspace_init`** — allocates each tensor once. Called at startup.
- **`conv2d_into`** — same math as `conv2d`, but writes into `out` instead of allocating a new tensor. The shape of `out` is not checked; the caller is responsible for allocating it at the right size.
- **`model_forward_ws`** — uses the `*_into` variants. No `tensor_alloc` or `tensor_free` in the hot path.

**CRITICAL: after switching to `model_forward_ws`, re-run Chapter 5's parity check.** The results must be identical. If they differ, the `*_into` variants have a bug.

### 2. Loop reordering

Only after (1) is measured. The current conv loop order is `oc, oy, ox, ic, ky, kx`. Since `kx` is innermost, both `weight[..., kx]` and `input[..., kx]` are contiguous — already cache-friendly. Reordering often does not help on modern out-of-order CPUs because the hardware already reorders loads.

### 3. Cache locality

Only after (2) is measured.

### 4. Compiler optimization

`-O2`/`-O3`, measured before and after. Do not assume a fixed speedup.

### 5. SIMD

Only if profiling shows a specific inner loop dominates and is vectorizable. Modern `gcc -O3` often auto-vectorizes simple loops — check the disassembly before hand-writing intrinsics.

### 6. Parallelism

Only if a single prediction's latency is still the bottleneck after 1–5, and only if the win justifies the added complexity.

### 7. Quantization

Changes numerical behavior. Requires re-running Chapter 5's parity check against the quantized output, not the float32 one.

**Every step:** hypothesis → baseline measurement (Chapter 12) → implementation → re-run Chapter 5's parity check → re-measure → keep or revert based on the number, not the feeling.

---

## Chapter 14 — ML experiments

Only after Chapters 1–13. One variable at a time.

| Experiment | Hypothesis | Metric |
|---|---|---|
| Data augmentation (rotation/shift) | Improves robustness to off-center/rotated hand-drawn digits (Chapter 6's domain-shift concern) | Test accuracy on a held-out set drawn from the actual UI, not just MNIST test set |
| Normalization (mean/std) | Current training has none; adding it may or may not help | Test accuracy, before/after |
| Kernel size / channel count | Larger model may reduce error but changes `CnnModel`'s struct sizes — cascades into Chapter 4's export/load contract | Accuracy vs. `sizeof(CnnModel)` and inference time trade-off |
| Optimizer / learning rate / batch size | Standard hyperparameter sensitivity | Convergence speed, final accuracy |

Each: hypothesis → **one** change → train → evaluate → record → keep or reject. Changing kernel size **and** learning rate in the same run makes the result uninterpretable.

---

## Chapter 15 — Mathematics through the project

| Topic | Where in Number Guesser |
|---|---|
| Vectors/matrices, dot product | `linear`'s inner loop: `sum += row[i] * x[i]` |
| Matrix multiply, `y = Wx + b` | `linear`, full function |
| Tensor shape, flattening | `Tensor` struct + `(c*H+y)*W+x`; `p2.data` passed straight to `linear` |
| Filters, channels, cross-correlation | `conv2d`'s six loops — PyTorch's `Conv2d` computes cross-correlation (no kernel flip), same as this code |
| Stride, padding, output dims | Chapter 3's worked `28→28→14→7` derivation |
| Derivatives, gradients, backprop | **Not in this codebase at all, by design** — training happens in PyTorch; `nn.c` implements only the forward pass |
| Logits, softmax, probabilities | `main.c`'s `softmax` — `exp(x-max)/Σexp` |
| Confidence | `app->confidence = app->probs[app->predicted_digit]` |
| Gradient descent, SGD/Adam | PyTorch training side — relevant to Chapter 14, not to `nn.c`/`ui.c` |

---

## Chapter 16 — D2L + MML learning map

| Topic | Why needed here | Material | Code connection | Study before |
|---|---|---|---|---|
| Linear algebra basics | `linear`, tensor indexing | MML Ch. 2 | `nn.c`'s `linear`, `tensor_get`/`set` | Chapter 3 |
| Convolutions | `conv2d` | D2L §6.1–6.3 | `nn.c`'s `conv2d` | Chapter 3 |
| Pooling | `maxpool2d` | D2L §6.5 | `nn.c`'s `maxpool2d` | Chapter 3 |
| Softmax/cross-entropy | `softmax`, training loss | D2L §3.4, §4.4 | `main.c`'s `softmax` | Chapter 7/15 |
| Optimization (SGD/Adam) | Training only | D2L Ch. 11, MML Ch. 7 | Not in `nn.c` — PyTorch side | Chapter 14, if running new experiments |

Just-in-time — read the row's material right before the chapter that needs it.

---

## Chapter 17 — AI-agent workflow

```
inspect → understand → plan → implement ONE change → compile → test →
benchmark → inspect diff → document → commit
```

Agents must not: claim parity without measurements; invent benchmarks; rewrite working code casually; optimize without profiling; change architecture casually; add dependencies without justification; delete files without checking references.

---

## Chapter 18 — Long-term phases

| Phase | Objective | Starting point | DoD |
|---|---|---|---|
| 0 — trustworthy baseline | Ch. 2 | Confirmed builds clean | `cmake --build` + manual smoke test pass |
| 1 — numerical parity | Ch. 5 | `nn.c` compiling; `benchmark/` not built | Every layer PASSES at stated tolerance vs. real PyTorch |
| 2 — preprocessing correctness | Ch. 6 | Bounding-box+bilinear centering implemented | Fixtures built, centering method's accuracy impact known |
| 3 — Raylib product | Ch. 7 | Substantially built | Preprocessing preview added |
| 4 — explainable inference | Ch. 11 | Not started | conv1 activation view works on real data |
| 5 — tests/sanitizers/CI | Ch. 8–10 | Not wired into CMake | Full CI pipeline green |
| 6 — profiling/performance | Ch. 12–13 | Not started | Real timing table exists; optimizations re-verified against Ch. 5 |
| 7 — model serialization | Ch. 4 | Header-less format, self-consistent | Versioned format built, only after Phase 1 |
| 8 — ML experiments | Ch. 14 | Not started | One-variable-at-a-time experiment log exists |

---

## Chapter 19 — Definition of done

A milestone is complete only when:

- Implementation works (compiled + run, not just written).
- Tests exist and pass.
- Parity passes where relevant (Chapter 5).
- Sanitizer checks pass (Chapter 9).
- Documentation matches reality.
- Build is reproducible (`cmake --build` from a clean checkout, no manual steps).
- Benchmark is recorded (Chapter 12's real numbers, not estimates).
- No known regression remains.

---

## Final — The next 10 tasks

Each task has a clear objective, exact commands, and a concrete definition of done. Prioritized by actual current bottleneck, not arbitrary new features.

### Task 1 — Send `python/model.py` and `python/export.py`

**Objective:** Confirm `LAYER_KEYS` order against real `model_load` read order.
**Why:** Chapter 4's `LAYER_KEYS` is currently inferred, not confirmed — the single highest-risk unverified claim in this book.
**Files:** `python/model.py`, `python/export.py`.
**Math:** None new.
**Steps:** Paste or upload them. I compare `LAYER_KEYS` order against the read order in `nn.c` line by line.
**Test:** A specific named matching or mismatch.
**Expected result:** Either confirmed matching, or a specific mismatch to fix.
**DoD:** Chapter 4's `[UNCONFIRMED]` tag removed.
**Next:** Unblocks Task 2 for real.

### Task 2 — Run Chapter 5's parity check on real weights

**Objective:** Get a real per-layer comparison.
**Why:** This is the single most important unverified claim in the project — no C code has been checked against real trained weights.
**Files:** `benchmark/verify.c` (Chapter 5 design), a saved MNIST test image.
**Math:** Chapter 5's tolerance rule.
**Steps:**

```bash
# 1. Save an MNIST test image as raw float32.
python - <<'PY'
import numpy as np
from torchvision import datasets
mnist = datasets.MNIST("data", train=False, download=True)
x, y = mnist[0]
arr = np.asarray(x, dtype=np.float32) / 255.0
arr.tofile("benchmark/debug_input.bin")
print(f"saved 3136 bytes, label={y}")
PY

# 2. Run PyTorch dump.
python python/dump_intermediate.py > notes/pytorch_stages.txt

# 3. Build and run C dump.
cd c/tools
gcc -O2 -Wall -Wextra -std=c11 -I../include verify.c ../src/nn.c -o verify -lm
./verify ../models/weights.bin ../models/debug_input.bin > ../../notes/c_stages.txt
cd ../..

# 4. Compare.
python c/tools/compare_stages.py
```

**Test:** Layer-by-layer PASS/FAIL.
**Expected result:** Not yet known — that's the point.
**DoD:** Every layer PASSES, or a named first-divergent layer identified and fixed.
**Next:** Unblocks trusting anything else in the app.

### Task 3 — Wire `tests/` into CMake

**Objective:** Automated tests run from a clean build.
**Why:** Currently zero automated tests run — flagged as a real, confirmed gap.
**Files:** `CMakeLists.txt`, `tests/test_nn.c` (Chapter 8 design).
**Steps:**

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

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

**Test:** Itself.
**Expected result:** All tests pass.
**DoD:** `ctest` runs and passes from a clean `cmake --build`.
**Next:** Makes Tasks 4 and 5 possible.

### Task 4 — Run sanitizer build against the tests

**Objective:** Confirm memory safety beyond "it compiled."
**Files:** None new.
**Command:**

```bash
cmake -S . -B build-asan \
    -DCMAKE_C_FLAGS="-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

**Test:** Itself.
**Expected result:** Zero ASan/UBSan errors. Reasonably likely given the careful `size_t` casting in `nn.c`, but not yet actually run under a sanitizer.
**DoD:** Zero errors, confirmed by a real run.
**Next:** Task 5.

### Task 5 — Add CI

**Objective:** Every push builds, tests, sanitizes.
**Files:** `.github/workflows/ci.yml` (Chapter 10 content).
**Command:** Push and check Actions.
**Test:** The workflow itself.
**Expected result:** Green on a clean push.
**DoD:** A deliberate regression (revert one fix from Task 2) makes CI fail — confirms the pipeline actually catches something.

### Task 6 — Add the preprocessing preview panel

**Objective:** Visually confirm Chapter 6's centering/cropping.
**Files:** `main.c` (Chapter 7's `draw_mnist_preview`).
**Steps:** Render the 28×28 float array from `run_prediction` as a small tile next to the canvas.
**Command:** `cmake --build build && ./build/number_guesser`.
**Test:** Manual — draw an off-center digit, confirm the preview shows it centered.
**Expected result:** Visually centered 20×20-in-28×28 digit.
**DoD:** Preview panel renders real data, not a placeholder.

### Task 7 — Build Chapter 6's deterministic fixtures

**Objective:** Test the preprocessing pipeline against controlled inputs.
**Files:** `c/tools/preprocessing_fixtures.c` (Chapter 6 design), possibly `tests/test_preprocessing.c`.
**Math:** None new — the bounding-box/bilinear formulas are in `ui.c`.
**Test:** Each fixture's 28×28 output inspected against hand-predicted expectations.
**DoD:** All fixtures pass, or reveal a specific real bug to fix.

### Task 8 — Check bounding-box-center vs. center-of-mass

**Objective:** Test Chapter 6's flagged hypothesis about asymmetric digit centering.
**Files:** `ui.c`.
**Steps:** After Task 2 gives a working parity/accuracy baseline, swap the centering formula to a pixel-weighted centroid and re-measure accuracy specifically on asymmetric digits (7, 2, 9).
**DoD:** Either confirmed negligible (keep current code) or confirmed to help (replace it) — a measured decision.

### Task 9 — Add `conv1` activation visualization

**Objective:** Visually catch preprocessing/weight bugs on real drawings.
**Files:** `main.c`, `nn.c`/`nn.h` (Chapter 11's `Activations` + `model_forward_full`).
**DoD:** Toggleable debug view renders real `conv1` output as a tile grid.

### Task 10 — Profile before optimizing

**Objective:** Get real numbers before touching Chapter 13's optimization ideas.
**Files:** `c/tools/benchmark.c` (Chapter 12 design).
**DoD:** A real, committed timing table exists on your actual hardware.

---

*End of the Number Guesser Engineering Book.*

Every chapter above is grounded in the actual state of your project. Chapters 2–5 and 8–12 are described in the state they exist today (compiling, tested on synthetic weights, not yet verified against real PyTorch output). Chapters 6–7 and 11 are additional work described at the design level. Task 1 (send `python/model.py` and `export.py`) unblocks the rest.

Work through the 10 tasks in order. Each one has a verifiable result. When all ten are complete, the project will be:

- **Provably correct** — C and PyTorch agree layer-by-layer on real weights.
- **Tested** — every operation has a unit test; every end-to-end path has an integration test.
- **Sanitized** — no memory bugs, no undefined behavior.
- **Benchmarked** — a real timing table exists.
- **Versioned** — the model format is self-describing.
- **Documented** — every claim is backed by code or a measurement.

That is the difference between a working prototype and a genuine engineering artifact.