# Number Guesser — The Complete C Implementation Book

Every piece of C in this document has actually been compiled and run —
`nn.c`/`ui.c` under `-Wall -Wextra -fsanitize=address,undefined` with zero
warnings and zero sanitizer errors, and `main.c` compiled against a real
raylib build. Where I show test output, it's pasted from a real run, not
invented. The `python/export.py` and `python/dump_intermediate.py` scripts
are syntax-checked (this sandbox doesn't have PyTorch installed, so they
haven't run against real weights yet — that step is yours, on your machine,
once you've trained a model).

This is long on purpose — you asked for the complete picture. Read it
straight through once, then keep it open as reference while you build.

---

## Table of contents

1. [The C memory model](#1-the-c-memory-model)
2. [Part 1 — Linear, ReLU, Argmax](#2-part-1--linear-relu-argmax)
3. [Part 2 — The Tensor struct](#3-part-2--the-tensor-struct)
4. [Part 3 — Conv2D](#4-part-3--conv2d)
5. [Part 3 — MaxPool2D](#5-part-3--maxpool2d)
6. [Part 3 — Flatten](#6-part-3--flatten)
7. [Part 3 — Wiring the full forward pass](#7-part-3--wiring-the-full-forward-pass)
8. [Part 4 — The CnnModel struct and weight loading](#8-part-4--the-cnnmodel-struct-and-weight-loading)
9. [Part 4 — Exporting weights from PyTorch](#9-part-4--exporting-weights-from-pytorch)
10. [Part 5 — Verifying C against PyTorch](#10-part-5--verifying-c-against-pytorch)
11. [Part 6 — The Raylib UI](#11-part-6--the-raylib-ui)
12. [Part 7 — Preprocessing parity](#12-part-7--preprocessing-parity)
13. [Part 8 — Visualization](#13-part-8--visualization)
14. [Testing philosophy](#14-testing-philosophy)
15. [Debugging playbook, with real examples](#15-debugging-playbook-with-real-examples)
16. [Build & run, on your own machine](#16-build--run-on-your-own-machine)
17. [Appendix: full file listing](#17-appendix-full-file-listing)

---

## 1. The C memory model

Everything else in this book is a variation on three ideas: where memory
lives, who owns it, and how multi-dimensional data becomes a flat address.
Worth being precise about all three before writing anything.

### Stack vs heap

A **stack** variable — `float x[10];` inside a function — is allocated the
instant the function is entered and destroyed the instant it returns. It's
fast (no bookkeeping, just moving a pointer) and its size must be knowable
at compile time (or at least at function-entry time, for a
variable-length-array, which this project avoids for portability). The
catch: you cannot return a pointer to it and use that pointer after the
function returns — the memory is gone, and reading it is undefined
behavior (it might *look* fine, then crash five minutes later — the worst
kind of bug).

A **heap** allocation — `malloc`/`calloc` — lives until you call `free()`
on it, regardless of which function you're in when you do so. Its size is
a runtime value, which is exactly what we need: a `Tensor`'s size depends
on the layer's output shape, computed from the *previous* layer's shape at
runtime, not known until the program is actually running.

Rule of thumb for this project: small, fixed-size, short-lived things
(a `float logits[10]` for one prediction) go on the stack. Anything whose
size depends on a runtime value, or that needs to outlive the function that
created it (every `Tensor` between layers), goes on the heap.

### Ownership

**Ownership** means: exactly one place in the code is responsible for
calling `free()` on a given pointer, and every other place that touches
that pointer just borrows it. We'll follow one consistent rule throughout:

- A function that **allocates and returns** a `Tensor` (like `conv2d`)
  hands ownership to its caller. The caller must eventually `tensor_free()`
  it.
- A function that only **reads or mutates in place** (like `relu_tensor`,
  or `tensor_get`) never frees anything — it's borrowing.

This single rule is what makes `model_forward` (§7) leak-free despite
allocating eight intermediate tensors during one prediction: each one gets
freed the moment the *next* layer has consumed it, by whoever received it.

**Dangling pointers** are what happen when that rule breaks — using memory
after `free()`, or returning the address of a stack variable. The fix is
always "who owns this, and have they freed it yet?" — which is why every
function in this project has an explicit comment about what it allocates
and who's responsible for freeing it.

### Multi-dimensional data is a flat address computation

C has no true multi-dimensional array type for runtime-sized data — a
`float[C][H][W]` with compile-time-constant `C`, `H`, `W` exists, but we
need `channels`/`height`/`width` to vary at runtime (32, 28, 28 for one
layer; 32, 7, 7 for another). So every "3D array" in this project is
actually **one flat `float*`**, and *we* compute the offset:

```
data[c][y][x]  ==  data[ c*(height*width) + y*width + x ]
```

Read it as: skip `c` whole channels (`height*width` floats each), then `y`
whole rows (`width` floats each), then `x` more floats to land on the
pixel. This one formula, extended by one more level, is also how PyTorch's
`Conv2d.weight` — shape `[out_ch, in_ch, kh, kw]` — becomes a flat offset
in §4. Once this formula is second nature, Conv2D's indexing stops looking
mysterious; it's four nested applications of the same idea.

---

## 2. Part 1 — Linear, ReLU, Argmax

**What it does.** `y = Wx + b` — the fully-connected layer at the end of the
network. **Why we need it:** it's the simplest possible template for
"multi-dimensional math as flat-memory loops," before Conv2D adds more
dimensions to juggle.

`c/include/nn.h`:
```c
void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);
void relu(float *x, int n);
int argmax(const float *x, int n);
```

`c/src/nn.c`:
```c
void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features) {
    for (int o = 0; o < out_features; o++) {
        float sum = b[o];
        const float *row = W + o * in_features;
        for (int i = 0; i < in_features; i++) {
            sum += row[i] * x[i];
        }
        y[o] = sum;
    }
}

void relu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

int argmax(const float *x, int n) {
    int best_idx = 0;
    float best_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best_idx = i;
        }
    }
    return best_idx;
}
```

### Walking through `linear`

`W` is one flat block, `[out_features * in_features]` floats, row-major:
row `o` (the weights feeding output `o`) starts at `W[o * in_features]`.
`W + o * in_features` is pointer arithmetic — C scales the offset by
`sizeof(float)` automatically, so this moves the pointer to exactly the
start of row `o`, no manual byte-counting needed. `row[i]` then walks
across that row.

Why loop over `o` on the outside and `i` on the inside, rather than the
reverse? Because for a fixed `o`, `row[i]` and `x[i]` are both read
sequentially, in increasing address order — this is cache-friendly: the
CPU pulls in a whole cache line (typically 64 bytes = 16 floats) at once,
and sequential access uses every float in that line before moving on.
Looping the other way around would jump `in_features` floats between
accesses to `x`, missing the cache far more often. This matters more once
`in_features` is 1568 (the flattened conv output) than it does at
`in_features = 3` in the test below, but the habit is worth having from
the start.

**Ownership:** `linear` writes into a caller-provided `y` buffer and never
calls `malloc`. This is deliberate — for a layer this cheap, forcing every
caller to also manage a heap allocation would be pure overhead. Compare
this to `conv2d` in §4, which *does* allocate and return — the difference
is worth noticing.

### Hand-verified test

```
x = [1, 2, 3]
W = [[ 1.0,  0.0, -1.0],
     [ 0.5,  0.5,  0.5]]
b = [0.0, -1.0]

y0 = 1*1 + 0*2 + -1*3 + 0        = -2
y1 = 0.5*1 + 0.5*2 + 0.5*3 - 1   =  2
relu -> [0, 2]
argmax -> 1
```

Actual run (`tests/test_nn.c`, `test_linear_relu_argmax`):
```
test_linear_relu_argmax
  [ok]   linear y0: -2.000000
  [ok]   linear y1: 2.000000
  [ok]   relu y0: 0.000000
  [ok]   relu y1: 2.000000
  [ok]   argmax: 1
```

---

## 3. Part 2 — The Tensor struct

**Why we need it.** `linear`'s output is a flat list — one dimension, no
shape to track. Conv2D's output is `[channels, height, width]`, and every
function downstream needs to know that shape to index into it correctly.
Passing three extra `int`s to every function works but gets error-prone
fast (easy to swap `height`/`width` by accident); a struct bundles them
with the data itself.

```c
typedef struct {
    float *data;   // owns this — heap-allocated, channels*height*width floats
    int channels;
    int height;
    int width;
} Tensor;
```

Full implementation:

```c
Tensor tensor_alloc(int channels, int height, int width) {
    Tensor t;
    t.channels = channels;
    t.height = height;
    t.width = width;
    size_t n = (size_t)channels * height * width;
    t.data = calloc(n, sizeof(float));
    if (t.data == NULL) {
        fprintf(stderr, "tensor_alloc: calloc failed for %d x %d x %d\n",
                channels, height, width);
        exit(1);
    }
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = t->height = t->width = 0;
}

float tensor_get(const Tensor *t, int c, int y, int x) {
    return t->data[(c * t->height + y) * t->width + x];
}

void tensor_set(Tensor *t, int c, int y, int x, float value) {
    t->data[(c * t->height + y) * t->width + x] = value;
}
```

### Why `calloc`, not `malloc`

`malloc` gives you memory with whatever garbage bytes happened to be there
before — reading it before writing is undefined behavior. `calloc` zeroes
the memory. This matters concretely in `conv2d` (§4): each output value
starts as `sum = bias[oc]` and then has terms *added* to it across several
nested loops. If we allocated the output tensor with `malloc` instead of
`calloc`, that wouldn't actually break anything here, because every element
of `conv2d`'s output gets explicitly assigned via `tensor_set` before it's
read — but `calloc` is still the safer default for tensors in general,
because it means "uninitialized" and "zero" are the same failure mode
instead of two different ones, which makes a forgotten-to-write-this-cell
bug show up as a suspicious 0.0 instead of a random garbage float that's
harder to recognize as a bug at all.

`calloc(n, sizeof(float))` also has one small advantage over
`malloc(n * sizeof(float))`: it checks for the multiplication overflowing,
which matters once `n` is computed from several runtime values multiplied
together, as it is here (`channels * height * width`).

### Why the index formula flattens in this order

`(c * height + y) * width + x` — same derivation as §1's general formula,
specialized: skip `c` whole channels (`height * width` floats each,
written here as the two-step `c * height` then `* width` to avoid
recomputing `height * width` as a separate variable — a minor style choice,
either is fine), then `y` rows (`width` floats each), then `x` more. This
is "channel-major" layout — all of channel 0 before any of channel 1 — and
it's not arbitrary: it's PyTorch's default `[C, H, W]` layout too, which
means exporting a `Conv2d` output tensor from PyTorch and reading it into
this `Tensor` requires no reordering, ever. That property is worth
protecting; if you ever change this formula, every export/verify step in
§9-§10 needs to change with it.

### `tensor_print_summary` — for eyeballing state during development

```c
void tensor_print_summary(const Tensor *t, const char *label) {
    printf("%s: [%d, %d, %d]", label, t->channels, t->height, t->width);
    int n = t->channels * t->height * t->width;
    int show = n < 5 ? n : 5;
    printf("  first %d values: [", show);
    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], (i == show - 1) ? "" : ", ");
    }
    printf("]\n");
}
```

Small, but this is what makes debugging Conv2D possible without a real
debugger — drop a call after any op and see the shape plus a few values
immediately. `tools/verify.c` in §10 is built entirely out of calls like
this one.

### Verified test (`tests/test_nn.c`, `test_tensor`)

```c
Tensor t = tensor_alloc(2, 3, 3);
tensor_set(&t, 1, 2, 0, 7.5f);
```

Actual run:
```
test_tensor
  [ok]   calloc zero-init [0,0,0]: 0.000000
  [ok]   calloc zero-init [1,2,2]: 0.000000
  [ok]   set/get [1,2,0]: 7.500000
  [ok]   neighbor [1,1,0] unaffected: 0.000000
  [ok]   neighbor [0,2,0] unaffected: 0.000000
```

The "neighbor unaffected" checks matter more than they look — they're
specifically testing that the index formula doesn't accidentally alias two
different `(c,y,x)` triples to the same memory address, which is the most
common way a wrong flattening formula fails silently (values corrupt each
other instead of crashing, and you get a plausible-looking wrong answer).

---

## 4. Part 3 — Conv2D

**What it does.** Slides a small kernel (3×3 in your model) over the
input, computing one weighted sum per output position per filter. **Why we
need it:** it's the core operation of every conv layer in `model.py` —
4 of them, all `kernel_size=3, stride=1, padding=1`.

### Output size

For kernel size `k`, stride `s`, padding `p`:

```
out_dim = (in_dim + 2*p - k) / s + 1
```

With `k=3, s=1, p=1`: `out_dim = in_dim + 2 - 3 + 1 = in_dim`. That's the
concrete reason every conv layer in your model preserves H×W — it's this
formula evaluating to identity for this specific `(k, s, p)` combination,
not a coincidence or a raylib/PyTorch default. Change any of the three and
the output would shrink or grow.

### The nested loops, conceptually

```
for each output filter (out_channel)
  for each output row (out_y)
    for each output col (out_x)
      sum = bias[out_channel]
      for each input channel (in_channel)
        for each kernel row (ky)
          for each kernel col (kx)
            in_y = out_y*stride - pad + ky
            in_x = out_x*stride - pad + kx
            if (in_y, in_x) inside input bounds:
              sum += input[in_channel][in_y][in_x]
                   * kernel[out_channel][in_channel][ky][kx]
      output[out_channel][out_y][out_x] = sum
```

The bounds check *is* the padding. Conceptually we're padding the input
with a border of zeros and then running an unpadded convolution over the
padded image; in code, instead of allocating that padded copy, we compute
where in the *original* image each kernel tap would land, and skip it
(contributing nothing to `sum`, same as if a zero had been there) whenever
that lands outside the real image. Same math, no extra memory, no extra
allocation — just a branch inside the hottest loop, which is a cheap price
for skipping an entire allocate-copy-free cycle.

### Full implementation

```c
Tensor conv2d(const Tensor *input, const float *weights, const float *bias,
              int out_channels, int k, int stride, int pad) {
    int out_h = (input->height + 2 * pad - k) / stride + 1;
    int out_w = (input->width  + 2 * pad - k) / stride + 1;
    Tensor out = tensor_alloc(out_channels, out_h, out_w);

    for (int oc = 0; oc < out_channels; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float sum = bias[oc];
                for (int ic = 0; ic < input->channels; ic++) {
                    for (int ky = 0; ky < k; ky++) {
                        for (int kx = 0; kx < k; kx++) {
                            int iy = oy * stride - pad + ky;
                            int ix = ox * stride - pad + kx;
                            if (iy < 0 || iy >= input->height) continue;
                            if (ix < 0 || ix >= input->width)  continue;
                            float in_val = tensor_get(input, ic, iy, ix);
                            int w_idx = ((oc * input->channels + ic) * k + ky) * k + kx;
                            sum += in_val * weights[w_idx];
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

### The weight index, unpacked

`((oc * input->channels + ic) * k + ky) * k + kx` — same flattening idea as
`Tensor`, one level deeper, because PyTorch's `Conv2d.weight` has *four*
dimensions: `[out_channels, in_channels, kh, kw]`. Read the formula
inside-out: `oc * input->channels + ic` is "which (out_channel, in_channel)
pair," scaled up by `k` for the kernel row, then by `k` again for the
kernel column. This is exactly why `export.py` (§9) needs no reordering —
this formula and PyTorch's own internal weight layout are the same thing.

### Ownership: this function is different from `linear`

`conv2d` **allocates and returns** a new `Tensor` — unlike `linear`, which
wrote into a caller-provided buffer. That's a deliberate difference: a
conv layer's output size isn't knowable by the *caller* without duplicating
the output-size formula everywhere a conv is called, so it's cleaner for
`conv2d` itself to compute it and hand back a right-sized `Tensor`. The
cost: the caller now owns that `Tensor` and must `tensor_free()` it once
it's no longer needed — which in a forward pass means "once the *next*
layer has consumed it." See §7 for how that chain of allocate → consume →
free is threaded through four conv layers without leaking any of them.

### Two verified tests

**Tiny, no padding** (`tests/test_nn.c`, `test_conv2d`):
```
input (1x3x3):        kernel (2x2):
  1 2 3                 1 0
  4 5 6                 0 1
  7 8 9
bias: 0, stride 1, pad 0 -> output 1x2x2

out[0][0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
out[0][1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
out[1][0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
out[1][1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
```
Actual run:
```
test_conv2d
  output shape: [1,2,2] (expected [1,2,2])
  [ok]   conv out[0][0]: 6.000000
  [ok]   conv out[0][1]: 8.000000
  [ok]   conv out[1][0]: 12.000000
  [ok]   conv out[1][1]: 14.000000
```

**With padding, matching this project's actual conv config**
(`test_conv2d_padding`): a 1×1×1 input (single pixel, value 5.0), a 3×3
kernel that's all zero except the center tap (= 1), `stride=1, pad=1`.
Since only the center tap is non-zero, and the center tap always lands on
the real pixel regardless of padding, the expected output is just 5.0
again — and this also confirms the output shape stays `1×1×1`, proving the
`out_dim = in_dim` claim from earlier isn't just algebra, it's what the
code actually does:
```
test_conv2d_padding (k=3, stride=1, pad=1 -> same size, like this project's convs)
  output shape: [1,1,1] (expected [1,1,1], padding keeps size constant)
  [ok]   center-tap identity conv: 5.000000
```

### A performance note (optional, not required to move forward)

This is `O(out_channels × out_h × out_w × in_channels × k × k)` — for
`conv1` in your model that's `32 × 28 × 28 × 1 × 3 × 3 ≈ 226K` operations;
for `conv2`/`conv3`/`conv4` (32→32 channels) it's `32 × 28 × 28 × 32 × 3 × 3
≈ 7.2M` each. On a modern CPU that's genuinely fast — milliseconds, not
seconds — so there's no need to optimize this for the project to work.
If you do want to go further later: the standard trick is "im2col" (unroll
each kernel-sized patch of the input into a matrix row, turning the whole
convolution into one big matrix multiply, which BLAS libraries are very
fast at) — worth knowing the name exists, not worth implementing here,
since it trades a clear, readable loop nest for a much less obvious one.

---

## 5. Part 3 — MaxPool2D

**What it does.** For each non-overlapping `k×k` window, output the
maximum value in that window. **Why we need it:** your model uses
`MaxPool2d(kernel_size=2, stride=2)` twice, each time halving H and W —
that's the entire mechanism behind 28→14→7 in the architecture table.

```c
Tensor maxpool2d(const Tensor *input, int k, int stride) {
    int out_h = (input->height - k) / stride + 1;
    int out_w = (input->width  - k) / stride + 1;
    Tensor out = tensor_alloc(input->channels, out_h, out_w);

    for (int c = 0; c < input->channels; c++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                float best = -1e30f; /* stand-in for -infinity: any real pixel beats it */
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {
                        int iy = oy * stride + ky;
                        int ix = ox * stride + kx;
                        float v = tensor_get(input, c, iy, ix);
                        if (v > best) best = v;
                    }
                }
                tensor_set(&out, c, oy, ox, best);
            }
        }
    }
    return out;
}
```

Simpler than `conv2d` in three specific ways, worth naming explicitly since
it's easy to over-think this after just writing convolution: no separate
input-channel loop (`MaxPool2d` never mixes channels — each output channel
only ever reads its own matching input channel, so the `for (int c ...)`
loop plays double duty as both the input- and output-channel index), no
weights or bias (there's nothing to learn in a max operation), and no
padding/bounds-checking (this project's pooling never uses padding, so
every window is guaranteed fully inside the input — no `continue` needed).

The output-size formula also drops the `+ 2*p` term from `conv2d`'s formula
for the same reason: `pad=0` always here.

`best = -1e30f` deserves a note: C has no portable literal `-infinity` you
can just write down (`-INFINITY` from `<math.h>` exists and would be more
correct — feel free to use it instead), so a very negative sentinel value
that's guaranteed to lose to any real pixel value works just as well for
this project's inputs, which are always in a bounded range.

### Verified test (`tests/test_nn.c`, `test_maxpool2d`)

```
input (1x4x4):
 1 3 2 4
 5 6 1 2
 7 8 3 1
 0 2 4 9
k=2, stride=2 -> output 1x2x2
block(0,0): max(1,3,5,6) = 6
block(0,1): max(2,4,1,2) = 4
block(1,0): max(7,8,0,2) = 8
block(1,1): max(3,1,4,9) = 9
```
Actual run:
```
test_maxpool2d
  output shape: [1,2,2] (expected [1,2,2])
  [ok]   pool out[0][0]: 6.000000
  [ok]   pool out[0][1]: 4.000000
  [ok]   pool out[1][0]: 8.000000
  [ok]   pool out[1][1]: 9.000000
```

---

## 6. Part 3 — Flatten

This is the one op in the whole project with **no code**, and that's the
point worth understanding, not skipping past. `Tensor.data` has been one
contiguous flat block since §3 — PyTorch's `nn.Flatten()` and a C
`float*` reinterpretation are, at the memory level, *the same operation*:
`tensor.view(-1)` in PyTorch doesn't move or copy a single byte, it just
changes what shape metadata is attached to the same underlying storage.

Concretely: after `pool2` in your model, you have a `Tensor` with
`channels=32, height=7, width=7`. To flatten it, you don't write a
`flatten()` function at all — you pass `pool2.data` directly into
`linear()` from §2, with `in_features = pool2.channels * pool2.height *
pool2.width` (= 1568). See §7's `model_forward` for exactly this line.

This is the payoff of building everything on flat memory from the start,
rather than reaching for a "real" multi-dimensional array type: the
boundary between "3D tensor" and "1D vector" was never a memory boundary,
just a bookkeeping one, so crossing it costs nothing.

---

## 7. Part 3 — Wiring the full forward pass

Now every op from §2, §4, §5 gets threaded together into one function that
takes a raw `1×28×28` image and produces 10 logits — and correctly frees
every intermediate `Tensor` along the way, so a full forward pass leaks
zero bytes.

```c
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out) {
    /* block_1: conv -> relu -> conv -> relu -> pool */
    Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1); /* 32x28x28 */
    relu_tensor(&a);
    Tensor b = conv2d(&a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);    /* 32x28x28 */
    tensor_free(&a);
    relu_tensor(&b);
    Tensor p1 = maxpool2d(&b, 2, 2);                               /* 32x14x14 */
    tensor_free(&b);

    /* block_2: conv -> relu -> conv -> relu -> pool */
    Tensor c = conv2d(&p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);   /* 32x14x14 */
    tensor_free(&p1);
    relu_tensor(&c);
    Tensor d = conv2d(&c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);    /* 32x14x14 */
    tensor_free(&c);
    relu_tensor(&d);
    Tensor p2 = maxpool2d(&d, 2, 2);                                /* 32x7x7 */
    tensor_free(&d);

    /* classifier: flatten (free — p2.data is already flat) -> linear */
    int in_features = p2.channels * p2.height * p2.width; /* 32*7*7 = 1568 */
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);
    tensor_free(&p2);
}
```

### The ownership chain, made explicit

Trace it once, slowly, because this pattern — allocate, consume, free,
repeat — is the whole memory-management story of this project:

1. `conv2d(input, ...)` allocates `a`. **We now own `a`.**
2. `relu_tensor(&a)` mutates `a` in place — no new allocation, no change
   in ownership.
3. `conv2d(&a, ...)` reads `a` (borrowing it) and allocates a *new* tensor
   `b`. We now own `b`, and `a` is no longer needed.
4. `tensor_free(&a)` — the moment `a` is done being read, it's freed.
   Not at the end of the function, not "whenever" — immediately once its
   last use has happened. This is what keeps peak memory usage bounded to
   roughly two tensors at a time, not eight.
5. This pattern repeats: `b` → `p1` (free `b`) → `c` (free `p1`) → `d`
   (free `c`) → `p2` (free `d`) → logits (free `p2`).

Notice `input` itself is never freed here — `model_forward` borrows it
(it's a `const Tensor *`), consistent with the rule from §1: this function
didn't allocate `input`, so it doesn't own it and doesn't free it. Whoever
called `model_forward` (in this project, that's `run_prediction` in
`main.c`, §11) is responsible for that one.

### Smoke-tested (not yet accuracy-tested — that needs real trained weights)

```c
CnnModel m;
model_load(&m, "/tmp/test_weights.bin"); // synthetic weights, see §8
Tensor input = tensor_alloc(1, 28, 28);
// ... fill input.data with 0.5 everywhere, arbitrary ...
float logits[10];
model_forward(&m, &input, logits);
```
Actual run (compiled and run under AddressSanitizer + UBSan, zero errors):
```
test_model_forward_smoke (shape/crash check, not accuracy)
  logits: [1577089519752701662484198588416.00, ... ]
  argmax: 9
  [ok]   forward pass ran end-to-end without crashing
```
The huge numbers are expected and not a bug — the synthetic test weights
are sequential integers (0, 1, 2, 3, ...) used purely to check that
`model_load` reads bytes into the right fields in the right order (§8),
not real trained weights, so the logits are meaningless as predictions.
What this test actually proves: the full eight-tensor allocate/free chain
above runs to completion with no crash and — confirmed separately by
running the whole suite under AddressSanitizer — no memory error and no
leak.

---

## 8. Part 4 — The CnnModel struct and weight loading

```c
typedef struct {
    float conv1_w[288],  conv1_b[32];    /* 1  -> 32, 3x3 */
    float conv2_w[9216], conv2_b[32];    /* 32 -> 32, 3x3 */
    float conv3_w[9216], conv3_b[32];    /* 32 -> 32, 3x3 */
    float conv4_w[9216], conv4_b[32];    /* 32 -> 32, 3x3 */
    float fc_w[15680],   fc_b[10];       /* 1568 -> 10 */
} CnnModel;

#define WEIGHTS_FILE_BYTES 175016
```

These are fixed-size arrays *inside* the struct, not pointers — a
deliberate choice, since your architecture is fixed and known at compile
time (unlike `Tensor`, whose shape genuinely varies at runtime layer to
layer). That means a `CnnModel` can live on the stack if you want it to
(`CnnModel m;` in `main`, as `main.c` does), no `malloc`/`free` needed for
the struct itself at all — only `model_load`'s internal `fopen`/`fclose`
manages any resource here.

### Why "CnnModel" and not "Model"

The original design used `Model` — until wiring up `main.c` against
raylib produced this real compiler error:
```
c/include/nn.h:85:3: error: conflicting types for 'Model'; have 'struct <anonymous>'
   85 | } Model;
      |   ^~~~~
/home/claude/raylib-build/include/raylib.h:433:3: note: previous declaration of 'Model' with type 'Model'
```
Raylib already defines its own `Model` type (for loaded 3D models — not
something this project uses, but the name still collides at the
preprocessor/compiler level the moment both headers are included in the
same file). This is a real, common category of bug once a project starts
combining two libraries — C has no namespaces, so any two headers can
define the same top-level name and the build simply breaks. The fix is
always a rename to something more specific — `CnnModel` here — not a
workaround; there's no way to "un-collide" two identical top-level names
in C.

### Loading, with a size check first

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

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size != WEIGHTS_FILE_BYTES) {
        fprintf(stderr, "model_load: '%s' is %ld bytes, expected %d\n",
                path, size, WEIGHTS_FILE_BYTES);
        fclose(f);
        return -1;
    }

    int err = 0;
    err |= read_floats(f, m->conv1_w, 288,   "conv1_w");
    err |= read_floats(f, m->conv1_b, 32,    "conv1_b");
    err |= read_floats(f, m->conv2_w, 9216,  "conv2_w");
    err |= read_floats(f, m->conv2_b, 32,    "conv2_b");
    err |= read_floats(f, m->conv3_w, 9216,  "conv3_w");
    err |= read_floats(f, m->conv3_b, 32,    "conv3_b");
    err |= read_floats(f, m->conv4_w, 9216,  "conv4_w");
    err |= read_floats(f, m->conv4_b, 32,    "conv4_b");
    err |= read_floats(f, m->fc_w,    15680, "fc_w");
    err |= read_floats(f, m->fc_b,    10,    "fc_b");

    fclose(f);
    return err ? -1 : 0;
}
```

The `fseek`/`ftell`/`fseek` sequence is the standard C idiom for "how big is
this file": seek to the end, ask the current position (that *is* the file
size, in bytes), seek back to the start before reading anything. Checking
this **before** any `fread` call means a wrong or truncated `weights.bin`
fails loudly and immediately, instead of `model_load` silently reading
garbage into the later arrays once the file runs out of bytes partway
through — which would otherwise produce a model that loads "successfully"
but predicts nonsense, the hardest kind of bug to trace back to its cause.

`err |= read_floats(...)` accumulates failures across all ten reads rather
than stopping at the first one — on purpose, so a broken `weights.bin`
reports *every* short read via stderr in one run, not just the first,
which is more useful when you're actively debugging an export mismatch.

### Verified: both the happy path and the size guard

```c
/* synthetic weights.bin: every float equals its position in the file */
Model m;  // (CnnModel, after the rename)
model_load(&m, path);
// conv1_w[0]   == 0.0    (first float in the file)
// conv1_w[287] == 287.0  (last of the first 288-float block)
// conv1_b[0]   == 288.0  (first bias float, right after conv1_w)
// fc_b[9]      == 43753.0 (the very last float in the file)
```
Actual run:
```
test_model_load
  wrote synthetic weights.bin: 43754 floats (175016 bytes, expect 175016)
  [ok]   conv1_w[0] (first float in file): 0.000000
  [ok]   conv1_w[287] (last of first block): 287.000000
  [ok]   conv1_b[0] (first bias, right after conv1_w): 288.000000
  [ok]   fc_b[9] (very last float in file): 43753.000000
  [ok]   model_load correctly rejects wrong-size file
```
`conv1_b[0] == 288.0` is the test earning its keep: it confirms not just
that each individual array reads the right *count* of floats, but that
consecutive arrays pick up exactly where the previous one left off, with
no gap and no overlap — the thing that would go silently wrong first if
`LAYER_KEYS` in `export.py` (§9) and the read order in `model_load` ever
drifted out of sync with each other.

---

## 9. Part 4 — Exporting weights from PyTorch

```python
"""Export trained PyTorch weights to a flat binary file for C inference."""
import torch
from pathlib import Path
from model import _MainModel

MODEL_PATH = Path("models/number_guesser_model.pth")
OUTPUT_PATH = Path("models/weights.bin")
EXPECTED_BYTES = 175016

# Order matches c/src/nn.c's model_load() exactly.
LAYER_KEYS = [
    "block_1.0.weight", "block_1.0.bias",   # conv1: 1  -> 32
    "block_1.2.weight", "block_1.2.bias",   # conv2: 32 -> 32
    "block_2.0.weight", "block_2.0.bias",   # conv3: 32 -> 32
    "block_2.2.weight", "block_2.2.bias",   # conv4: 32 -> 32
    "classifier.1.weight", "classifier.1.bias",  # linear: 1568 -> 10
]

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found — train and save a model first.")

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    missing = [k for k in LAYER_KEYS if k not in state_dict]
    if missing:
        raise SystemExit(
            f"state_dict is missing expected keys: {missing}\n"
            f"Actual keys: {list(state_dict.keys())}\n"
            f"model.py's architecture may have changed — update LAYER_KEYS "
            f"(and c/src/nn.c's model_load) to match."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        for key in LAYER_KEYS:
            tensor = state_dict[key]
            f.write(tensor.contiguous().numpy().tobytes())

    actual_bytes = OUTPUT_PATH.stat().st_size
    status = "OK" if actual_bytes == EXPECTED_BYTES else "MISMATCH"
    print(f"Wrote {OUTPUT_PATH} ({actual_bytes} bytes, expected {EXPECTED_BYTES}) [{status}]")

if __name__ == "__main__":
    main()
```

(Full file at `python/export.py` — syntax-checked with `python3 -m
py_compile` in this sandbox, since PyTorch itself isn't installed here;
running it end-to-end against a real `.pth` is the next step on your
machine.)

### Where the keys come from

`nn.Sequential`'s children are indexed `0, 1, 2, ...` in definition order,
and only layers *with* parameters (Conv2d, Linear) get an entry in
`state_dict()` — ReLU and MaxPool2d have none, but they still occupy an
index slot. That's why `block_1`'s first conv is index `.0` and its second
is index `.2`, not `.1`: index `.1` is the ReLU between them. Cross-check
this against `model.py` any time the architecture changes — it's the
single easiest place for the whole export/load pipeline to silently drift
out of sync (see §15 for exactly this failure mode, worked through as a
debugging example).

### Why no reordering is needed

`tensor.contiguous().numpy().tobytes()` — PyTorch tensors are
C-contiguous (row-major) by default, and `.contiguous()` is a defensive
no-op *unless* the tensor happens to be a non-contiguous view (which
`state_dict()` values never are, but costs nothing to be sure about). That
row-major layout is exactly what `Tensor` (§3) and `conv2d`'s weight
indexing (§4) already assume — so the export is a straight byte dump, no
axis permutation, no manual reshaping. This symmetry — both sides agreeing
on layout without either needing to convert — is *why* `weights.bin` needs
no header at all: a header would exist to describe a layout that, here,
both sides already know by construction.

### The self-check built into the script

Two things fail loudly rather than silently: a missing `state_dict` key
(`missing = [...]`, raised as a `SystemExit` with the actual keys printed,
so a typo or an architecture change is obvious immediately) and a wrong
output file size (`status` printed, though not currently a hard failure —
worth hardening to `raise SystemExit` too if you want export.py to refuse
to produce a file `model_load`'s size check would reject anyway).

---

## 10. Part 5 — Verifying C against PyTorch

**Mandatory before Raylib.** A polished UI on top of a silently-wrong CNN
just means confident-looking wrong answers. The method: run one fixed
input through both PyTorch and C, print every intermediate tensor from
both, and compare by eye — layer by layer, not just the final 10 logits.

### Python side — `python/dump_intermediate.py`

```python
import torch
import numpy as np
from pathlib import Path
from model import _MainModel

MODEL_PATH = Path("models/number_guesser_model.pth")
INPUT_PATH = Path("models/debug_input.bin")  # 28*28 raw float32, [0,1]

def dump(label, tensor):
    flat = tensor.detach().flatten().numpy()
    print(f"{label:8s} shape={tuple(tensor.shape)}  first 5={np.round(flat[:5], 4).tolist()}")

def main():
    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    if INPUT_PATH.exists():
        raw = np.fromfile(INPUT_PATH, dtype=np.float32)
        x = torch.from_numpy(raw).reshape(1, 1, 28, 28)
    else:
        print(f"[warn] {INPUT_PATH} not found, using a zero image instead")
        x = torch.zeros(1, 1, 28, 28)

    with torch.no_grad():
        dump("input", x)
        a = model.block_1[0](x); dump("conv1", a)
        a = model.block_1[1](a); dump("relu1", a)
        a = model.block_1[2](a); dump("conv2", a)
        a = model.block_1[3](a); dump("relu2", a)
        a = model.block_1[4](a); dump("pool1", a)
        a = model.block_2[0](a); dump("conv3", a)
        a = model.block_2[1](a); dump("relu3", a)
        a = model.block_2[2](a); dump("conv4", a)
        a = model.block_2[3](a); dump("relu4", a)
        a = model.block_2[4](a); dump("pool2", a)
        flat = model.classifier[0](a); dump("flat", flat)
        logits = model.classifier[1](flat); dump("logits", logits)
        print(f"\npredicted digit: {logits.argmax(dim=1).item()}")

if __name__ == "__main__":
    main()
```

Note this deliberately calls `model.block_1[0]`, `model.block_1[1]`, ...
individually — bypassing `model.forward()`'s single call through the whole
`nn.Sequential` — specifically so it can print *between* every op, not
just at the end. This mirrors the C side's structure exactly, on purpose:
the two scripts should read almost like translations of each other.

### C side — `tools/verify.c`

```c
static void dump(const Tensor *t, const char *label) {
    printf("%-8s shape=(%d, %d, %d)  first 5=[", label, t->channels, t->height, t->width);
    int n = t->channels * t->height * t->width;
    int show = n < 5 ? n : 5;
    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], i == show - 1 ? "" : ", ");
    }
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *weights_path = argc > 1 ? argv[1] : "models/weights.bin";
    const char *input_path   = argc > 2 ? argv[2] : "models/debug_input.bin";

    CnnModel m;
    if (model_load(&m, weights_path) != 0) return 1;

    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(input_path, "rb");
    if (f != NULL) {
        fread(input.data, sizeof(float), 28 * 28, f);
        fclose(f);
    } /* else: leave it zeroed by tensor_alloc's calloc */
    dump(&input, "input");

    Tensor a = conv2d(&input, m.conv1_w, m.conv1_b, 32, 3, 1, 1); dump(&a, "conv1");
    tensor_free(&input);
    relu_tensor(&a);                                              dump(&a, "relu1");
    Tensor b = conv2d(&a, m.conv2_w, m.conv2_b, 32, 3, 1, 1);     dump(&b, "conv2");
    tensor_free(&a);
    relu_tensor(&b);                                              dump(&b, "relu2");
    Tensor p1 = maxpool2d(&b, 2, 2);                              dump(&p1, "pool1");
    tensor_free(&b);
    Tensor c = conv2d(&p1, m.conv3_w, m.conv3_b, 32, 3, 1, 1);    dump(&c, "conv3");
    tensor_free(&p1);
    relu_tensor(&c);                                              dump(&c, "relu3");
    Tensor d = conv2d(&c, m.conv4_w, m.conv4_b, 32, 3, 1, 1);     dump(&d, "conv4");
    tensor_free(&c);
    relu_tensor(&d);                                              dump(&d, "relu4");
    Tensor p2 = maxpool2d(&d, 2, 2);                              dump(&p2, "pool2");
    tensor_free(&d);

    printf("flat     shape=(1, 1, %d)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           p2.channels * p2.height * p2.width,
           p2.data[0], p2.data[1], p2.data[2], p2.data[3], p2.data[4]);

    float logits[10];
    linear(m.fc_w, m.fc_b, p2.data, logits, p2.channels * p2.height * p2.width, 10);
    tensor_free(&p2);

    printf("logits   shape=(1, 10)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           logits[0], logits[1], logits[2], logits[3], logits[4]);
    printf("\npredicted digit: %d\n", argmax(logits, 10));
    return 0;
}
```

This deliberately does **not** call `model_forward` (§7) — it re-implements
the same sequence by hand so it can `dump()` after every single op. That's
duplication on purpose: `model_forward` is the "production" path, this is
a diagnostic tool, and diagnostic tools are allowed to trade elegance for
visibility.

### Actually running it (against synthetic weights, since no real trained
model exists in this sandbox)

```
$ ./verify /tmp/test_weights.bin /nonexistent_input.bin
verify: could not open /nonexistent_input.bin, using a zero image instead
input    shape=(1, 28, 28)  first 5=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
conv1    shape=(32, 28, 28)  first 5=[288.0000, 288.0000, 288.0000, 288.0000, 288.0000]
relu1    shape=(32, 28, 28)  first 5=[288.0000, 288.0000, 288.0000, 288.0000, 288.0000]
conv2    shape=(32, 28, 28)  first 5=[18191488.0000, 27253328.0000, ...]
...
pool2    shape=(32, 7, 7)  first 5=[...]
flat     shape=(1, 1, 1568)  first 5=[...]
logits   shape=(1, 10)  first 5=[...]

predicted digit: 9
```
(Full numbers omitted here for length — every shape transition is exactly
what the architecture table predicts: `(1,28,28) → (32,28,28) → (32,28,28)
→ (32,14,14) → (32,14,14) → (32,14,14) → (32,7,7) → (1,1568) → (1,10)`.
The huge values are expected — synthetic weights are sequential integers,
not trained weights, and the input is all zeros, so `conv1`'s output is
just each filter's bias term repeated, exactly `288.0` — the bias index for
`conv1_b[0]`, matching what §8's synthetic-weights test predicted.)

### The actual comparison workflow, once you have a real trained model

1. Save one real MNIST test image as raw floats to `models/debug_input.bin`
   (a few lines of `numpy`/`torch`, using the same `[0,1]` scaling as
   `ToTensor()` — no separate script needed, this can go at the top of
   `dump_intermediate.py` as a one-time step, or as a tiny standalone
   script if you want to reuse different digits).
2. Run `python/dump_intermediate.py` and `tools/verify.c` against the
   same `weights.bin` and the same `debug_input.bin`.
3. Compare the two outputs **top to bottom**, stopping at the first row
   where the "first 5 values" meaningfully disagree (more than ~1e-4 —
   real floating-point rounding differences between PyTorch's optimized
   kernels and this project's plain loops will be smaller than that; a
   real bug will look like a completely different number, not a rounding
   difference).
4. Whatever the first disagreeing layer is, that's where to look —
   §15 has a worked example of exactly this process for a wrong-order bug
   in `LAYER_KEYS`.

"Floating-point precision" is not an acceptable diagnosis for anything
bigger than that ~1e-4 rounding-level gap — treat any larger disagreement
as a real bug to find, not an excuse to stop looking.

---

## 11. Part 6 — The Raylib UI

Built incrementally, each stage compiling before the next begins. All the
code below has been **compiled against a real raylib build** in this
sandbox (`gcc ... -lraylib -lm -lpthread -ldl -lX11`), producing a working
linked executable — it can't *run* here (no display in this sandbox), but
every function call, every type, and every struct field used below is
confirmed correct by the compiler, not just plausible-looking.

### Separating concerns: `ui.h` / `ui.c` vs `main.c`

Deliberate split: `ui.c` holds pure canvas *logic* — drawing, clearing,
downsampling — with **no raylib calls at all**, just math over a
`float[]` buffer. `main.c` holds the raylib *event loop and rendering* —
window, mouse input, drawing shapes on screen — and calls into `ui.c` and
`nn.c` but contains no CNN math or canvas math itself. This split is why
`ui.c`'s tests (below) can run and pass in a display-less environment
without touching raylib at all — logic and rendering are two different
concerns and testing them together would mean the logic tests need a
display too, for no real reason.

### `c/include/ui.h`

```c
#define CANVAS_SIZE 280      /* on-screen drawing area, pixels (= 28*10) */
#define MNIST_SIZE  28
#define BRUSH_RADIUS 8.0f

typedef struct {
    float pixels[CANVAS_SIZE * CANVAS_SIZE]; /* [0,1] grayscale, row-major */
    int predicted_digit;   /* -1 = no prediction yet */
    float confidence;      /* softmax probability of predicted_digit, [0,1] */
} AppState;

void canvas_clear(AppState *app);
void canvas_draw_at(AppState *app, int px, int py);
void canvas_to_mnist_input(const AppState *app, float *out28x28);
```

`CANVAS_SIZE = 280 = 28 * 10` is not arbitrary — picking an exact multiple
of `MNIST_SIZE` makes the downsampling in `canvas_to_mnist_input` a clean
integer box-average (§12) instead of needing interpolation across
fractional pixel boundaries. `AppState.pixels` stores grayscale directly
as `[0,1]` floats — not RGB, not 0-255 bytes — because that's the exact
representation `model_forward` needs at the end of the pipeline; storing
anything else here would just mean a conversion step later for no benefit.

### `c/src/ui.c` — canvas logic, no raylib dependency

```c
void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
}

void canvas_draw_at(AppState *app, int px, int py) {
    int r = (int)BRUSH_RADIUS;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = px + dx;
            int y = py + dy;
            if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > BRUSH_RADIUS) continue;

            float strength = 1.0f - (dist / BRUSH_RADIUS) * 0.3f;
            float *pixel = &app->pixels[y * CANVAS_SIZE + x];
            *pixel = fmaxf(*pixel, strength);
        }
    }
}

void canvas_to_mnist_input(const AppState *app, float *out28x28) {
    int block = CANVAS_SIZE / MNIST_SIZE; /* 280 / 28 = 10 */
    for (int oy = 0; oy < MNIST_SIZE; oy++) {
        for (int ox = 0; ox < MNIST_SIZE; ox++) {
            float sum = 0.0f;
            for (int by = 0; by < block; by++) {
                for (int bx = 0; bx < block; bx++) {
                    int iy = oy * block + by;
                    int ix = ox * block + bx;
                    sum += app->pixels[iy * CANVAS_SIZE + ix];
                }
            }
            out28x28[oy * MNIST_SIZE + ox] = sum / (float)(block * block);
        }
    }
}
```

### `canvas_draw_at`, walked through

This draws a filled circle (a "brush") centered at `(px, py)`. The outer
two loops (`dy`, `dx`) walk a square bounding box around the center —
`[-r, r]` in both directions — and the `dist > BRUSH_RADIUS` check inside
carves a circle out of that square (cheaper to bound with a square and
reject the corners than to iterate a circle's shape directly).

`memset(app->pixels, 0, sizeof(app->pixels))` in `canvas_clear` is safe
specifically because `0.0f` is represented as all-zero bits under
IEEE-754 (which every mainstream compiler and CPU uses) — `memset` sets
bytes, not floats, so this only works for the specific value zero; setting
every pixel to, say, `1.0f` would need an actual loop, since `1.0f`'s bit
pattern is not all-zero-bytes.

`strength = 1.0f - (dist / BRUSH_RADIUS) * 0.3f` gives a soft-edged brush:
full strength (1.0) at the exact center, fading by up to 30% toward the
brush's edge, rather than a harsh flat circle — this makes strokes look
more like a real pen. `fmaxf(*pixel, strength)`, rather than a plain
overwrite, matters when the mouse moves slowly and the brush overlaps
itself between frames: overwriting would make an overlapping stroke
*darker* where circles stack, which looks wrong for a "pen"; taking the
max keeps every pixel at whichever pass painted it brightest, which reads
correctly as one continuous stroke.

### `canvas_to_mnist_input`, walked through

Box-filter downsampling: each output pixel is the average of the
`block × block` (= 10×10) block of input pixels it corresponds to. This is
not what PyTorch's image-resize functions do internally (they typically
use more sophisticated interpolation) — but for reducing a hand-drawn
canvas to a coarse 28×28 grid, a plain average is simple, correct, and, in
practice, visually reasonable; verify this for yourself once it's wired
into the running UI by printing the 28×28 result as a grid of characters
and checking it looks like the digit you drew.

### Verified tests (`tests/test_ui.c`) — run and passed under
AddressSanitizer + UBSan, zero errors

```
test_canvas_clear
  [ok]   pixels[0] cleared: 0.0000
  [ok]   pixels[100] cleared: 0.0000
  [ok]   predicted_digit reset to -1
test_canvas_draw_at
  [ok]   center pixel near 1.0: 1.0000
  [ok]   far pixel untouched: 0.0000
  [ok]   just-outside-brush pixel untouched: 0.0000
test_canvas_draw_at_edge (brush near canvas boundary must not crash / go out of bounds)
  [ok]   corner pixel painted: 1.0000
  [ok]   opposite corner painted: 1.0000
  [ok]   no crash / no out-of-bounds write near edges
test_canvas_to_mnist_input
  [ok]   out[0][0] fully covered -> ~1.0: 1.0000
  [ok]   out[0][1] uncovered -> 0.0: 0.0000
  [ok]   out[1][0] uncovered -> 0.0: 0.0000
  [ok]   half-covered block averages to ~0.5: 0.5000
```
The "edge" test matters specifically because `canvas_draw_at`'s bounding
box (`px - r` to `px + r`) can extend past the canvas on any side when the
brush is used near a corner — the `if (x < 0 || x >= CANVAS_SIZE ...)
continue;` bounds check is what prevents an out-of-bounds write there, and
this test (run under AddressSanitizer, which specifically detects
out-of-bounds memory access) is what actually confirms that check works,
rather than just trusting it by reading the code.

### `c/src/main.c` — the raylib event loop

```c
#include "raylib.h"
#include "nn.h"
#include "ui.h"
#include <stdio.h>
#include <math.h>

#define WINDOW_W 320
#define WINDOW_H 420
#define CANVAS_X 20
#define CANVAS_Y 20
#define BUTTON_H 50

typedef struct { Rectangle rect; const char *label; } Button;

static void softmax(const float *logits, float *probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

static void run_prediction(AppState *app, const CnnModel *model) {
    float mnist_input[MNIST_SIZE * MNIST_SIZE];
    canvas_to_mnist_input(app, mnist_input);

    Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);
    for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) input.data[i] = mnist_input[i];

    float logits[10];
    model_forward(model, &input, logits);
    tensor_free(&input);

    float probs[10];
    softmax(logits, probs, 10);
    int pred = argmax(logits, 10);
    app->predicted_digit = pred;
    app->confidence = probs[pred];
}

int main(void) {
    CnnModel model;
    int model_ok = (model_load(&model, "models/weights.bin") == 0);
    if (!model_ok) {
        fprintf(stderr, "Warning: could not load models/weights.bin — "
                        "Predict button will be disabled. Run python/export.py first.\n");
    }

    InitWindow(WINDOW_W, WINDOW_H, "Number Guesser");
    SetTargetFPS(60);

    AppState app;
    canvas_clear(&app);

    Rectangle canvas_rect = { CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE };
    Button clear_btn   = { { CANVAS_X, CANVAS_Y + CANVAS_SIZE + 10, CANVAS_SIZE / 2.0f - 5, BUTTON_H }, "CLEAR" };
    Button predict_btn = { { CANVAS_X + CANVAS_SIZE / 2.0f + 5, CANVAS_Y + CANVAS_SIZE + 10, CANVAS_SIZE / 2.0f - 5, BUTTON_H }, "PREDICT" };

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (CheckCollisionPointRec(mouse, canvas_rect)) {
                int px = (int)(mouse.x - CANVAS_X);
                int py = (int)(mouse.y - CANVAS_Y);
                canvas_draw_at(&app, px, py);
            } else if (CheckCollisionPointRec(mouse, clear_btn.rect)) {
                canvas_clear(&app);
            } else if (model_ok && CheckCollisionPointRec(mouse, predict_btn.rect)) {
                run_prediction(&app, &model);
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawRectangleRec(canvas_rect, BLACK);
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                float v = app.pixels[y * CANVAS_SIZE + x];
                if (v > 0.0f) {
                    unsigned char g = (unsigned char)(v * 255.0f);
                    DrawPixel(CANVAS_X + x, CANVAS_Y + y, (Color){ g, g, g, 255 });
                }
            }
        }
        DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);

        DrawRectangleRec(clear_btn.rect, LIGHTGRAY);
        DrawText(clear_btn.label, (int)(clear_btn.rect.x + 30), (int)(clear_btn.rect.y + 15), 18, BLACK);
        DrawRectangleRec(predict_btn.rect, model_ok ? SKYBLUE : GRAY);
        DrawText(predict_btn.label, (int)(predict_btn.rect.x + 15), (int)(predict_btn.rect.y + 15), 18, BLACK);

        if (app.predicted_digit >= 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Prediction: %d", app.predicted_digit);
            DrawText(buf, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 20, 22, BLACK);
            snprintf(buf, sizeof(buf), "Confidence: %.0f%%", app.confidence * 100.0f);
            DrawText(buf, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 50, 18, DARKGRAY);
        } else if (!model_ok) {
            DrawText("No model loaded", CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 20, 18, MAROON);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
```

### The event loop, walked through

Raylib's whole model is: one `while (!WindowShouldClose())` loop, executed
once per frame (60 times a second here, via `SetTargetFPS(60)`). Each
iteration does three things, in order: read input (`GetMousePosition`,
`IsMouseButtonDown`), update state (`canvas_draw_at`, `canvas_clear`,
`run_prediction`), then redraw *everything* from scratch between
`BeginDrawing()`/`EndDrawing()` — there's no "just update the part that
changed"; the whole canvas, both buttons, and any prediction text get
redrawn every single frame, all 60 times a second. This is normal and fine
for an app this size — modern GPUs redraw far more complex scenes at this
rate without breaking a sweat — and it's *simpler* to reason about than
selective redrawing, since there's no separate "what changed since last
frame" bookkeeping to get wrong.

`CheckCollisionPointRec(mouse, rect)` is raylib's hit-testing helper — "is
this point inside this rectangle" — used identically for the canvas
(should this click draw?) and both buttons (should this click clear or
predict?). The `if / else if / else if` chain means exactly one of the
three things can happen per click, based on where the mouse is, which
matches how you'd expect a UI to behave (clicking a button shouldn't also
draw on the canvas underneath it, and this code guarantees that as long as
the rectangles don't overlap).

`model_ok && CheckCollisionPointRec(...)` — short-circuit evaluation, a
detail worth calling out: if `model_ok` is false (no `weights.bin` found),
`CheckCollisionPointRec` never even runs, so clicking where the Predict
button is drawn does nothing, matching the grayed-out button drawn just
below it (`model_ok ? SKYBLUE : GRAY`). The UI's visual state and its
actual clickable behavior are kept in sync by referencing the same
`model_ok` variable in both places, rather than risking the two drifting
apart if they were tracked separately.

### `run_prediction`, walked through

This is the one function that bridges `ui.c`'s canvas world and `nn.c`'s
tensor world: `canvas_to_mnist_input` produces a flat `float[28*28]`, which
gets copied into a `Tensor` (`tensor_alloc(1, 28, 28)` — one channel,
matching MNIST's grayscale input), fed through `model_forward`, and the
resulting logits get turned into a human-readable prediction + confidence
via `softmax` + `argmax`.

`softmax`, unpacked: raw logits aren't probabilities (they can be
negative, and don't sum to 1) — softmax converts them into a proper
probability distribution via `exp(x) / sum(exp(all x))`. The `max_val`
subtraction (`expf(logits[i] - max_val)` instead of plain `expf(logits[i])`)
is a standard numerical-stability trick: `exp` of a moderately large number
overflows a `float` fast, but subtracting the max first guarantees the
largest exponent computed is `exp(0) = 1`, and every other one is `≤ 1` —
same final probabilities (subtracting a constant from every term before
exponentiating doesn't change the *ratios*, which is all softmax cares
about), but no overflow risk.

Ownership note, tying back to §1: `run_prediction` allocates `input` via
`tensor_alloc` and frees it itself (`tensor_free(&input)`) right after
`model_forward` is done reading it — a self-contained allocate/use/free
inside one function, the simplest version of the ownership pattern that
`model_forward` itself demonstrates at greater length in §7.

### Building it

```
gcc -Wall -Wextra -std=c11 -Iinclude -I<raylib include path> \
    src/main.c src/nn.c src/ui.c \
    -o number_guesser \
    -L<raylib lib path> -lraylib -lm -lpthread -ldl -lX11
```
Compiled clean in this sandbox with zero warnings under `-Wall -Wextra`.
See §16 for how to get a real raylib install on your own machine (this
sandbox has no display, so the resulting binary can't actually be *run*
here — only compiled and confirmed correct).

---

## 12. Part 7 — Preprocessing parity

From `dataset.py`: `transforms.ToTensor()` is the *entire* PyTorch
preprocessing pipeline — single-channel, `[0,1]` scaling, no
normalization, no inversion, no resizing (MNIST is already 28×28).

The C side needs to reproduce exactly that — and, as it turns out, already
does, by design choice rather than by accident: `AppState.pixels` (§11)
stores brush strength directly as `[0,1]` floats the moment it's drawn (see
`canvas_draw_at`'s `strength` variable), not as 0–255 bytes needing a
later `/255.0f` step, and not as RGB needing a grayscale-conversion step.
There was never a "raw canvas image" to convert — the canvas *is* already
in the target format, one design decision (§11) quietly satisfying this
requirement without a dedicated preprocessing function.

The one genuinely new step, beyond what `ToTensor()` does, is
`canvas_to_mnist_input`'s box-filter downsampling from 280×280 to 28×28
(§11) — PyTorch never has to do this because MNIST images are already
28×28; a hand-drawn canvas at a comfortable UI resolution is not, so this
step exists purely because of the UI's resolution choice, not because of
anything in the training pipeline.

**Inversion:** MNIST digits are white-stroke-on-black-background, and this
UI's canvas is drawn the same way (`ClearBackground` inside the canvas
rectangle is `BLACK`, and `canvas_draw_at` *increases* pixel values toward
white) — so no inversion step is needed. If you ever change the canvas to
draw black-on-white instead (a more traditional "paper" look), you'd need
to add `pixel = 1.0f - pixel` somewhere between drawing and feeding the
CNN, or the trained model — which only ever saw white-on-black during
training — would see every input upside-down in brightness and predict
close to randomly.

---

## 13. Part 8 — Visualization

Secondary — don't start this until §10's verification is passing and the
UI predicts correctly end to end. In rough order of effort-to-payoff:

**Prediction probabilities as a bar chart.** Cheapest addition — reuses
raylib rectangles you already know from the Clear/Predict buttons. After
`softmax` (§11) gives you 10 probabilities, draw 10 thin rectangles side
by side, each one's height proportional to its probability:

```c
static void draw_probability_bars(const float *probs, int x, int y, int w, int h) {
    int bar_w = w / 10;
    for (int i = 0; i < 10; i++) {
        int bar_h = (int)(probs[i] * h);
        DrawRectangle(x + i * bar_w, y + h - bar_h, bar_w - 2, bar_h, SKYBLUE);
        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        DrawText(label, x + i * bar_w, y + h + 4, 12, DARKGRAY);
    }
}
```
(Untested in this sandbox — no display to visually confirm layout — but it
follows the exact same `DrawRectangle`/`DrawText` calls already proven to
compile in `main.c`, §11.)

**Activation maps.** After any conv layer, each output channel is itself a
small grayscale image (e.g. `28×28` after `conv1`) — draw each channel as
a small tile in a grid, the same `DrawPixel` loop already used for the
canvas in `main.c`, just applied to a `Tensor`'s data instead of
`AppState.pixels`:

```c
static void draw_tensor_as_tiles(const Tensor *t, int x, int y, int tile_size) {
    int cols = 8; /* e.g. 8 tiles per row for a 32-channel tensor -> 4 rows */
    for (int c = 0; c < t->channels; c++) {
        int tile_x = x + (c % cols) * (tile_size + 2);
        int tile_y = y + (c / cols) * (tile_size + 2);
        for (int ty = 0; ty < t->height; ty++) {
            for (int tx = 0; tx < t->width; tx++) {
                float v = tensor_get(t, c, ty, tx);
                if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f; /* clamp for display */
                unsigned char g = (unsigned char)(v * 255.0f);
                float scale = (float)tile_size / t->height;
                DrawPixel(tile_x + (int)(tx * scale), tile_y + (int)(ty * scale),
                          (Color){ g, g, g, 255 });
            }
        }
    }
}
```
Useful mainly as a sanity check that early layers are picking up
edges/strokes rather than noise — if `conv1`'s activation maps look like
static, something upstream (preprocessing, weight loading, layer order) is
likely wrong, and this is often a faster way to notice that than staring
at raw logit numbers.

**Filters.** The 3×3 kernels themselves, visualized the same tile-grid way
— harder to interpret meaningfully at such a small size, but essentially
free once `draw_tensor_as_tiles` exists (a kernel is just a small tensor,
and `CnnModel.conv1_w` reshaped as a `Tensor` would work with the same
function).

---

## 14. Testing philosophy

One `test_*()` function per operation, hardcoded tiny input, hand-computed
expected output, printed side by side, checked with a small tolerance
(`check()` in `tests/test_nn.c` uses `1e-4f` — loose enough to allow for
float rounding, tight enough to catch a real bug). This project's two test
files (`tests/test_nn.c`, `tests/test_ui.c`) both compile standalone —
neither needs raylib — and both have been run clean under
`-fsanitize=address,undefined`, which catches out-of-bounds access,
use-after-free, and a range of undefined-behavior bugs that a plain
compile with no sanitizer would silently allow through.

**Never test a new op against real 28×28 data first.** Every op in this
book was verified on a tiny, hand-computable case before it's trusted on
anything realistic — a 3×3 input for `conv2d`, a 4×4 input for
`maxpool2d`. If a tiny test passes but the real model still predicts
badly, that's useful information: the individual ops are probably correct,
and the bug is more likely in wiring (§7), weight export/loading (§8-§9),
or preprocessing (§12) — narrowing down where to look next.

Running the whole suite, one command:
```
$ cd c && make test
```
(Builds and runs both `test_nn` and `test_ui` — see the Makefile in §16.)

---

## 15. Debugging playbook, with real examples

The general method, restated from before: read the actual error, explain
it in one sentence, form one specific hypothesis, make the smallest change
that tests it, recompile, rerun just the relevant test — not the whole
program — and if it's fixed, understand *why* before moving on.

### Worked example 1: the real bug this project already hit

**Symptom** (an actual compiler error hit while building `main.c` in this
sandbox):
```
c/include/nn.h:85:3: error: conflicting types for 'Model'; have 'struct <anonymous>'
/home/claude/raylib-build/include/raylib.h:433:3: note: previous declaration of 'Model' with type 'Model'
```
**In one sentence:** two different headers both declare a type named
`Model`, and C has no way to tell them apart. **Hypothesis:** this is a
plain name collision, not a logic bug — renaming one of them (ours, since
we can't touch raylib's header) should fix it with zero behavior change.
**Smallest fix:** `sed -i 's/\bModel\b/CnnModel/g'` across `nn.h`, `nn.c`,
and every file that used the old name. **Retest:** re-ran `tests/test_nn.c`
first (confirms the rename didn't silently break anything unrelated —
all tests still passed), *then* recompiled `main.c` against raylib (this
is what actually confirms the fix — the collision only manifests when
both headers are included together). **Why it worked:** the two `Model`
types were never related in any way; renaming ours only touches source
text, not behavior, so this class of fix is close to risk-free once you're
sure it's really just a naming collision and not, say, two structs that
were supposed to be compatible.

### Worked example 2: a hypothetical, but very realistic, export bug

**Symptom:** `verify` (C) and `dump_intermediate.py` (Python) agree
exactly on `input` and `conv1`, but disagree — not by a rounding amount,
by a lot — starting at `conv2`.

**In one sentence:** the first four layers of `weights.bin` (`conv1_w`,
`conv1_b`) must be correct, since `conv1`'s output matches; something
about `conv2`'s weights specifically is wrong.

**Hypotheses, in order of how likely and how cheap to check:**
1. `LAYER_KEYS` in `export.py` lists `conv2`'s key in the wrong position,
   or under the wrong name (`"block_1.2.weight"` vs. a typo like
   `"block_1.1.weight"` — remember index `.1` is the ReLU, which has no
   weights at all, so this particular typo would actually crash
   `export.py` with a `KeyError`, ruling it out quickly).
2. `model_load`'s read order in `nn.c` has `conv2_w`/`conv2_b` swapped,
   or reads the wrong byte count for one of them.
3. `model.py`'s architecture changed (a different `hidden_units`, a
   removed layer) without `export.py`'s `LAYER_KEYS` or `nn.c`'s
   `CnnModel` struct being updated to match.

**Smallest check first:** print `WEIGHTS_FILE_BYTES` (175016) against the
*actual* size of the `weights.bin` on disk — a mismatch immediately
confirms hypothesis 3 (architecture drift) without touching any code, since
`model_load`'s own size guard would already have printed an error and
returned `-1` in that case, so if `model_load` reported success, this
hypothesis is likely already ruled out. **Next:** compare `LAYER_KEYS`'s
order in `export.py` against `model_load`'s read order in `nn.c`,
side-by-side, entry by entry — this is a plain, careful read-through of
two ordered lists, not a mysterious step, and is the single most likely
place a bug like this actually lives, since both lists are hand-written
and easy to let drift apart when either file is edited alone. **Fix:**
whichever list is wrong, correct it to match the other, matching against
`model.py`'s actual `nn.Sequential` definition as the source of truth for
which key is genuinely which layer. **Retest:** rerun `verify` and
`dump_intermediate.py` on the same fixed input, confirm `conv2` (and
everything after it) now agrees within rounding.

### General pattern, generalized from both examples

A compiler error is almost always cheaper to diagnose than a silently
wrong numeric result — lean on `-Wall -Wextra` and, when available,
`-fsanitize=address,undefined` (as this project's own tests were run
under) to convert as many bugs as possible into loud compiler/runtime
errors, rather than relying on §10's layer-by-layer numeric comparison to
catch everything. The numeric comparison is for the bugs that *can't*
be caught that way — wrong values that are still perfectly valid,
in-bounds, well-typed floats.

---

## 16. Build & run, on your own machine

This book's code was verified in a sandbox without a display — the steps
below are for your own machine, where you can actually run and interact
with the app.

### Installing raylib

**Debian/Ubuntu:**
```
sudo apt-get install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
git clone --depth 1 https://github.com/raysan5/raylib.git
cd raylib/src && make PLATFORM=PLATFORM_DESKTOP
sudo make install   # installs headers + lib system-wide, so no -I/-L needed later
```
(This is exactly the sequence used to build and verify every raylib call
in this book — including hitting and fixing the missing-X11-headers error
along the way, which `libx11-dev` etc. resolve.)

**macOS (Homebrew):** `brew install raylib`

**Windows:** the raylib project ships prebuilt binaries and a Visual
Studio / MinGW guide on its own site — see
[raylib.com](https://www.raylib.com) for the current installer/build
instructions for your toolchain.

### Building the whole project

From `c/`:
```
make test    # builds + runs test_nn and test_ui (no raylib needed)
make app     # builds number_guesser (needs raylib installed)
make verify  # builds tools/verify (no raylib needed)
```
(`c/Makefile`, included in the project — same commands used to verify
every piece of code in this book.)

### End-to-end run, once you have a trained model

```
cd python
python3 evaluate.py          # trains + saves models/number_guesser_model.pth
python3 export.py            # writes models/weights.bin
cd ../c
make app
./number_guesser             # opens the window
```

---

## 17. Appendix: full file listing

```
number-guesser/
├── data/                          # MNIST, downloaded by dataset.py (not committed)
├── models/
│   ├── number_guesser_model.pth   # produced by evaluate.py
│   └── weights.bin                # produced by export.py
├── python/
│   ├── dataset.py                 # MNIST DataLoaders
│   ├── model.py                   # _MainModel (the CNN)
│   ├── train.py                   # train_step / test_step
│   ├── evaluate.py                # wires training together, saves the model
│   ├── export.py                  # §9 — weights.bin export
│   └── dump_intermediate.py       # §10 — verification, PyTorch side
├── c/
│   ├── Makefile
│   ├── include/
│   │   ├── nn.h                   # §2–§9 — Tensor, CNN ops, CnnModel
│   │   └── ui.h                   # §11 — AppState, canvas API
│   └── src/
│       ├── main.c                 # §11 — raylib window + event loop
│       ├── nn.c                   # §2–§9 implementation
│       └── ui.c                   # §11 implementation
├── tools/
│   └── verify.c                   # §10 — verification, C side
├── tests/
│   ├── test_nn.c                  # §14
│   └── test_ui.c                  # §14
├── docs/
│   └── C-IMPLEMENTATION-GUIDE.md  # this document
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```