# Number Guesser — The Complete Master Reference

## From a Single MNIST Digit to a Full Handwritten-Number OCR System

### Volume I — Foundations, Pipeline, Inference, Verification, and Beyond

> **What this is.** This is the single, unified, superseding master reference for the entire Number Guesser project. It absorbs every earlier guide — the foundations text, the code-first textbook, the forward engineering guide, the project bible, the long-term roadmap, the C lab notebooks, the Python lab notebooks, and the mastery drills — and expands them into one continuous, executable document. It carries the project from "a trained CNN in PyTorch" all the way through "a native C handwritten-number recognizer that measures itself, improves itself, and eventually recognizes arbitrary-length numbers via CTC."
>
> **What this is not.** It is not a checklist. It is not motivational. It is not a collection of snippets to paste. It is a laboratory notebook + textbook + engineering specification combined, designed to be kept open while you work.
>
> **How to read it.** Straight through once, at least for the milestone you are currently on. Then keep it open as reference. Every section with code has a matching "compile this, run this, verify this" block. Do not skip those. The book is designed to be *executed*, not read.

---

## Preface: The Rule of This Book

Do not treat the project as a school assignment to finish. Treat it as:

> **A laboratory where every concept in ML, C, and systems engineering is something you built with your hands and proved with a test.**

For every subsystem, the loop is:

```
Question
  ↓
Theory (why does this exist, what does it compute)
  ↓
Math (derive the shapes and the arithmetic)
  ↓
Python reference (make it work in the comfortable language first)
  ↓
Experiment (measure what it actually does)
  ↓
C implementation (translate it into the honest language)
  ↓
Unit test (prove a tiny case by hand)
  ↓
Parity test (prove it matches the reference)
  ↓
Real-input test (prove it survives reality)
  ↓
Profile (understand the cost)
  ↓
Document (write down what you learned)
```

If you skip from "I know what Conv2D is" to "let me add a fifth layer," you will build a system you cannot debug. If you follow the loop, you will build a system where every layer has evidence behind it.

---

## Table of Contents

This master reference is organized into nineteen parts plus appendices.

**Part 0 — Orientation**
- 0.1 What you are building (system diagram)
- 0.2 The three languages (Python, C, math)
- 0.3 The current repository state
- 0.4 How to use this book
- 0.5 The verification mindset

**Part I — Foundations**
- 1.1 The C memory model
- 1.2 Tensor layout and the Python↔C contract
- 1.3 Binary serialization and endianness
- 1.4 Why C for inference (and why not, sometimes)
- 1.5 Pointers, arrays, and the flat buffer
- 1.6 Structs, ownership, and lifetime

**Part II — The Python Training Pipeline**
- 2.1 `dataset.py` — loading MNIST honestly
- 2.2 `model.py` — the CNN, and why each shape
- 2.3 `train.py` — one epoch, one step
- 2.4 `evaluate.py` — the training loop and checkpointing
- 2.5 `export.py` — writing `weights.bin`
- 2.6 `helper_functions.py` — utilities
- 2.7 The full end-to-end training recipe

**Part III — The C Inference Engine**
- 3.1 `nn.h` — the header and its contracts
- 3.2 Tensor operations — alloc, free, get, set
- 3.3 Linear, ReLU, Argmax
- 3.4 Conv2D — the heart
- 3.5 MaxPool2D
- 3.6 Model loading
- 3.7 The full forward pass
- 3.8 Ownership discipline
- 3.9 The `ui.h` / `ui.c` canvas
- 3.10 `main.c` — the Raylib application
- 3.11 Preprocessing as a first-class component

**Part IV — Verification**
- 4.1 Why "it compiles and runs" is not verification
- 4.2 `dump_intermediate.py` — the PyTorch reference dump
- 4.3 `verify.c` — the C reference dump
- 4.4 Comparing layers — the workflow
- 4.5 Numerical tolerance — what "same" means
- 4.6 Real-weight verification — the milestone that matters
- 4.7 What to do when they disagree
- 4.8 The debugging playbook

**Part V — Testing and Build**
- 5.1 Test philosophy
- 5.2 Unit tests
- 5.3 The Makefile
- 5.4 CMake, and when it earns its keep
- 5.5 Sanitizers
- 5.6 CI
- 5.7 The test assertion library

**Part VI — Real Handwriting Evaluation**
- 6.1 Why MNIST accuracy isn't enough
- 6.2 Building a handwriting dataset
- 6.3 The batch evaluator
- 6.4 Metrics — accuracy, confusion matrix, per-class
- 6.5 Confidence calibration
- 6.6 Error analysis
- 6.7 The failure gallery

**Part VII — ML Improvement**
- 7.1 The experiment harness
- 7.2 Preprocessing experiments
- 7.3 Data augmentation
- 7.4 Training improvements
- 7.5 Architecture experiments
- 7.6 The experiment log format

**Part VIII — Observability**
- 8.1 Activation visualization
- 8.2 Layer timing
- 8.3 The debugging playbook
- 8.4 Logits and softmax inspection

**Part IX — Performance Engineering**
- 9.1 Measure first
- 9.2 Memory reuse
- 9.3 Cache-aware convolution
- 9.4 SIMD, eventually
- 9.5 Quantization
- 9.6 Compiler flags

**Part X — Two-Digit Recognition**
- 10.1 The two-digit problem
- 10.2 Connected-component segmentation
- 10.3 Flood fill in C
- 10.4 Bounding boxes and sorting
- 10.5 Number decoding
- 10.6 Failure cases
- 10.7 Projection-based segmentation
- 10.8 When segmentation stops working

**Part XI — Variable-Length OCR**
- 11.1 Why segmentation stops working
- 11.2 Sliding windows and feature sequences
- 11.3 CTC — the intuition
- 11.4 CTC in practice
- 11.5 Decoding — greedy and beam search
- 11.6 The full OCR architecture

**Part XII — C Engineering for OCR**
- 12.1 Sequence types
- 12.2 Error propagation
- 12.3 Model format v2
- 12.4 Determinism and experiment metadata
- 12.5 Model loader hardening

**Part XIII — Backpropagation (Optional Keystone)**
- 13.1 Why you might want to
- 13.2 Linear layer, forward and backward
- 13.3 ReLU
- 13.4 MaxPool
- 13.5 Conv2D
- 13.6 Gradient checking
- 13.7 A complete tiny training loop from scratch

**Part XIV — Reference**
- 14.1 The complete file listing
- 14.2 Mathematics reference
- 14.3 Numerical reference
- 14.4 Debugging playbook
- 14.5 Glossary

**Part XV — Long-Term Roadmap**
- 15.1 Milestones, in order
- 15.2 What "done" means at each level
- 15.3 What not to do

**Part XVI — Labs**
- 16.1 C array lab
- 16.2 C struct lab
- 16.3 C heap lab
- 16.4 C file lab
- 16.5 Python numpy lab
- 16.6 Python shape lab
- 16.7 Python ReLU lab
- 16.8 Python softmax lab
- 16.9 Python gradient lab
- 16.10 Python conv lab

**Part XVII — Mastery Drills**
- 17.1 Round 1
- 17.2 Round 2
- ... (many rounds)
- 17.N Round N

**Part XVIII — The Study Contract**
- 18.1 How to read this book
- 18.2 The workflow
- 18.3 The final rulebook

**Part XIX — Closing**
- 19.1 What "done" means
- 19.2 The point of this project

---

# Part 0 — Orientation

## 0.1 What You Are Building

The finished system, at the highest level:

```
                         ┌─────────────────────┐
                         │      Dataset        │
                         │ MNIST / EMNIST /    │
                         │ synthetic / custom  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Training        │
                         │      PyTorch        │
                         │ CNN / OCR model     │
                         └──────────┬──────────┘
                                    │
                              trained model
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Export / Format    │
                         │ deterministic       │
                         │ binary weights      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                 Python reference        Native C inference
                 implementation          implementation
                         │                     │
                         └──────────┬──────────┘
                                    │
                             parity verification
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Raylib UI        │
                         │ draw / clear /      │
                         │ predict / visualize │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Single digit        │
                         │ recognition         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Multi-digit image   │
                         │ segmentation / OCR  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Number decoder      │
                         │ 7 / 42 / 128 / ...  │
                         └─────────────────────┘
```

Four coupled systems live in this diagram:

1. **The ML model** — a CNN, trained in PyTorch, whose job is to turn a 28×28 grayscale image into 10 logits.
2. **The model artifact** — `weights.bin`, a headerless binary file, whose every byte must match what the C side expects.
3. **The C inference engine** — a hand-written implementation of the exact same forward pass PyTorch computes, with the same weights.
4. **The preprocessing pipeline** — turning a 280×280 mouse drawing into a 28×28 grayscale tensor that looks, statistically, like something MNIST would have produced.

Every prediction the application makes is the output of all four. If it is wrong, any of the four could be the cause. This is why verification (Part IV) and evaluation (Part VI) come before improvement (Part VII).

## 0.2 The Three Languages

You will think and work in three languages simultaneously, and this book will always make clear which one a given section is in.

**Mathematics** is where ideas live. When you derive the output shape of a convolution (`out = floor((N + 2P - K)/S) + 1`), you are working in math. When you derive the gradient of cross-entropy with respect to logits (`∂L/∂z_i = p_i - 1(i=y)`), you are working in math. Math is where you go when you want to know *why* something is true, not *that* it is.

**Python (PyTorch)** is where the model is trained and where the reference implementation lives. Python is allowed to be comfortable. It is allowed to use libraries. It is allowed to be slow. It is the place where you prototype, where you experiment, and where you produce the numbers that the C side has to match. When Python and C disagree, Python is the arbiter — provided it agrees with math.

**C** is where the inference engine lives. C is deliberately uncomfortable: no autograd, no garbage collector, no high-level tensor type, no broadcasting, no `.to(device)`. Every operation is a nested loop you wrote. Every allocation is one you own. The reward for this discipline is that you understand *exactly* what a forward pass costs, in operations and bytes, at every level.

The three are not independent. Every Conv2D has a math definition, a PyTorch implementation, and a C implementation, and the whole point of the project is that you can point at any one of them and explain how it corresponds to the other two.

## 0.3 The Current Repository State

The current repository is `sahandkhodayi/Number-Guesser`. The verified current architecture is:

```
1×28×28
→ Conv 1→32, 3×3, stride 1, padding 1
→ ReLU
→ Conv 32→32, 3×3, stride 1, padding 1
→ ReLU
→ MaxPool 2×2, stride 2
→ Conv 32→32, 3×3, stride 1, padding 1
→ ReLU
→ Conv 32→32, 3×3, stride 1, padding 1
→ ReLU
→ MaxPool 2×2, stride 2
→ Flatten 32×7×7 = 1568
→ Linear 1568→10
→ 10 logits
```

The current repository contains Python training/evaluation/export code, a native C inference runtime, Raylib UI code, CMake, `models/`, `data/`, `benchmark/`, and `tests/`.

Important distinctions used throughout this book:

- **CURRENT** means the repository already contains it.
- **TARGET** means this book asks you to create it.
- **EXPERIMENT** means it is optional work used to learn or measure something.
- **DO NOT CLAIM DONE** means you must run the verification yourself.

## 0.4 How to Use This Book

Do not read this book the way you read a novel. Read it the way you read a lab manual: with a terminal open, a scratch file for hand-derivations, and the actual repository on disk.

When you hit a section that says "compile this":

Compile it.

When it says "run this":

Run it.

When it says "check that this equals X":

Check.

When it says "break this intentionally, then figure out why":

Break it. That is not a digression; that is the point.

The rhythm is:

```
Read a section
  ↓
Type the code (do not paste)
  ↓
Run it
  ↓
Predict the output before you run it
  ↓
Observe the output
  ↓
If it disagrees with your prediction, understand why before moving on
  ↓
Change one thing, predict again, run again
```

That last step — deliberately changing one thing and re-predicting — is the difference between "I read this and understood it" and "I own this." You will use it constantly. It is the same skill you will use when you debug the model, when you run an experiment, and when you profile.

## 0.5 The Verification Mindset

The single most important sentence in this book:

> **Compiling and running without crashing proves that the code has no memory errors. It does not prove the code is correct.**

You can write a `conv2d` that has the weight index transposed, and it will compile cleanly, pass sanitizers, run in the UI, and produce predictions. Those predictions will be wrong — but not so wrong that they look like a crash. They will just be subtly, plausibly wrong, in a way that is easy to miss.

This is the class of bug that verification exists to catch.

**Verification** means: running the same input through both PyTorch and C, and comparing the outputs at every stage, within a numerical tolerance, and confirming that the differences are small enough to attribute to floating-point rounding rather than to a bug.

Adopt the mindset: **do not claim done until you have the command that proves it.** "It works" is not a proof. "I ran `python tools/compare.py` and every layer reported `max_diff < 1e-4`" is a proof.

---

# Part I — Foundations

## 1.1 The C Memory Model

Every C bug in this project traces back to two questions:

> Where does this memory live, and who owns it?

Get these right and C feels like a language. Get them wrong and you get segfaults, silent corruption, or leaks that surface months later. The memory model is not an abstract concept to memorize; it is the vocabulary you use to describe what every `Tensor`-returning function does.

### 1.1.1 Stack vs. heap

**Stack memory** is allocated when a function is entered, and freed the instant it returns:

```c
void foo(void) {
    int x = 42;            /* stack */
    float arr[10];         /* stack */
    /* ... */
}   /* x and arr are destroyed right here */
```

Characteristics:

- Fast — allocating on the stack is just moving a pointer.
- Size known at compile time.
- Freed automatically on return.
- **Cannot be returned as a pointer.** The memory is gone the moment the function returns. If you return `&arr[0]`, you are returning a pointer to memory that is about to be reused for something else.

**Heap memory** is allocated explicitly, and freed explicitly:

```c
void foo(void) {
    int *p = malloc(sizeof(int));   /* heap */
    *p = 42;
    /* ... */
    free(p);                        /* freed here, not on return */
}
```

Characteristics:

- Size can be determined at runtime.
- Lives until you call `free()`.
- Can be returned across function boundaries.
- **You must free it, or it leaks.**

**The rule for this project**: small fixed-size buffers go on the stack; anything whose size depends on runtime values goes on the heap.

Concretely:

```c
float logits[10];                        /* stack — always exactly 10 */
Tensor a = tensor_alloc(32, 28, 28);     /* heap — size known only at runtime */
```

The `logits[10]` array lives in the stack frame of whatever function declares it. The `Tensor a` owns a heap pointer; the `Tensor` struct itself (`a.channels`, `a.height`, `a.width`, `a.data`) lives wherever it was declared, but the actual data — the 32×28×28 floats — lives on the heap.

### 1.1.2 Ownership

Every heap-allocated pointer has exactly **one owner** — the piece of code responsible for calling `free()` on it. Everyone else is a borrower.

The rule for this project:

- A function that **allocates and returns** a `Tensor` hands ownership to its caller.
- A function that **reads or mutates in place** never frees; it borrows.

Consider `conv2d`:

```c
Tensor conv2d(const Tensor *input,          /* borrowed — we do not free */
              const float *weights,          /* borrowed — never freed here */
              const float *bias,             /* borrowed */
              int out_channels, int k, int stride, int pad) {

    int out_h = (input->height + 2 * pad - k) / stride + 1;
    int out_w = (input->width  + 2 * pad - k) / stride + 1;

    Tensor out = tensor_alloc(out_channels, out_h, out_w);   /* we allocate — we own — we return */

    /* ... fill out ... */

    return out;   /* ownership transfers to the caller */
}
```

And `model_forward`:

```c
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out) {

    Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);   /* we own `a` */
    relu_tensor(&a);                                                  /* mutate in place — no new owner */
    Tensor b = conv2d(&a, m->conv2_w, m->conv2_b, 32, 3, 1, 1);       /* we own `b` */
    tensor_free(&a);                                                  /* a consumed — free now */
    relu_tensor(&b);
    Tensor p1 = maxpool2d(&b, 2, 2);
    tensor_free(&b);                                                  /* b consumed */

    Tensor c = conv2d(&p1, m->conv3_w, m->conv3_b, 32, 3, 1, 1);
    tensor_free(&p1);
    relu_tensor(&c);

    Tensor d = conv2d(&c, m->conv4_w, m->conv4_b, 32, 3, 1, 1);
    tensor_free(&c);
    relu_tensor(&d);

    Tensor p2 = maxpool2d(&d, 2, 2);
    tensor_free(&d);

    int in_features = p2.channels * p2.height * p2.width;   /* 1568 */
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);

    tensor_free(&p2);
}
```

Read that function again, watching the ownership at each line. There are eight heap-allocated tensors in the whole forward pass — `a`, `b`, `p1`, `c`, `d`, `p2` — plus the input (allocated by the caller) and the logits array (also the caller's). At any moment, at most two are alive. And every single one is freed.

**Peak memory**: roughly `32·28·28 + 32·28·28` floats = 200 KB. Not eight tensors' worth. This is why the "free as soon as you are done with it" pattern matters — it is not just good hygiene, it is what keeps the peak memory small.

### 1.1.3 Flat memory and 3D indexing

C has no true multi-dimensional runtime-sized array. A `float[32][28][28]` requires compile-time constants. So every "tensor" in this project is a single `float*` plus three integers, and we compute the flat offset by hand.

For a tensor with `C` channels, `H` rows, `W` columns:

```
offset(c, y, x) = (c * H + y) * W + x
```

Read it inside-out:

1. `c * H` — skip `c` whole channels' worth of rows.
2. `+ y` — walk down `y` rows within the selected channel.
3. `* W` — convert rows to elements.
4. `+ x` — add the column offset.

Worked example: in a `(2, 3, 3)` tensor (2 channels, 3 rows, 3 cols), the element at `(c=1, y=2, x=0)`:

```
(1 * 3 + 2) * 3 + 0 = 5 * 3 + 0 = 15
```

Element 15 of the flat array. This is what `tensor_get` and `tensor_set` compute.

```c
float tensor_get(const Tensor *t, int c, int y, int x) {
    size_t index =
        ((size_t)c * (size_t)t->height + (size_t)y) *
        (size_t)t->width + (size_t)x;
    return t->data[index];
}
```

Note the `(size_t)` casts. Every one of them matters. Without them, if `c`, `height`, `y`, `width` were each up to ~10,000, `c * height` could exceed `INT_MAX` (about 2.1 billion) and overflow — a silent, undefined-behavior-producing bug. With `size_t` (unsigned 64-bit on any modern platform), the intermediate products are safe up to about `1.8 × 10^19`.

**The critical bug class**: writing `(c * H + y) * (W + x)` instead of `(c * H + y) * W + x`. Same symbols, wildly different addresses. Adding `x` to `W` before multiplying by the channel-row offset shifts every element by `x` whole rows. This is why the test suite includes "neighbor unaffected" checks — after setting one cell, verify that its neighbors are still zero. Without that check, a wrong formula could produce a program that "runs fine" while silently corrupting adjacent cells.

### 1.1.4 The memory layout is a contract

Our `Tensor` uses **channel-major** layout: all of channel 0's data, then all of channel 1's, and so on. Within a channel, rows are stored top-to-bottom, and within a row, columns left-to-right.

This is not an arbitrary choice. It is exactly what PyTorch uses by default — `[N, C, H, W]` with C as the second-fastest-varying dimension after W, and C-contiguous memory meaning "last index varies fastest."

That agreement between the two sides is what makes `export.py` a straight byte copy with no reordering. It is also what will silently break everything if you ever change it.

If you ever change the layout, you must change all of:

- `tensor_get` and `tensor_set`.
- `conv2d`'s weight indexing (`w_idx = ((oc * C_in + ic) * k + ky) * k + kx`).
- `maxpool2d`'s channel loop.
- `export.py`'s write order.
- Every test that checks numeric values.

Change it in one place and not the others, and the entire pipeline silently breaks — the weights load without error, the forward pass runs without crashing, and predictions are garbage. That is the worst possible failure mode: silent, plausible-looking wrongness. Which is why the next section matters.

### 1.1.5 A note on `calloc` vs `malloc`

The `tensor_alloc` in this project uses `calloc`, which zero-initializes. This is deliberate: if there is a bug where a tensor cell is read before it is written, `malloc` gives you garbage floats (which might be NaN, might be huge, might be small — hard to spot), while `calloc` gives you a suspicious `0.0`. That is easier to recognize as a bug when you see it in a debugger or a print statement.

Use `calloc` for tensors. Use `malloc` only where you intend to immediately overwrite every byte.

### 1.1.6 Use-after-free and the defensive NULL-set

`tensor_free` sets `t->data = NULL` after calling `free`:

```c
void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = 0;
    t->height = 0;
    t->width = 0;
}
```

After `free`, the pointer is dangling: it still holds the old address, but that address is no longer valid. Setting it to `NULL` means that any accidental use-after-free immediately segfaults, instead of silently reading whatever the allocator has since reused the memory for. That is a strictly better failure mode.

The shape metadata is also zeroed, for the same reason: after freeing, the tensor has no meaningful shape. Setting it to zero makes accidental reuse of a freed tensor produce obviously-wrong behavior (all shapes zero) rather than subtly-wrong behavior (shape says 32×28×28, but data is garbage).

## 1.2 Tensor Layout and the Python↔C Contract

The `Tensor` struct in `nn.h`:

```c
typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} Tensor;
```

Three integers and a pointer. The pointer is to a heap-allocated flat array of `channels * height * width` floats, in channel-major, row-major order.

The Python side of the contract lives in the model definition. In PyTorch:

```python
self.block_1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
)
```

Each `Conv2d` has a `.weight` of shape `(out_channels, in_channels, kernel_size, kernel_size)`, stored in that exact nesting order in memory. `conv2d` in C indexes weights as:

```c
int w_idx = ((oc * input->channels + ic) * k + ky) * k + kx;
```

which is exactly the flattened form of `[oc][ic][ky][kx]`. Same convention, same order, no reordering needed.

**The contract** is therefore:

| Concept | PyTorch | C |
|---|---|---|
| Channel order | `[C, H, W]` | `(c * H + y) * W + x` |
| Weight order | `[out, in, ky, kx]` | `((oc * C_in + ic) * k + ky) * k + kx` |
| Dtype | `float32` | `float` |
| Row-major | yes (default) | yes |

Every one of these must remain true. When you change the model, you change all four at once, or none.

There is a subtlety worth naming: the agreement is not automatic. PyTorch's tensors can be non-contiguous (e.g., after a transpose), and `.contiguous()` is what forces them into the standard layout before `.numpy().tobytes()`. The `export.py` code deliberately calls `.contiguous()` even though it is a no-op for freshly-loaded `state_dict` tensors — the call documents the invariant and defends against ever changing PyTorch's default behavior.

## 1.3 Binary Serialization and Endianness

`weights.bin` is 175,016 bytes: 43,754 floats, each 4 bytes, laid out in this exact order:

```
Offset      Size        Contents
─────────────────────────────────────────────────
0x00000     1152 B      conv1_w   (288 floats)
0x00480     128 B       conv1_b   (32 floats)
0x00500     36864 B     conv2_w   (9216 floats)
0x0E500     128 B       conv2_b   (32 floats)
0x0E580     36864 B     conv3_w   (9216 floats)
0x1C600     128 B       conv3_b   (32 floats)
0x1C680     36864 B     conv4_w   (9216 floats)
0x26D80     128 B       conv4_b   (32 floats)
0x26E00     62720 B     fc_w      (15680 floats)
0x36240     40 B        fc_b      (10 floats)
─────────────────────────────────────────────────
Total:      175016 B    (43754 floats)
```

**There is no header.** No magic number, no version tag, no dtype indicator, no tensor metadata. Just raw floats. This works because both sides — `export.py`'s `LAYER_KEYS` and `model_load`'s hardcoded read order — already agree, in the source code, on the answer to every question a header would have answered.

**Why this is fine, for now**: the file exists to communicate weights, and both sides of the conversation already know how to interpret weights. A header would encode information that no one is uncertain about.

**Why this is not fine, forever**: the moment you have more than one model (a single-digit model and a two-digit model and an OCR model), you need a way to distinguish them. The moment you want to support different dtypes (float16, quantized int8), you need a way to indicate which one a file uses. The moment you want to support loading a model on a machine with different endianness, you need a byte-order convention.

The plan for a v2 format (Part XII) is documented there. For now, keep the raw format, but be explicit that it is a contract written in code, not a specification written in a file.

**Endianness** — the byte order in which a multi-byte value is stored — is worth understanding even though it is a non-issue for this project's actual setup. x86-64 is little-endian: the least-significant byte of a float comes first in memory. ARM64, in most modern uses, is also little-endian. If `export.py` runs on a little-endian machine and `model_load` runs on a little-endian machine, the bytes on disk are read back correctly with zero conversion. Both `fwrite` (via `.tobytes()`) and `fread` just move raw bytes.

If you ever tried to load a `weights.bin` on a big-endian platform (some embedded targets, historically some POWER and MIPS machines, and a handful of others), every float would be byte-swapped and predictions would be garbage. The fix is a byte-order check in `model_load` — either a compile-time check against a known file-endianness marker (would require a header) or a runtime check (also requires a header). Neither exists yet, and neither needs to exist until such a platform is a real target.

**Reading the file in C**: `model_load` uses the `fseek`/`ftell`/`fseek` idiom to determine the file size, checks it against `sizeof(CnnModel)` before reading a single float, then calls `read_floats` on each array in the exact order that `export.py` wrote them:

```c
int model_load(CnnModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "model_load: could not open '%s'\n", path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long size = ftell(f);
    if (size < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

    if ((unsigned long)size != sizeof(CnnModel)) {
        fprintf(stderr, "model_load: '%s' is %ld bytes, expected %zu\n",
                path, size, sizeof(CnnModel));
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

`sizeof(CnnModel)` is computed by the compiler from the array sizes in the struct. If you add a layer, `sizeof` updates automatically, and a stale `weights.bin` fails the size check with a clear error. That is the point of computing the expected size from the struct rather than hardcoding 175016.

## 1.4 Why C for Inference

You are going to write the forward pass twice — once in PyTorch, once in C. Why?

**The official reason**: deployment. A production inference system often runs in environments where Python is not available, or is too slow, or has too large a dependency footprint. C is the language of "runs on anything, starts instantly, uses exactly the memory you tell it to."

**The real reason, for this project**: C is the language where you cannot accidentally hide a computation. When you write `logits = model(x)` in PyTorch, a hundred operations happen under the hood, and you never see them. When you write `model_forward(&m, &input, logits)` in C, you wrote every nested loop, and you can point at every multiplication.

This is the pedagogical value of the C implementation, and it is why it is worth the effort even if you never actually deploy the model.

**The honest caveat**: C is not always the right choice. For batch inference on a GPU, PyTorch or TensorRT wins. For rapid experimentation, Python wins. For a small model on a modern CPU, C wins for latency and memory footprint, but the margin is smaller than you might expect. What C always wins at, for a project like this, is *transparency*.

The plan for the rest of this book is that the C implementation stays the *reference implementation* — the one you can point at and say "this is what inference is, at the level of arithmetic." Every optimization, every new model, everything else goes through the Python side first, then gets translated to C, then gets verified against Python.

## 1.5 Pointers, Arrays, and the Flat Buffer

### 1.5.1 The pointer experiment

Before you touch the CNN, understand pointers. Create a scratch file and run it:

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;

    printf("x = %d\n", x);
    printf("*p = %d\n", *p);

    *p = 99;

    printf("x = %d\n", x);
    return 0;
}
```

Line by line:

- `#include <stdio.h>` — Includes the standard I/O declarations so `printf` is available.
- `int main(void) {` — Defines the program entry point. `void` means this function takes no arguments.
- `int x = 42;` — Creates an integer named `x` and initializes it to 42.
- `int *p = &x;` — Creates a pointer to `int`. `&x` means "the address of x".
- `printf("x = %d\n", x);` — Prints the value stored directly in x.
- `printf("*p = %d\n", *p);` — Dereferences p. `*p` means "the integer stored at the address held by p".
- `*p = 99;` — Changes the integer through the pointer.
- `printf("x = %d\n", x);` — Prints x again, proving that p pointed to x itself.
- `return 0;` — Returns success to the operating system.

**Predict the output before you run it.** Then run it. Then change `*p = 99` to `*p = 1000`, and predict again.

### 1.5.2 Pointer arithmetic

```c
#include <stdio.h>

int main(void) {
    int values[4] = {10, 20, 30, 40};

    int *p = values;

    for (int i = 0; i < 4; ++i) {
        printf("%d\n", *(p + i));
    }

    return 0;
}
```

Line by line:

- `int values[4] = {10, 20, 30, 40};` — Creates four contiguous integers.
- `int *p = values;` — An array expression used without indexing becomes a pointer to its first element.
- `for (int i = 0; i < 4; ++i) {` — Moves through the array.
- `printf("%d\n", *(p + i));` — `p + i` moves by i integers, not i bytes. `*(p+i)` reads the value.

**Predict**: what would `*(p + 1)` print? What about `*p + 1`? These are different. `*(p+1)` is the *second element*; `*p + 1` is the *first element plus one*. Work them out on paper before running.

### 1.5.3 Pointer arithmetic moves by element size, not bytes

This is the single most confusing thing about C pointers. If `p` is `int*` and `sizeof(int) == 4`, then `p + 1` advances the address by 4 bytes, not 1. If `p` is `float*`, `p + 1` advances by 4 bytes. If `p` is `Tensor*`, `p + 1` advances by `sizeof(Tensor)` bytes.

This is *why* array indexing works. `values[i]` is defined to be `*(values + i)`, which is "the int at address `values + i * sizeof(int)`". The compiler handles the multiplication for you.

### 1.5.4 Flat buffers

C has no true 2D array whose dimensions are known at runtime. What you do instead is allocate a 1D buffer of size `width * height` and compute the row-major offset yourself:

```c
float *pixel(const float *image, int width, int row, int col) {
    return (float *)&image[row * width + col];
}
```

That is exactly what `tensor_get` does, generalized to 3D.

### 1.5.5 The tensor indexing demonstration

Create `tests/tensor_index_demo.c`:

```c
#include <stdio.h>
#include <stddef.h>

static size_t index3d(int c, int y, int x, int height, int width) {
    return ((size_t)c * (size_t)height + (size_t)y) * (size_t)width
           + (size_t)x;
}

int main(void) {
    printf("%zu\n", index3d(1, 1, 2, 2, 3));
    return 0;
}
```

- `#include <stdio.h>` — Provides `printf`.
- `#include <stddef.h>` — Provides `size_t`, the unsigned integer type commonly used for memory sizes and array indices.
- `static size_t index3d(...)` — Defines the exact channel-first indexing formula used by the project.
- `return ((size_t)c * (size_t)height + (size_t)y) * (size_t)width + (size_t)x;` — Converts the dimensions to `size_t` before multiplication to keep the arithmetic in an appropriate unsigned size type.
- `printf("%zu\n", index3d(1, 1, 2, 2, 3));` — Prints the expected flat index 11.

**Predict the output before running.** It should be 11.

## 1.6 Structs, Ownership, and Lifetime

A `struct` bundles multiple values under one name:

```c
typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} Tensor;
```

The struct is a *value type*: when you write `Tensor t;`, you get a struct on the stack. When you write `Tensor *p = &t;`, you get a pointer to that struct. The `.` operator accesses a field of a value; the `->` operator accesses a field of a pointer. `p->data` is shorthand for `(*p).data`.

`tensor_alloc` returns a `Tensor` *by value* — the struct (a pointer plus three integers, 16 or 24 bytes total, depending on alignment) is copied out of the function on return. The data it points to stays on the heap. This is the standard pattern for value-typed handles.

Lifetime rules:

1. A `Tensor` value (the struct) lives as long as the variable that holds it — stack frame for locals, heap for `malloc`'d structs.
2. The data a `Tensor` points to lives until `tensor_free` is called.
3. Returning a `Tensor` by value copies the handle but does not copy the data.

This is why `model_forward` can have eight tensors alive at various points without any of them accidentally sharing memory: each `tensor_alloc` returns a fresh handle with its own fresh data.

---

# Part II — The Python Training Pipeline

The Python side of this project has one job: produce a `weights.bin` file that the C side can read, whose contents are numerically correct. Everything else — model definition, training loop, checkpointing — exists in service of that.

Python is allowed to be comfortable. It is allowed to use PyTorch, torchvision, numpy, and anything else that speeds up the training process. The whole point of the C/Python split is that Python handles the messy, expensive, experimental parts of building a model, and C handles the deployment-shaped, arithmetic-heavy part of running it.

## 2.1 `dataset.py` — Loading MNIST Honestly

MNIST is the canonical "hello world" of image classification. It is 60,000 training images and 10,000 test images, all 28×28 grayscale, each showing a single handwritten digit. Every digit has been centered and normalized so that any reasonable model can learn to classify them.

The full `dataset.py`:

```python
"""MNIST dataset loaders for the Number Guesser project.

Two DataLoaders:
    train_loader — shuffled, batch size 64
    test_loader  — not shuffled, batch size 64

The only preprocessing is ToTensor(), which:
    1. Converts the PIL image to a torch.Tensor
    2. Scales pixel values from [0, 255] to [0.0, 1.0]
    3. Arranges the tensor as (C, H, W) = (1, 28, 28)

There is NO normalization. There is NO augmentation here.
Augmentation is added later, in the experiments phase, so the
baseline stays reproducible.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(batch_size: int = 64, data_root: Path | str = "data"):
    """
    Return (train_loader, test_loader) for MNIST.

    Arguments:
        batch_size — number of samples per gradient step
        data_root  — directory where MNIST will be downloaded/cached

    Returns:
        (train_loader, test_loader)
    """

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
```

### 2.1.1 Why `ToTensor` and nothing else

The temptation, when you read any MNIST tutorial, is to add:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

The `(0.1307, 0.3081)` is the mean and std of MNIST. Normalizing subtracts the mean and divides by the std, which puts the pixel values into a roughly standard-normal distribution. It often improves training stability and final accuracy slightly.

The problem is that if you normalize on the Python side, you must normalize *identically* on the C side, or predictions will be garbage. And the C side does not currently normalize. So either you add normalization to both sides (which is a legitimate choice — the transform becomes part of the model's specification), or you do not normalize at all (which is what we do).

We do not normalize. `ToTensor()` is the entire preprocessing. This keeps the contract between Python and C simple: the model expects `[0, 1]` grayscale, and both sides feed it that.

**There is a real cost to this choice**: training is slightly slower to converge, and the model is slightly more sensitive to outliers than it would be with normalization. But the simplicity is worth it for this project, and it forces you to be honest about what "preprocessing" means — if you later add normalization, you must add it to *both* sides, and you must verify the C side agrees.

### 2.1.2 What can go wrong

- **Wrong download directory**: if `data_root` is relative and you change the working directory, MNIST gets re-downloaded. Use an absolute path or be careful about where you run the script from.
- **Forgetting `ToTensor()`**: the images come through as PIL images, not tensors. `DataLoader` still works, but `model(x)` crashes on the first batch. This is actually a *good* failure mode — it fails loudly and immediately.
- **Adding `Normalize` without updating C**: this is the bad failure mode. The model trains beautifully, the C side loads the weights, the forward pass runs without error, and predictions are wrong in a subtle, hard-to-debug way. The verification pipeline (Part IV) catches this, which is why verification exists.

## 2.2 `model.py` — The CNN, and Why Each Shape

The full model:

```python
"""CNN architecture for the Number Guesser project.

Architecture (input is a 1×28×28 grayscale image):

    block_1:
        Conv2d(1  -> 32, kernel_size=3, stride=1, padding=1)
        ReLU
        Conv2d(32 -> 32, kernel_size=3, stride=1, padding=1)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)

    block_2:
        Conv2d(32 -> 32, kernel_size=3, stride=1, padding=1)
        ReLU
        Conv2d(32 -> 32, kernel_size=3, stride=1, padding=1)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)

    classifier:
        Flatten
        Linear(1568 -> 10)

Shape flow:

    1  × 28 × 28
    ↓ conv1
    32 × 28 × 28
    ↓ conv2
    32 × 28 × 28
    ↓ pool1
    32 × 14 × 14
    ↓ conv3
    32 × 14 × 14
    ↓ conv4
    32 × 14 × 14
    ↓ pool2
    32 × 7 × 7
    ↓ flatten
    1568
    ↓ linear
    10 logits
"""

import torch.nn as nn


class _MainModel(nn.Module):
    """
    The CNN. The leading underscore is deliberate: this is the
    project's primary model and should not be confused with any
    experiments that might live alongside it later.
    """

    def __init__(self, input_shape: int = 1,
                 hidden_units: int = 32,
                 output_shape: int = 10):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units,
                      kernel_size=3, stride=1, padding=1),
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

### 2.2.1 Why four convolutions?

The `Conv → ReLU → Conv → ReLU → Pool` block is the classic pattern. The first conv learns simple local features (edges, short strokes). The second conv combines those into slightly more complex patterns (corners, curves). The pool operation reduces spatial resolution, which (a) shrinks the tensor so later layers are cheaper, and (b) makes the network somewhat invariant to small translations.

Repeating the block twice gives the network a chance to build up a hierarchy of features. Two blocks is enough for MNIST; deeper networks overfit quickly on such a small dataset.

### 2.2.2 Why 3×3 kernels with padding 1

The output spatial size for a conv with input `N`, kernel `K`, stride `S`, padding `P` is:

```
out = floor((N + 2P - K) / S) + 1
```

Plugging in `K=3`, `S=1`, `P=1`:

```
out = floor((N + 2 - 3) / 1) + 1 = N
```

So each conv preserves the spatial dimensions. This is not arbitrary — it means we do not have to track "did this layer shrink the tensor" as a separate concern from "what did this layer compute." Every conv is a same-size transformation.

### 2.2.3 Why 32 channels

32 is enough to represent the variety of features MNIST needs without being so many that the network overfits or slows down. Doubling to 64 is a plausible experiment (Part VII). Halving to 16 is also plausible.

### 2.2.4 Why two max-pools

Each 2×2 maxpool halves the spatial dimension. Starting at 28×28:

```
28 / 2 = 14
14 / 2 = 7
```

After two pools, we are at 7×7. This is small enough that flattening gives us a manageable 1568-element vector, and small enough that the FC layer does not dominate the parameter count. If you had three pools, you would be at 3×3 and lose too much spatial information for the shapes MNIST digits have.

### 2.2.5 Why the flatten size is 1568

`32 channels × 7 height × 7 width = 1568`. That is what feeds the FC layer.

### 2.2.6 The parameter count

- `conv1_w`: 32 × 1 × 3 × 3 = 288
- `conv1_b`: 32
- `conv2_w`: 32 × 32 × 3 × 3 = 9216
- `conv2_b`: 32
- `conv3_w`: 32 × 32 × 3 × 3 = 9216
- `conv3_b`: 32
- `conv4_w`: 32 × 32 × 3 × 3 = 9216
- `conv4_b`: 32
- `fc_w`: 10 × 1568 = 15680
- `fc_b`: 10

Total: 43,754 floats = 175,016 bytes. That is the size of `weights.bin`.

Do this arithmetic by hand at least once. It is the kind of thing that if you do not know, you will eventually get bitten by — you will change a layer, forget to update the expected file size somewhere, and spend an hour chasing a phantom bug.

### 2.2.7 What can go wrong

- **Change `hidden_units`**: everything downstream changes. `flatten_size`, `fc_w`'s shape, `CnnModel`'s fixed-size array, `weights.bin`'s size. Changing one without the others is a classic source of "the file loaded but predictions are wrong."
- **Remove a conv layer**: `LAYER_KEYS` in `export.py` must match, `model_load` must match, or the C side reads a file that is the wrong size or wrong structure.
- **Reorder layers within a `Sequential`**: `block_1.0` is the first conv, `block_1.1` is the first ReLU, `block_1.2` is the second conv. If you insert a layer, all subsequent indices shift, and `LAYER_KEYS` becomes wrong.

The model is not just the architecture. It is the architecture plus the exact order in which the state_dict's keys come out, plus the exact flattened shapes of every tensor. Any change to the architecture is a change to the export contract, and vice versa.

### 2.2.8 A tiny convolution example before reading the real model

```python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

kernel = torch.tensor([
    [1.0, 0.0],
    [0.0, -1.0],
])

patch = x[0:2, 0:2]
score = (patch * kernel).sum()

print(patch)
print(score)
```

- `x = torch.tensor([...])` — Creates a small 3×3 input.
- `kernel = torch.tensor([...])` — Creates a 2×2 kernel.
- `patch = x[0:2, 0:2]` — Extracts a local 2×2 neighborhood.
- `score = (patch * kernel).sum()` — Multiplies corresponding values and sums them. This is the core operation inside convolution/cross-correlation.

**Predict the score before you run it.** `patch` is `[[1,2],[4,5]]`. `patch * kernel` is `[[1*1, 2*0], [4*0, 5*-1]] = [[1,0],[0,-5]]`. Sum = `1 + 0 + 0 - 5 = -4`. Run it to confirm.

## 2.3 `train.py` — One Epoch, One Step

```python
"""Training and evaluation step functions.

Each function does one full pass over its loader:
    train_step — forward + backward + optimizer update
    test_step  — forward only, in eval mode, no gradients
"""

import torch


def train_step(model, loader, optimizer, loss_fn, device):
    """
    Run one training epoch.

    Returns:
        average training loss over the epoch
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test_step(model, loader, loss_fn, device):
    """
    Run one evaluation pass.

    Returns:
        (average test loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy
```

### 2.3.1 The universal training step pattern

Every gradient-based training loop has this shape:

1. Move the batch to the device (CPU or GPU).
2. Zero the gradients.
3. Forward: compute logits.
4. Compute loss.
5. Backward: compute gradients.
6. Optimizer step: update weights.
7. Accumulate the loss for reporting.

Memorize this. You will see it in every framework in every language, and you will write it yourself if you ever implement training in C (Part XIII).

### 2.3.2 Why `zero_grad()`

PyTorch accumulates gradients by default. Every call to `loss.backward()` *adds to* the existing `.grad` tensors rather than overwriting them. This is deliberate — it is how you do multi-step accumulation (for example, simulating a larger batch than fits in memory). But for a normal training loop, you want each step's gradients to reflect only that step's batch, so you call `optimizer.zero_grad()` at the top of each iteration.

Forgetting `zero_grad()` is one of the most common PyTorch bugs. The symptoms: loss oscillates wildly, then diverges. If your training loss curve looks like a sawtooth that is getting worse, this is the first thing to check.

### 2.3.3 Why `model.eval()` and `torch.no_grad()`

`model.eval()` puts the model into inference mode. For this model, it does nothing (there is no dropout, no batch norm), but it is the correct convention — if you later add dropout, this call becomes essential.

`torch.no_grad()` disables the autograd graph. Without it, every forward pass in the test loop builds a computation graph that is never used, wasting memory and time. For a small model like this, the difference is small; for a big one, it is the difference between running and crashing.

### 2.3.4 Why `argmax(dim=1)`

The logits have shape `(batch, 10)`. `argmax(dim=1)` finds the index of the max along the class dimension, giving a `(batch,)` tensor of predicted classes. `argmax(dim=0)` would compute the max across the batch dimension, which is meaningless here. This is another classic bug — get the dimension wrong, get nonsense.

## 2.4 `evaluate.py` — The Training Loop and Checkpointing

```python
"""Top-level training script.

Runs training for a few epochs and saves the trained model to disk.

The saved file is a state_dict — a dict mapping parameter names to
tensors. It is NOT a full pickled model. That distinction matters:
a state_dict is portable across Python versions and PyTorch
versions, and it's what export.py expects to load.
"""

from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_loaders
from model import _MainModel
from train import train_step, test_step


EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

MODEL_SAVE_PATH = Path("models/number_guesser_model.pth")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    model = _MainModel(
        input_shape=1,
        hidden_units=32,
        output_shape=10,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_step(
            model, train_loader, optimizer, loss_fn, device
        )
        test_loss, test_acc = test_step(
            model, test_loader, loss_fn, device
        )

        print(f"epoch {epoch}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"test_loss={test_loss:.4f}  "
              f"test_acc={test_acc:.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"saved {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
```

### 2.4.1 The one caveat worth flagging

The version of this script above is the *correct* one — five epochs, with `train_step` and `test_step` called inside a loop. An earlier version of the project's `evaluate.py` (mentioned in some of the older guides) ran `train_step` and `test_step` exactly once, producing a real but under-trained model. If your actual `evaluate.py` looks like that, change it to loop, or you will be chasing "why is my model only 97% accurate" when the answer is "it trained for one epoch instead of five."

### 2.4.2 Why `state_dict` and not the whole model

`torch.save(model)` pickles the entire module, including a reference to `_MainModel` by class name. If you ever rename the class, or move it to a different file, or change its `__init__` signature, loading breaks with a confusing error. `state_dict()` is a plain dict of tensors, with no class references at all — much more portable, and much easier for the C side to consume. This is why `export.py` expects a `state_dict` and not a pickled module.

### 2.4.3 Why Adam and why 1e-3

Adam is a reasonable default optimizer: adaptive learning rate per parameter, well-behaved on a wide range of problems. `1e-3` is a reasonable default learning rate for Adam on MNIST. Both are *defaults*, not tuned values — Part VII's experiments will explore alternatives.

### 2.4.4 What can go wrong

- **Saving to a directory that does not exist**: `mkdir(parents=True, exist_ok=True)` handles this.
- **Training for too few epochs**: 5 epochs gets ~99% on MNIST. 1 epoch gets ~97%. 20 epochs might overfit. The number is a choice, not a truth.
- **Running on GPU and then trying to load on CPU**: `torch.load` will attempt to place tensors on the device they were saved from. `export.py` uses `map_location="cpu"` to force them to CPU regardless, which is why the C side never has to worry about this.

## 2.5 `export.py` — Writing `weights.bin`

This is the bridge between Python and C. Everything upstream produces a `state_dict`; everything downstream consumes a raw byte file.

```python
"""Export trained PyTorch weights for the C inference implementation.

Writes weights.bin — a flat sequence of float32 values, in the exact
order that c/src/nn.c's model_load() reads them.

No header. No metadata. Just raw float32 bytes. The file size is the
first and only check on the C side.

The order in LAYER_KEYS is the contract. If you change it here, you
must change model_load's read order in c/src/nn.c to match. There is
no automatic way to detect a mismatch except by seeing wrong
predictions.
"""

from pathlib import Path

import torch

from model import _MainModel


MODEL_PATH = Path("models/number_guesser_model.pth")
OUTPUT_PATH = Path("models/weights.bin")


LAYER_KEYS = [
    "block_1.0.weight",     # conv1 weights
    "block_1.0.bias",       # conv1 bias
    "block_1.2.weight",     # conv2 weights
    "block_1.2.bias",       # conv2 bias
    "block_2.0.weight",     # conv3 weights
    "block_2.0.bias",       # conv3 bias
    "block_2.2.weight",     # conv4 weights
    "block_2.2.bias",       # conv4 bias
    "classifier.1.weight",  # FC weights
    "classifier.1.bias",    # FC bias
]

EXPECTED_BYTES = 175_016


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"{MODEL_PATH} not found. Train a model first:\n"
            f"    python evaluate.py"
        )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    missing = [key for key in LAYER_KEYS if key not in state_dict]
    if missing:
        raise SystemExit(
            f"state_dict is missing expected keys:\n"
            f"  {missing}\n"
            f"Actual keys:\n"
            f"  {list(state_dict.keys())}\n"
            f"model.py's architecture may have changed. Update LAYER_KEYS "
            f"and c/src/nn.c's model_load() to match."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        for key in LAYER_KEYS:
            tensor = state_dict[key]
            f.write(tensor.contiguous().numpy().tobytes())

    actual_bytes = OUTPUT_PATH.stat().st_size
    if actual_bytes == EXPECTED_BYTES:
        print(f"wrote {OUTPUT_PATH} ({actual_bytes} bytes) [OK]")
    else:
        print(f"wrote {OUTPUT_PATH} ({actual_bytes} bytes) "
              f"[MISMATCH — expected {EXPECTED_BYTES}]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

### 2.5.1 The contract, spelled out

There are three things that must agree between this file and `model_load`:

1. **Order.** `LAYER_KEYS` lists the keys in the sequence the C side reads them. Changing the list changes the file format. There is no way for `model_load` to detect a change here — it reads the same number of floats either way — so the failure mode is silently wrong predictions.
2. **Layout.** Each tensor is written with `.contiguous()`, which is a no-op for fresh tensors but documents the invariant that C-contiguous row-major is the expected layout. If PyTorch ever changed its default (it will not) or a `state_dict` were saved with non-contiguous tensors (it is not, for this model), the `.contiguous()` call would fix it. This is defensive, not necessary.
3. **Dtype.** Each tensor is `float32`. If any were `float64`, the byte count would double and the size check would fail with a clear error. This is a good failure mode.

### 2.5.2 Why no header

See §1.3 for the full reasoning. The short version: `export.py` and `model_load` already agree on the format, in code. A header would communicate information that neither side is uncertain about.

The cost of this design: no version tag, no dtype tag, no tensor names. Which is fine as long as `export.py` and `model_load` are edited together, and dangerous the moment they are not.

### 2.5.3 When to expect this to change

The moment you have more than one model (single-digit, two-digit, OCR), you need a way to distinguish them. The moment you want to support multiple dtypes, you need a way to indicate which one a file uses. Part XII lays out the "v2 format" design.

For now, one model, one format, one contract.

## 2.6 `helper_functions.py` — Utilities

Depending on what the repository has, this file might contain:

```python
"""Utility functions shared across the Python pipeline."""

import torch
from pathlib import Path


def save_tensor_as_bin(tensor: torch.Tensor, path: Path) -> None:
    """Write a single float32 tensor to a raw binary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(tensor.contiguous().cpu().numpy().tobytes())


def load_tensor_from_bin(path: Path, shape) -> torch.Tensor:
    """Read a raw float32 binary file into a tensor of the given shape."""
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return torch.from_numpy(raw).reshape(*shape)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

These utilities get used by `dump_intermediate.py`, `make_debug_input.py`, and the comparison tools.

## 2.7 The Full End-to-End Training Recipe

```bash
# From repository root
python python/train.py           # or python evaluate.py — depends on file layout
python python/evaluate.py
python python/export.py
stat -c%s models/weights.bin     # should print 175016
```

If everything worked, you now have a `models/weights.bin` that is exactly 175016 bytes.

**Do not proceed until this works.** A wrong-sized file means the C side will refuse to load it, and the whole project stalls. Get this step right first.

---

# Part III — The C Inference Engine

The C side of this project is where the arithmetic lives. Every operation here corresponds to an operation in PyTorch, and everything here must match PyTorch's output within numerical tolerance. The parity verification (Part IV) is what proves this.

C is deliberately uncomfortable. There is no autograd. There is no `Tensor` type with built-in shape checking. There is no `model.to(device)`. There is a flat array, three integers, and every nested loop written out by hand. This discomfort is the point: it forces you to know exactly what each operation costs, in arithmetic and in bytes.

## 3.1 `nn.h` — The Header

```c
#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * nn.h — Neural network inference API for the Number Guesser project.
 *
 * This header declares:
 *   - the Tensor type (a runtime-sized, heap-allocated float buffer)
 *   - the CnnModel type (fixed-size arrays matching the trained model)
 *   - every op used in the forward pass
 *
 * Layout contract:
 *   Tensors are channel-first, row-major, C-contiguous:
 *     offset(c, y, x) = (c * height + y) * width + x
 *   This matches PyTorch's default [C, H, W] layout. If you change
 *   it here, you must change it in export.py and model_load.
 *
 * Ownership contract:
 *   - Functions that RETURN a Tensor allocate it. Caller frees.
 *   - Functions that take a Tensor* and return void either mutate
 *     in place or read only.
 */


typedef struct {
    float *data;      /* heap-allocated, owns this memory */
    int channels;
    int height;
    int width;
} Tensor;


typedef struct {
    float conv1_w[32 * 1  * 3 * 3];   /* 288 */
    float conv1_b[32];
    float conv2_w[32 * 32 * 3 * 3];   /* 9216 */
    float conv2_b[32];
    float conv3_w[32 * 32 * 3 * 3];   /* 9216 */
    float conv3_b[32];
    float conv4_w[32 * 32 * 3 * 3];   /* 9216 */
    float conv4_b[32];
    float fc_w[10 * 1568];            /* 15680 */
    float fc_b[10];
} CnnModel;


Tensor tensor_alloc(int channels, int height, int width);
void   tensor_free(Tensor *t);
float  tensor_get(const Tensor *t, int c, int y, int x);
void   tensor_set(Tensor *t, int c, int y, int x, float value);
void   tensor_print_summary(const Tensor *t, const char *label);

void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);

void relu(float *x, int n);
void relu_tensor(Tensor *t);

int  argmax(const float *x, int n);

Tensor conv2d(const Tensor *input, const float *weights, const float *bias,
              int out_channels, int k, int stride, int pad);

Tensor maxpool2d(const Tensor *input, int k, int stride);

int  model_load(CnnModel *m, const char *path);
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out);

#endif
```

### 3.1.1 Reading the header as documentation

The header has three sections, and each has a specific job.

**The `Tensor` type** — a pointer plus three integers. The pointer points to heap memory whose lifetime is managed by convention (`tensor_alloc` gives it, `tensor_free` takes it back).

**The `CnnModel` type** — a bundle of fixed-size arrays. No pointers, no `malloc`. The size of the struct is known at compile time, and `sizeof(CnnModel)` is the byte count of `weights.bin`. This is not just convenient; it is what makes the file-size check in `model_load` work.

**Function declarations** — the API. `linear`, `relu`, `conv2d`, etc. Each one has a documented ownership rule: `conv2d` returns a `Tensor` (allocate-and-return), `relu` takes an array and a length (mutate-in-place), etc.

Notice what is *not* in the header:

- No `softmax`. Softmax is applied in the UI layer or a helper, not in the inference engine, because the engine's job is to produce logits, and softmax is a separate concern.
- No `BatchNorm`, no `Dropout`, no anything stateful. The model has none, so neither does the header.
- No error type. `model_load` returns `int`; everything else aborts on failure. This is fine for a small project but would need to change if the code ever became a library (Part XII).

### 3.1.2 What can go wrong

- **Forgetting the include guard**: duplicate type definitions, compile errors.
- **A typo in an array size**: `sizeof(CnnModel)` changes, the size check in `model_load` fails, and the file will not load. Actually a *good* failure mode — the error is loud.
- **Changing the header's types without changing the .c file**: usually a compile error, occasionally (with more subtle mismatches) undefined behavior. Be careful when editing.

## 3.2 Tensor Operations — Alloc, Free, Get, Set

The core of the tensor abstraction:

```c
Tensor tensor_alloc(int channels, int height, int width) {
    Tensor t;
    t.channels = channels;
    t.height = height;
    t.width = width;

    size_t n = (size_t)channels * (size_t)height * (size_t)width;

    t.data = calloc(n, sizeof(float));
    if (t.data == NULL) {
        fprintf(stderr,
                "tensor_alloc: calloc failed for %d x %d x %d\n",
                channels, height, width);
        exit(EXIT_FAILURE);
    }
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = 0;
    t->height = 0;
    t->width = 0;
}

float tensor_get(const Tensor *t, int c, int y, int x) {
    size_t index =
        ((size_t)c * (size_t)t->height + (size_t)y) *
        (size_t)t->width + (size_t)x;
    return t->data[index];
}

void tensor_set(Tensor *t, int c, int y, int x, float value) {
    size_t index =
        ((size_t)c * (size_t)t->height + (size_t)y) *
        (size_t)t->width + (size_t)x;
    t->data[index] = value;
}
```

### 3.2.1 Line by line

**`size_t n = (size_t)channels * (size_t)height * (size_t)width;`** — compute the total element count. The casts to `size_t` are what keep this multiplication from overflowing `int` if the dimensions ever get large. On a 64-bit platform, `size_t` is 64 bits, and the product is safe up to about `1.8 × 10^19` — vastly more than any tensor in this project.

**`t.data = calloc(n, sizeof(float));`** — `calloc` allocates and zero-initializes. The zero-initialization is a deliberate choice over `malloc`: if there is a bug where a tensor cell is read before it is written, `malloc` gives you garbage floats (which might be NaN, might be huge, might be small — hard to spot), while `calloc` gives you a suspicious `0.0`. That is easier to recognize as a bug when you see it in a debugger or a print statement.

**`exit(EXIT_FAILURE)`** — on allocation failure, we abort. In a larger program, you would want to return an error code and let the caller decide. In this project, allocation failure is never expected, so aborting is fine.

**`free(t->data); t->data = NULL;`** — the defensive null-set. After `free`, the pointer is dangling: it still holds the old address, but that address is no longer valid. Setting it to `NULL` means that any accidental use-after-free immediately segfaults, instead of silently reading whatever the allocator has since reused the memory for. That is a strictly better failure mode.

**`t->channels = t->height = t->width = 0;`** — same defensive reasoning for the shape metadata. After freeing, the tensor has no meaningful shape. Setting it to zero makes accidental reuse of a freed tensor produce obviously-wrong behavior (all shapes zero) rather than subtly-wrong behavior (shape says 32×28×28, but data is garbage).

**`tensor_get` and `tensor_set`** — the one place in the entire codebase that computes the flat offset. Every other function reads or writes through these two functions. That is important: if there is ever a layout bug, it is here, and only here.

### 3.2.2 What can go wrong

- **Wrong formula**: writing `(c * H + y) * (W + x)` instead of `(c * H + y) * W + x` shifts every element by `x` whole rows, silently corrupting adjacent cells. This is caught by "neighbor unaffected" tests.
- **Forgetting the `size_t` casts**: silent integer overflow on large tensors. Not a risk for the current shapes, but a landmine for future ones.
- **Forgetting to check `calloc`'s return**: NULL dereference on the next access. The `exit` on failure handles this.
- **Use-after-free**: reading `t->data` after `tensor_free`. The NULL-set makes this crash immediately.

## 3.3 Linear, ReLU, Argmax

The three "simple" ops:

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
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}


void relu_tensor(Tensor *t) {
    int n = t->channels * t->height * t->width;
    relu(t->data, n);
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

### 3.3.1 Linear, line by line

**`for (int o = 0; o < out_features; o++)`** — iterate over output neurons. Each output is one dot product.

**`float sum = b[o];`** — start with the bias. This is slightly cheaper than starting at zero and adding the bias at the end (one fewer addition per neuron), but the more important reason is that it matches the mathematical definition: `y = Wx + b`.

**`const float *row = W + o * in_features;`** — pointer arithmetic to get the start of row `o`. Since `W` has shape `(out_features, in_features)` and is stored row-major, row `o` is the `in_features` floats starting at offset `o * in_features`.

**`for (int i = 0; i < in_features; i++) sum += row[i] * x[i];`** — the dot product. Both `row` and `x` are read sequentially, which is cache-friendly.

**`y[o] = sum;`** — store.

### 3.3.2 ReLU, line by line

**`if (x[i] < 0.0f) { x[i] = 0.0f; }`** — clamp to zero. Note the `<` rather than `<=`: a value of exactly 0 stays 0, which is the correct ReLU behavior (`ReLU(0) = 0`), but not a write, which is slightly more efficient and avoids touching already-correct memory.

### 3.3.3 relu_tensor, line by line

**`int n = t->channels * t->height * t->width;`** — ReLU does not care about shape, only about the total number of elements. This is what makes `relu_tensor` trivial to implement on top of `relu`.

### 3.3.4 Argmax, line by line

**`int best_idx = 0; float best_val = x[0];`** — initialize to the first element rather than `-infinity`. This saves a comparison in the loop (we start at index 1).

**`if (x[i] > best_val)`** — uses strict `>` so that ties go to the earliest index. This matches PyTorch's default `argmax` behavior.

### 3.3.5 What can go wrong

- **`linear` with a transposed `W`**: if you pass `W` as `(in_features, out_features)` instead of `(out_features, in_features)`, the sums are wrong. The test suite checks this with a hand-computed case.
- **`argmax` on an empty array**: reads `x[0]` out of bounds. Not a risk here (logits always has 10 elements), but a landmine if you ever generalize the function.

## 3.4 Conv2D — The Heart

```c
Tensor conv2d(const Tensor *input,
              const float *weights,
              const float *bias,
              int out_channels,
              int k, int stride, int pad) {

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

                            int w_idx =
                                ((oc * input->channels + ic) * k + ky) * k + kx;

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

### 3.4.1 The math

Output spatial size:

```
out = floor((N + 2P - K) / S) + 1
```

For `K=3, S=1, P=1`:

```
out = floor((N + 2 - 3) / 1) + 1 = N
```

So spatial dimensions are preserved. This is *the* property that makes the "same-shape block" architecture easy to reason about.

For each output `(oc, oy, ox)`:

```
sum = bias[oc]
for ic in 0..C_in:
    for ky in 0..k:
        for kx in 0..k:
            iy = oy*S - P + ky
            ix = ox*S - P + kx
            if (iy, ix) inside input:
                sum += input[ic][iy][ix] * weights[oc][ic][ky][kx]
output[oc][oy][ox] = sum
```

### 3.4.2 The index arithmetic, read inside-out

**`int iy = oy * stride - pad + ky;`**

- `oy * stride` — walks across the input in output-sized jumps (with stride 1, this is a 1-to-1 walk; with stride 2, every other input row).
- `- pad` — shifts into "padded coordinates." With padding 1, the leftmost output pixel (at `ox=0`) corresponds to input column `-1`, which is outside the actual input and treated as zero.
- `+ ky` — picks the tap within the kernel.

**`int w_idx = ((oc * input->channels + ic) * k + ky) * k + kx;`**

This is the flattened form of a 4D array indexed `[oc][ic][ky][kx]`. Read inside-out:

- `oc * input->channels + ic` — which (output channel, input channel) pair.
- `* k + ky` — row within that pair's kernel.
- `* k + kx` — column.

This exactly matches PyTorch's `[out_ch, in_ch, kh, kw]` layout.

### 3.4.3 Why the bounds check rather than materializing a padded tensor

You could, in principle, allocate a padded copy of the input (add a 1-pixel border of zeros) and then run the convolution with no bounds check. This is what `torch.nn.functional.pad` does in the reference implementation. It is cleaner code but uses more memory (the padded tensor is larger) and more time (the copy is O(N²) with a padding factor).

The bounds-check approach avoids the copy: out-of-range taps simply contribute zero, which is exactly what zero-padding means. It is slightly more code inside the inner loop, but it is the standard approach for inference-time convolution.

### 3.4.4 Cost analysis

For a conv with `C_out` outputs, `H × W` output pixels, `C_in` inputs, kernel `k`:

```
ops ≈ C_out · H · W · C_in · k²
```

For `conv2` (32→32, 28×28, k=3):

```
32 · 28 · 28 · 32 · 9 = 7,225,344 multiply-adds
```

For `conv1` (1→32, 28×28, k=3):

```
32 · 28 · 28 · 1 · 9 = 225,792
```

Conv2 is 32× more expensive than conv1 because it has 32 input channels. In a bigger network, the later convs dominate.

This is why profiling matters (Part IX): the intuition "later layers are slower" is right, but the *ratio* is worth measuring rather than assuming.

### 3.4.5 What can go wrong

- **Wrong padding/stride in the output shape**: subsequent layers crash or produce wrong shapes.
- **Transposed weight index**: sums are wrong, predictions silently incorrect. Caught by parity verification.
- **Using `pad` where you meant `-pad`**: taps land in the wrong place. Caught by hand-computed tests.
- **Forgetting to free `out` in the caller**: leak. Caught by sanitizers.

## 3.5 MaxPool2D

```c
Tensor maxpool2d(const Tensor *input, int k, int stride) {
    int out_h = (input->height - k) / stride + 1;
    int out_w = (input->width  - k) / stride + 1;

    Tensor out = tensor_alloc(input->channels, out_h, out_w);

    for (int c = 0; c < input->channels; c++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {

                float best = -INFINITY;

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

### 3.5.1 The math

Pooling output size:

```
out = floor((N - K) / S) + 1
```

No padding term because pooling has no padding in this project. For `k=2, stride=2`:

- 28 → `(28-2)/2 + 1 = 14`
- 14 → `(14-2)/2 + 1 = 7`

### 3.5.2 The critical detail: `-INFINITY`

Initialize the running maximum to `-INFINITY`, not to `0.0f`.

Why: consider a 2×2 window with all-negative values, e.g. `[-3, -5; -2, -7]`. The maximum is `-2`. If you initialize `best = 0.0f`, then the loop's `if (v > best)` never fires (since all `v` are negative), and `best` stays at `0.0f`. Wrong answer.

This bug is *extremely* easy to write, extremely easy to miss, and would only show up when the network's activations go negative — which they do, all the time, in the layers *before* the ReLU. Initializing to `-INFINITY` is the fix, and it costs nothing.

### 3.5.3 Cost analysis

Pooling is cheap: for each output pixel, it reads `k²` values and does `k²` comparisons. Total work is `C · H_out · W_out · k²`, but the constant factor is one comparison per tap rather than one multiply-add. For a 2×2 pool over a 32×28×28 tensor, that is `32 · 14 · 14 · 4 = 250,880` comparisons — negligible compared to the conv layers.

### 3.5.4 What can go wrong

- **Using `0.0f` as the sentinel**: silently wrong on all-negative windows.
- **Wrong stride**: output shape wrong, subsequent layers crash.
- **Bounds check missing**: if there were padding, `iy` could go out of range. Not a risk here since pooling has no padding.

## 3.6 Model Loading

Already covered in §1.3. The key point: `model_load` reads the file in the exact order `export.py` wrote it, checks the file size first, and returns an error code on failure.

## 3.7 The Full Forward Pass

Already covered in §1.1.2. The key point: ownership. Every tensor is allocated, used, freed. Peak memory is two tensors at a time.

## 3.8 Ownership Discipline

The `model_forward` function is the single best illustration of the project's ownership rule. Look at it again:

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

Read it as three rules:

1. **Allocate** (via a function that returns a `Tensor`).
2. **Use** (via functions that take `&tensor` and mutate or read).
3. **Free** (`tensor_free(&tensor)`) — immediately after the last use.

The line `tensor_free(&a);` right after `Tensor b = conv2d(&a, ...)` is the pattern: free `a` the moment `b` no longer depends on it. This keeps peak memory usage at two tensors, and it makes the code linear in the number of layers rather than quadratic.

The discipline is not natural at first. It becomes natural after writing a few of these functions. The reward is that C stops feeling dangerous.

## 3.9 `ui.h` / `ui.c` — Canvas Logic

The canvas is the interface between the user's mouse and the model's input tensor. It is a 280×280 buffer of `[0, 1]` floats, and a small set of functions for drawing, clearing, and downsampling.

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

### 3.9.1 Why 280×280

The canvas is 280×280 because `280 = 28 × 10`, so downsampling to 28×28 is an exact 10:1 box-filter operation, with no rounding or fractional blocks. This is not just convenient; it makes the downsampling deterministic and easy to reason about.

### 3.9.2 Why `float pixels[CANVAS_SIZE * CANVAS_SIZE]`

Fixed-size array inside a struct, not a pointer. The canvas size is a compile-time constant, so no allocation is needed, and the struct can live on the stack. This is the same pattern as `CnnModel` — fixed size when known at compile time.

### 3.9.3 The clear function

```c
void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
    app->has_prediction = 0;
    memset(app->probs, 0, sizeof(app->probs));
    app->is_drawing = 0;
}
```

`memset` to zero works because `0.0f` is represented as all-zero bytes in IEEE-754. This is a *specific* property of floating point: `0.0f` is all zeros, but `1.0f` is *not* all ones, so setting to any other value would need a loop. The memset is fast and correct *because* we are setting to zero.

### 3.9.4 The brush

```c
static void draw_circle_brush(AppState *app,
                              float cx, float cy,
                              float radius, float strength) {
    int r = (int)(radius + 1);
    int center_x = (int)cx;
    int center_y = (int)cy;

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = center_x + dx;
            int y = center_y + dy;

            if (x < 0 || x >= CANVAS_SIZE) continue;
            if (y < 0 || y >= CANVAS_SIZE) continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > radius) continue;

            float falloff = 1.0f - (dist * dist) / (radius * radius);
            float val = strength * falloff;

            float *pixel = &app->pixels[y * CANVAS_SIZE + x];
            *pixel = fmaxf(*pixel, val);
        }
    }
}
```

The brush is a soft-edged circle. At the center (`dist = 0`), the falloff is 1.0, so `val = strength`. At the edge (`dist = radius`), the falloff is 0.0, so `val = 0`. In between, it is a quadratic ramp.

**`*pixel = fmaxf(*pixel, val);`** — take the max with the existing value. This means overlapping brush strokes do not add up (which would saturate the pixel to white); instead, the pixel takes the *brightest* contribution from any stroke that touched it. This matches how a physical pen behaves.

### 3.9.5 The edge case

The bounds check `if (x < 0 || x >= CANVAS_SIZE) continue;` is what prevents an out-of-bounds write when the brush center is near a canvas edge. This is the exact kind of bug that is easy to write and easy to miss: without the check, the loop would write to `pixels[y * 280 + x]` for `x` outside `[0, 279]`, corrupting adjacent memory.

The test suite includes a specific "draw at edge" test that runs under AddressSanitizer to confirm this works.

### 3.9.6 The line

```c
void canvas_draw_line(AppState *app,
                      float x1, float y1,
                      float x2, float y2) {
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
        float x = x1 + dx * t;
        float y = y1 + dy * t;
        draw_circle_brush(app, x, y, BRUSH_RADIUS * 0.8f, BRUSH_STRENGTH);
    }
}
```

The line function is a "sampled brush" — it draws circles at intervals along the segment, spaced closely enough that they overlap into a smooth stroke. The `1.5×` density ensures no gaps.

### 3.9.7 The downsampling

```c
void canvas_to_mnist_input(const AppState *app, float *out28x28) {
    /* find bounding box of non-zero pixels */
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

    /* Empty canvas -> all-zero 28×28 */
    if (max_x < 0 || max_y < 0) return;

    /* compute bounding box, expand to square, add margin, crop and resize */
    /* ... */
}
```

The downsampling function is more interesting than it looks. It:

1. Finds the bounding box of non-background pixels.
2. Expands the box to a square (so a "1" and a "0" get the same treatment).
3. Adds a 10% margin around the digit.
4. Crops the canvas to this box.
5. Resizes the cropped region to 20×20 using bilinear interpolation.
6. Centers the 20×20 image in a 28×28 frame with 4-pixel padding.

Steps 5 and 6 match MNIST's own preprocessing: MNIST digits are 20×20 images centered in a 28×28 frame. Doing the same here reduces the distribution mismatch between training and inference.

**This is the most important part of the preprocessing**, and it is the reason why the model works on real drawings. Without the bounding box, a small drawing in the corner of the canvas would appear as a tiny dark blob in the 28×28 input, and the model would fail. With the bounding box, the drawing is scaled up and centered, matching what the model was trained to see.

The design-level match between Python training and C inference is documented in §3.11. It is important to understand this as a *design* choice, not a guarantee — real drawings may still have stroke-width statistics that differ from MNIST, which is why the evaluation milestone (Part VI) is separate from the parity verification milestone (Part IV).

## 3.10 `main.c` — The Raylib Application

The Raylib application is the user-facing entry point. It:

- Opens a window.
- Loads the model.
- Runs the event loop: mouse input, drawing, button clicks.
- Runs inference on demand.
- Displays results (probability bars, prediction, confidence).

The full `main.c` is long; the important parts:

### 3.10.1 The prediction function

```c
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
```

This function is the bridge between the UI and the model. It:

1. Downsample the canvas to 28×28 (using `canvas_to_mnist_input`).
2. Copies the 784 floats into a `Tensor`.
3. Runs the forward pass.
4. Frees the input tensor.
5. Computes softmax over the logits.
6. Updates the app state with the prediction and confidence.

Notice the `tensor_alloc` / `model_forward` / `tensor_free` pattern: allocate, use, free. The `input` tensor is owned by `run_prediction` and freed by it.

### 3.10.2 The event loop

```c
while (!WindowShouldClose()) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        /* buttons */
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) &&
        CheckCollisionPointRec(mouse, canvas_rect)) {
        /* drawing */
    }

    BeginDrawing();
    /* ... render everything ... */
    EndDrawing();
}
```

The event loop is standard Raylib: check input, update state, render, repeat. `IsMouseButtonPressed` fires once (for buttons); `IsMouseButtonDown` fires every frame while held (for drawing). This distinction matters — using `Pressed` for drawing would produce one dot per click, not a stroke.

### 3.10.3 The softmax

```c
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
```

The `- max_val` is the numerical stability trick: without it, `exp(large_number)` overflows. Since softmax is invariant to adding a constant to all logits, subtracting the max preserves the result while preventing overflow.

## 3.11 Preprocessing as a First-Class System Component

Preprocessing is not an afterthought. It is the interface between two distributions:

- **Training distribution**: 28×28 grayscale, centered, specific stroke-width statistics (from MNIST).
- **Inference distribution**: 280×280 canvas, arbitrary position/size, drawn with a mouse.

The closer the inference distribution is to the training distribution (after preprocessing), the better the model performs. Any mismatch — different stroke widths, different centering, different scale — degrades accuracy.

This is why preprocessing deserves its own module and its own tests, and its own experiments (Part VII).

### 3.11.1 The preprocessing pipeline, point by point

Training-side:

```python
transform = transforms.ToTensor()
```

That is it. Converts to float, scales `[0, 255]` to `[0, 1]`, arranges as `(1, 28, 28)`.

Inference-side:

1. Canvas starts as 280×280 `[0, 1]` grayscale (already in the right range).
2. Find bounding box of non-background pixels.
3. Expand to square, add margin.
4. Crop to the box.
5. Resize to 20×20 (bilinear).
6. Center in 28×28 with 4-pixel padding.

Comparing:

| Aspect | Training | Inference | Match? |
|---|---|---|---|
| Grayscale | Yes | Yes | ✓ |
| Range | `[0, 1]` | `[0, 1]` | ✓ |
| Normalization | None | None | ✓ |
| Inversion | No (white on black) | No (white on black) | ✓ |
| Digit size | 20×20 centered in 28×28 | 20×20 centered in 28×28 | ✓ |
| Stroke width | MNIST stylus | Mouse brush | ✗ (potential mismatch) |

The first five rows are *design matches* — the code is written to make them match. The last row is a *potential mismatch* — mouse strokes and stylus strokes have different widths, and this can degrade accuracy even with everything else matching.

The evaluation milestone (Part VI) is where you measure whether this mismatch is a problem in practice.

---

# Part IV — Verification

## 4.1 Why "It Compiles and Runs" Is Not Verification

The single most important sentence in this book:

> **Compiling and running without crashing proves that the code has no memory errors. It does not prove the code is correct.**

You can write a `conv2d` that has the weight index transposed, and it will compile cleanly, pass sanitizers, run in the UI, and produce predictions. Those predictions will be wrong — but not so wrong that they look like a crash. They will just be subtly, plausibly wrong, in a way that is easy to miss.

This is the class of bug that verification exists to catch.

**Verification** means: running the same input through both PyTorch and C, and comparing the outputs at every stage, within a numerical tolerance, and confirming that the differences are small enough to attribute to floating-point rounding rather than to a bug.

## 4.2 `dump_intermediate.py` — The PyTorch Reference Dump

```python
"""Run one image through PyTorch, dump every intermediate tensor.

Output format (one line per stage):
    <label>   shape=(...)  first 5=[...]

The C side's verify.c produces the SAME format, so you can diff the
two outputs directly.
"""

import numpy as np
import torch
from pathlib import Path

from model import _MainModel


MODEL_PATH = Path("models/number_guesser_model.pth")
INPUT_PATH = Path("models/debug_input.bin")


def dump(label: str, t: torch.Tensor) -> None:
    flat = t.detach().flatten().cpu().numpy()
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

        a = model.block_1[0](x)[0];  dump("conv1", a)
        a = model.block_1[1](a);      dump("relu1", a)
        a = model.block_1[2](a);      dump("conv2", a)
        a = model.block_1[3](a);      dump("relu2", a)
        a = model.block_1[4](a);      dump("pool1", a)

        a = model.block_2[0](a);      dump("conv3", a)
        a = model.block_2[1](a);      dump("relu3", a)
        a = model.block_2[2](a);      dump("conv4", a)
        a = model.block_2[3](a);      dump("relu4", a)
        a = model.block_2[4](a);      dump("pool2", a)

        flat = a.flatten()
        dump("flat", flat)

        logits = model.classifier[1](flat.unsqueeze(0))[0]
        dump("logits", logits)

        print(f"\npredicted digit: {logits.argmax().item()}")


if __name__ == "__main__":
    main()
```

### 4.2.1 Why dump at every stage

The obvious thing to do is compare the final prediction: "PyTorch says 7, C says 7, we are good." That is not verification. Two implementations can agree on the final prediction while disagreeing on every intermediate value — the prediction is a *lossy* summary of the computation.

By dumping every intermediate tensor, you can compare stage by stage and find the *first* stage where the two implementations diverge. That is where the bug is. Everything upstream of the first divergence is correct; the bug is in the layer that produces the first divergence, or in the layer right before it (via bad input).

This is the single most important debugging technique in the entire project.

### 4.2.2 Why "first 5 values"

You cannot diff the full tensors by eye — they have thousands of elements. The "first 5 values, rounded to 4 decimals" gives a compact summary that will differ if the tensors differ. For a more rigorous comparison, you would want max absolute error and mean absolute error, but for the initial "do these agree roughly" check, first-5 is fine.

For a stronger comparison, see §4.5.

### 4.2.3 Why a fixed input

The input `debug_input.bin` is a fixed 28×28 float file that both sides read. This is essential for the comparison: if PyTorch and C read different inputs, they will produce different outputs, and you cannot tell whether the difference is due to the input or the computation.

Generating this file is part of the workflow: pick an image, save it as raw floats, use it for both runs.

## 4.3 `verify.c` — The C Reference Dump

```c
#include "../include/nn.h"
#include <stdio.h>


static void dump(const Tensor *t, const char *label) {
    printf("%-8s shape=(%d, %d, %d)  first 5=[",
           label, t->channels, t->height, t->width);

    int n = t->channels * t->height * t->width;
    int show = n < 5 ? n : 5;

    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], i == show - 1 ? "" : ", ");
    }
    printf("]\n");
}


int main(int argc, char **argv) {
    const char *weights_path =
        argc > 1 ? argv[1] : "../models/weights.bin";
    const char *input_path =
        argc > 2 ? argv[2] : "../models/debug_input.bin";

    CnnModel m;
    if (model_load(&m, weights_path) != 0) {
        return 1;
    }

    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(input_path, "rb");
    if (f != NULL) {
        fread(input.data, sizeof(float), 28 * 28, f);
        fclose(f);
    } else {
        fprintf(stderr,
                "verify: could not open %s, using zero image\n",
                input_path);
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
    linear(m.fc_w, m.fc_b, p2.data, logits,
           p2.channels * p2.height * p2.width, 10);
    tensor_free(&p2);

    printf("logits   shape=(1, 10)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           logits[0], logits[1], logits[2], logits[3], logits[4]);

    printf("\npredicted digit: %d\n", argmax(logits, 10));
    return 0;
}
```

### 4.3.1 Why identical format

The whole point is to be able to diff the two outputs. `verify.c`'s `dump` matches `dump_intermediate.py`'s format: same label padding, same shape formatting, same rounding to 4 decimals.

This is a small thing, but it saves a *lot* of time. You run both, `diff py_stages.txt c_stages.txt`, and you see immediately which lines differ.

### 4.3.2 Compile and run

```bash
cd c/tools
gcc -Wall -Wextra -std=c11 -I../include verify.c ../src/nn.c -o verify -lm
./verify ../models/weights.bin ../models/debug_input.bin > ../../notes/c_stages.txt
```

And on the Python side:

```bash
python dump_intermediate.py > notes/py_stages.txt
```

## 4.4 Comparing Layers — The Workflow

The comparison script:

```python
"""Compare PyTorch and C stage dumps line-by-line.

Both files have lines of the form:
    <label>  shape=(...)  first 5=[...]

We parse shape and first 5 values, then compute max abs diff.
"""

import re
import sys
from pathlib import Path


LINE_RE = re.compile(
    r"^(\w+)\s+shape=\(([^)]+)\)\s+first 5=\[([^\]]+)\]"
)


def parse(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        label = m.group(1)
        shape = tuple(int(x.strip()) for x in m.group(2).split(",") if x.strip())
        vals = [float(x.strip()) for x in m.group(3).split(",") if x.strip()]
        result[label] = (shape, vals)
    return result


def main():
    TOL = 1e-4

    py = parse("notes/pytorch_stages.txt")
    c = parse("notes/c_stages.txt")

    ok = True
    for label in py:
        if label not in c:
            print(f"MISSING in C: {label}")
            ok = False
            continue

        py_shape, py_vals = py[label]
        c_shape, c_vals = c[label]

        if py_shape != c_shape:
            print(f"SHAPE MISMATCH {label}: py={py_shape} c={c_shape}")
            ok = False
            continue

        diffs = [abs(a - b) for a, b in zip(py_vals, c_vals)]
        max_diff = max(diffs)
        status = "PASS" if max_diff < TOL else "FAIL"
        if max_diff >= TOL:
            ok = False

        print(f"{label:8s} shape={py_shape}  max_diff={max_diff:.2e}  {status}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

### 4.4.1 Reading the output

**All PASS**: the C implementation matches PyTorch, within tolerance, at every stage. This is what you want to see. It means: the weights loaded correctly, the forward pass computes the right thing, and the two implementations agree.

**First FAIL at `input`**: `debug_input.bin` is being read differently on the two sides. Check the file size (should be 784 floats × 4 bytes = 3136 bytes) and the dtype (float32).

**First FAIL at `conv1`**: the conv1 weight order, kernel indexing, or padding logic is wrong. Check `export.py`'s `LAYER_KEYS` against `model_load`'s read order. Check `conv2d`'s `w_idx` computation.

**First FAIL at `pool1`**: the max sentinel or window position is wrong. Verify `-INFINITY` initialization.

**First FAIL at `logits`**: the flatten order or the FC weight layout is wrong. Verify that `fc_w` is being read as `(10, 1568)` and not `(1568, 10)`.

### 4.4.2 The underlying logic

The first divergence is the bug. Everything upstream of it is correct; the divergence is either:

- The layer that produces the divergence (its math is wrong), or
- The layer right before it (its output is subtly wrong, and the current layer amplifies the error).

By looking at the *first* divergence, you narrow the search to a small handful of functions.

## 4.5 Numerical Tolerance — What "Same" Means

Floating-point arithmetic is not exact. PyTorch and C, computing "the same" dot product, will produce *slightly* different results, because:

- PyTorch may vectorize (compute four multiplications at once) and sum in a different order than C's left-to-right loop.
- PyTorch may use FMA (fused multiply-add) instructions, which compute `a*b + c` with a single rounding instead of two.
- The compiler may reorder operations.

For a sum of `N` terms, the accumulated rounding error is on the order of `N · ε · max|term|`, where `ε ≈ 1.2 × 10^-7` for float32. For a dot product of 1568 terms, this is roughly `1.9 × 10^-4 · max|term|`.

This is why we use a tolerance of `1e-4` on absolute difference. It is loose enough to accommodate legitimate rounding differences, and tight enough to catch actual bugs.

**What tolerance means in practice**: if `max_diff < 1e-4` at every stage, and the differences are all positive and roughly uniform, you are fine. If `max_diff` is around `1e-3` or larger at any stage, that is a bug, not rounding.

**A different tolerance for different layers**: in principle, the tolerance should scale with the magnitude of the values. The first layer's outputs might be around 1.0; a deep layer's outputs might be around 10.0. A relative tolerance (`|a-b| / max(|a|, |b|)`) would be more principled. For this project, absolute tolerance with a fixed threshold works because all the intermediate values stay in a bounded range.

## 4.6 Real-Weight Verification — The Milestone That Matters

The workflow above works. Now the hard part: running it on the *real* trained weights.

```bash
# 1. On the Python side, produce weights.bin from the trained model
cd python
python export.py

# 2. Confirm the file size
ls -l models/weights.bin
# Should be 175016

# 3. Produce a fixed input file (pick any MNIST digit, save as raw floats)
#    This is a one-time setup step; you can write a small script for it.

# 4. Run the PyTorch dump
python dump_intermediate.py > ../notes/py_stages.txt

# 5. Run the C dump
cd ../c/tools
gcc -Wall -Wextra -std=c11 -I../include verify.c ../src/nn.c -o verify -lm
./verify ../models/weights.bin ../models/debug_input.bin > ../../notes/c_stages.txt

# 6. Compare
cd ../..
python tools/compare.py
```

### 4.6.1 What to expect

If everything is correct, you should see something like:

```
input    shape=(1, 28, 28)  max_diff=0.00e+00  PASS
conv1    shape=(32, 28, 28)  max_diff=2.1e-06  PASS
relu1    shape=(32, 28, 28)  max_diff=2.1e-06  PASS
conv2    shape=(32, 28, 28)  max_diff=8.7e-06  PASS
relu2    shape=(32, 28, 28)  max_diff=8.7e-06  PASS
pool1    shape=(32, 14, 14)  max_diff=8.7e-06  PASS
conv3    shape=(32, 14, 14)  max_diff=1.9e-05  PASS
relu3    shape=(32, 14, 14)  max_diff=1.9e-05  PASS
conv4    shape=(32, 14, 14)  max_diff=3.2e-05  PASS
relu4    shape=(32, 14, 14)  max_diff=3.2e-05  PASS
pool2    shape=(32, 7, 7)   max_diff=3.2e-05  PASS
flat     shape=(1, 1, 1568)  max_diff=3.2e-05  PASS
logits   shape=(1, 10)       max_diff=1.1e-04  PASS
```

Notice how the errors *grow* with depth. That is expected — each layer's rounding error is amplified by the next layer's dot products. The important thing is that they stay below the tolerance threshold.

### 4.6.2 What if you see a failure

If the first failure is at `input`, check the input file.
If it is at `conv1`, check the weight order and indexing.
If it is deeper, work from the first divergence.
If you get NaN or Inf at any stage, there is likely an out-of-bounds read that is pulling garbage into the computation — check under AddressSanitizer.

## 4.7 What to Do When They Disagree

The most important debugging methodology:

1. **Find the first divergence.** Not the last, not the largest. The first.
2. **Check the input to that layer.** If the input is also wrong, the bug is upstream.
3. **Check the shape.** A shape mismatch is often a symptom of a wrong parameter (kernel size, stride, padding).
4. **Check the weights.** If shapes are right and the input is right but the output is wrong, the weights for that layer are probably being read wrong.
5. **Check the math.** If input, shape, and weights all look right, the operation itself is wrong. Compare against the PyTorch reference implementation line by line.

The key point: **do not modify code randomly**. Every change should be motivated by a specific hypothesis about where the bug is.

Once you have found the bug:

- **Write a test that would have caught it.**
- **Fix the bug.**
- **Run the test.**
- **Run the parity comparison again.**
- **Commit.**

This is the debugging cycle. It is slow. It is worth it.

## 4.8 The Debugging Playbook

When something is wrong, classify the failure before fixing it.

**Compilation error** — check the file, line, symbol. Missing include? Wrong type? Typo?

**Linker error** — missing definition? Wrong library linked? Name mismatch between declaration and definition?

**Crash at runtime** — null pointer? Out-of-bounds? Use-after-free? Wrong file path? Run under ASan.

**Wrong prediction** — do NOT immediately change the model. Check:
1. Input preprocessing.
2. Weights (loaded correctly? correct file?).
3. Tensor shapes (do they match the architecture?).
4. Layer order (does `model_forward` match `forward`?).
5. Conv indexing.
6. Pool indexing.
7. Flatten order.
8. Softmax.

**C differs from Python** — find the *first* divergent layer (Part IV). Fix the layer that first diverges.

**Model trains but performance is bad** — check:
1. Is training loss actually decreasing? (If not, learning rate or optimizer.)
2. Is validation loss much higher than training loss? (Overfitting.)
3. Is validation accuracy stagnating? (Learning rate too high, or model capacity.)

---

# Part V — Testing and Build

## 5.1 Test Philosophy — Failure-Mode by Failure-Mode

The test suite is not about coverage. It is about specific failure modes. Every test exists because there is a specific way the code could be wrong, and the test is designed to catch that.

### 5.1.1 `test_tensor` — the aliasing bug

A flattening formula like `(c*H + y)*W + x` has a class of bug where two different `(c, y, x)` triples alias to the same flat address. The bug would not crash — the program would run fine — but writing to one cell would silently corrupt another.

The test:

```c
Tensor t = tensor_alloc(2, 3, 3);
tensor_set(&t, 1, 2, 0, 7.5f);
CHECK(CLOSE(tensor_get(&t, 1, 2, 0), 7.5f), "set/get [1,2,0]");
CHECK(CLOSE(tensor_get(&t, 1, 1, 0), 0.0f), "neighbor [1,1,0] unaffected");
CHECK(CLOSE(tensor_get(&t, 0, 2, 0), 0.0f), "neighbor [0,2,0] unaffected");
```

The "neighbor unaffected" checks are what catch the aliasing bug. If `(1,2,0)` and `(1,1,0)` aliased to the same address, writing to the first would set the second to 7.5, and the check for 0.0 would fail.

### 5.1.2 `test_conv2d` and `test_conv2d_padding` — two different checks

The no-padding test checks the arithmetic: does the nested-loop sum compute what the convolution formula says it should.

The padding test checks something else: does the bounds-checking logic behave like real zero-padding. It uses an input where the answer is unambiguous by hand, so any error in the padding handling is obvious.

### 5.1.3 `test_maxpool2d` — window placement

Confirms window placement (output `(0, 1)` reads from input `(0, 2)-(1, 3)`, not from `(0, 0)-(1, 1)` or some other wrong position) and max selection.

### 5.1.4 `test_model_load` — the read order

Uses a synthetic file where every float equals its own byte position. If the read order is wrong, the values that come back will be obviously wrong (e.g., `conv1_b[0]` reading back as `288.0` instead of `0.0`, because the boundary between `conv1_w` and `conv1_b` got shifted).

### 5.1.5 `test_canvas_draw_at_edge`, under ASan

Draws at the canvas corner and confirms no out-of-bounds write. Under AddressSanitizer, this catches the specific class of bug where the bounds check is missing or wrong.

### 5.1.6 What is deliberately not tested

Whether real trained weights produce correct predictions. That is not a unit-testable property (no hand-computable expected value). It is covered by parity verification (Part IV) and evaluation (Part VI).

## 5.2 Unit Tests

The full `test_nn.c` includes `test_tensor`, `test_linear`, `test_relu`, `test_argmax`, `test_conv2d`, `test_maxpool2d`. Each is small and hand-computable.

The test runner:

```c
int main(void) {
    test_tensor();
    test_linear();
    test_relu();
    test_argmax();
    test_conv2d();
    test_maxpool2d();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
```

Exit code 0 means pass, 1 means fail. This is what `make test` checks.

### 5.2.1 The test assertion library

Create `tests/test_assert.h`:

```c
#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H

#include <math.h>
#include <stdio.h>

#define TEST_ASSERT(condition)                                          \
    do {                                                                \
        if (!(condition)) {                                             \
            fprintf(stderr, "FAIL: %s:%d: %s\n",                        \
                    __FILE__, __LINE__, #condition);                    \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance)                   \
    do {                                                                \
        double _a = (double)(actual);                                   \
        double _e = (double)(expected);                                 \
        if (fabs(_a - _e) > (tolerance)) {                              \
            fprintf(stderr,                                             \
                    "FAIL: %s:%d: actual=%f expected=%f tol=%f\n",      \
                    __FILE__, __LINE__, _a, _e, (double)(tolerance));   \
            return 1;                                                   \
        }                                                               \
    } while (0)

#endif
```

Line by line:

- `#ifndef TEST_ASSERT_H` — Prevents duplicate inclusion.
- `#define TEST_ASSERT(condition)` — Defines a reusable assertion macro.
- `do { ... } while (0)` — Begins a single-execution block so the macro behaves like one statement.
- `if (!(condition)) { ... }` — Stops the test when the condition is false.
- The backslash continues the macro onto the next source line.

### 5.2.2 `tests/test_relu.c`

```c
#include "../c/include/nn.h"
#include "test_assert.h"

int main(void) {
    float values[] = {-3.0f, -1.0f, 0.0f, 2.0f, 5.0f};

    relu(values, 5);

    TEST_ASSERT_NEAR(values[0], 0.0f, 1e-6);
    TEST_ASSERT_NEAR(values[1], 0.0f, 1e-6);
    TEST_ASSERT_NEAR(values[2], 0.0f, 1e-6);
    TEST_ASSERT_NEAR(values[3], 2.0f, 1e-6);
    TEST_ASSERT_NEAR(values[4], 5.0f, 1e-6);

    return 0;
}
```

Line by line:

- `#include "../c/include/nn.h"` — Includes the public neural-network API.
- `#include "test_assert.h"` — Includes the local test helpers.
- `float values[] = {...}` — Creates values covering negative, zero, and positive cases.
- `relu(values, 5);` — Runs the actual project ReLU implementation.
- The `TEST_ASSERT_NEAR` lines check the expected outputs.

### 5.2.3 `tests/test_linear.c`

A linear layer is easy to test because we can calculate the expected answer manually.

Choose:

```
W = [[1, 2],
     [3, 4]]

b = [10, 20]

x = [5, 6]
```

Then:

```
y0 = 1*5 + 2*6 + 10 = 27
y1 = 3*5 + 4*6 + 20 = 59
```

```c
#include "../c/include/nn.h"
#include "test_assert.h"

int main(void) {
    const float W[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    const float b[] = {10.0f, 20.0f};
    const float x[] = {5.0f, 6.0f};
    float y[2];

    linear(W, b, x, y, 2, 2);

    TEST_ASSERT_NEAR(y[0], 27.0f, 1e-6);
    TEST_ASSERT_NEAR(y[1], 59.0f, 1e-6);

    return 0;
}
```

## 5.3 The Makefile

Reproduced here for reference:

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude
LDFLAGS_APP = -lraylib -lm -lpthread -ldl -lX11

SRC = src/nn.c src/ui.c
TEST_NN_SRC = ../tests/test_nn.c
TEST_UI_SRC = ../tests/test_ui.c
VERIFY_SRC = ../tools/verify.c

.PHONY: all app test test_nn test_ui verify clean

all: test app

app: src/main.c $(SRC)
	$(CC) $(CFLAGS) $^ -o number_guesser $(LDFLAGS_APP)

test: test_nn test_ui
	./test_nn
	./test_ui

test_nn: $(TEST_NN_SRC) src/nn.c
	$(CC) $(CFLAGS) $^ -o test_nn -lm

test_ui: $(TEST_UI_SRC) src/ui.c
	$(CC) $(CFLAGS) $^ -o test_ui -lm

verify: $(VERIFY_SRC) src/nn.c
	$(CC) $(CFLAGS) $^ -o verify -lm

clean:
	rm -f number_guesser test_nn test_ui verify
```

Key points:

- `-Wall -Wextra -std=c11` for warnings and standard.
- `-Iinclude` for header search.
- `-lm` for math functions (link math library).
- Separate targets for `test` (no raylib needed), `app` (needs raylib), `verify` (no raylib).

The distinction matters: `make test` runs in a fraction of a second and requires no display. `make app` requires raylib and a window. Keeping them separate means tests are always runnable.

## 5.4 CMake, and When It Earns Its Keep

CMake exists to solve cross-platform build generation. For a project with one platform and one dependency, it adds indirection without benefit. Stick with Make until you actually need CMake.

A minimal `CMakeLists.txt` for future reference:

```cmake
cmake_minimum_required(VERSION 3.20)
project(NumberGuesser C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)

add_executable(number_guesser
    c/src/main.c
    c/src/nn.c
    c/src/ui.c
)

target_include_directories(number_guesser PRIVATE c/include)
target_link_libraries(number_guesser PRIVATE raylib m)
```

The trigger to switch would be: you need to support Windows and Linux with different toolchains, or you need to integrate with an IDE that speaks CMake.

### 5.4.1 Wiring tests into CMake

If you do use CMake, the tests get wired in like this:

```cmake
enable_testing()

add_executable(test_relu
    tests/test_relu.c
    c/src/nn.c
)

target_include_directories(test_relu PRIVATE c/include)

if(UNIX AND NOT APPLE)
    target_link_libraries(test_relu PRIVATE m)
endif()

add_test(NAME test_relu COMMAND test_relu)
```

Explanation:

- `enable_testing()` activates CTest.
- `add_executable` creates a separate executable.
- Reusing `nn.c` means the test exercises real production code.
- `target_include_directories` exposes `nn.h`.
- `m` is required on Linux because `nn.c` uses math functions.
- `add_test` registers the executable with CTest.

Run:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Do not call the test complete until CTest actually runs it.

## 5.5 Sanitizers

Sanitizers instrument the code at compile time to catch bugs at runtime that would otherwise be silent.

**AddressSanitizer (ASan)** catches:

- Heap buffer overflow
- Stack buffer overflow
- Use after free
- Double free
- Memory leaks

**UndefinedBehaviorSanitizer (UBSan)** catches:

- Signed integer overflow
- Division by zero
- Misaligned access
- Null pointer dereference

To build with both:

```bash
gcc -Wall -Wextra -std=c11 -Iinclude \
    -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1 \
    tests/test_nn.c src/nn.c -o test_nn_asan -lm
./test_nn_asan
```

Flags explained:

- `-fsanitize=address,undefined` — enable both.
- `-fno-omit-frame-pointer` — accurate stack traces.
- `-g` — debug symbols.
- `-O1` — light optimization. `-O0` is too slow; `-O2` might inline away the bug.

### 5.5.1 Reading a report

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address ...
WRITE of size 4 at ...
    #0 tensor_set nn.c:42
    #1 conv2d nn.c:104
    #2 model_forward nn.c:180
    #3 main main.c:87
```

Top frame is where the bug is. Trace back up to find which caller passed an out-of-bounds coordinate. Usually the fix is in the caller, not in `tensor_set`.

### 5.5.2 Common ASan reports and fixes

- `heap-buffer-overflow`: loop bound is wrong. Check indices.
- `use-after-free`: tensor used after `tensor_free`.
- `leak`: a `tensor_alloc` without matching `tensor_free`.

### 5.5.3 Adding to the Makefile

Should be a persistent target, not a manual invocation:

```makefile
test-asan: tests/test_nn.c src/nn.c
	$(CC) -fsanitize=address,undefined -fno-omit-frame-pointer \
	      -g -O1 $(CFLAGS) $^ -o test_nn_asan -lm
	./test_nn_asan
```

### 5.5.4 The CMake sanitizer option

Add this option to CMake:

```cmake
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(ENABLE_SANITIZERS AND NOT MSVC)
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )
    add_link_options(
        -fsanitize=address,undefined
    )
endif()
```

Run:

```bash
cmake -S . -B build-sanitize -DENABLE_SANITIZERS=ON
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure
```

Do not interpret "tests passed" without sanitizer testing as "C is safe".

## 5.6 CI

CI's value proposition is "catch regressions automatically." That is only valuable once you have a regression-worthy thing to protect — i.e., once real-weight verification has produced a baseline.

**What to add, once milestone 1 is done**:

```yaml
name: CI
on: [push, pull_request]
jobs:
  c-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd c && make test
      - run: cd c && make test-asan
      - run: cd c && make verify
      - run: ./verify models/weights.bin models/debug_input.bin | diff - notes/expected_verify_output.txt
  python-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python -m py_compile python/*.py
```

The `verify` step is the one that actually protects correctness — it would fail if a future change to `conv2d`, `model_load`, or `export.py`'s key order silently broke the pipeline.

### 5.6.1 The full CI workflow file

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake build-essential libraylib-dev

      - name: Configure
        run: cmake -S . -B build

      - name: Build
        run: cmake --build build -j2

      - name: Test
        run: ctest --test-dir build --output-on-failure
```

Line by line:

- `name: CI` — Names the workflow.
- `on:` — Runs the workflow on pushes and pull requests.
- `jobs:` — Defines the build-and-test job.
- `runs-on: ubuntu-latest` — Uses a fresh Ubuntu environment.
- `steps:` — Starts the first step.
- `- name: Checkout` — Checks out the repository.
- `- name: Install dependencies` — Refreshes package metadata, installs CMake, GCC/build tools, and raylib.
- `- name: Configure` — Configures the CMake build.
- `- name: Build` — Compiles the project.
- `- name: Test` — Runs the registered tests.

## 5.7 The Test Assertion Library

Already shown in §5.2.1. Reproduced here for completeness.

```c
#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H

#include <math.h>
#include <stdio.h>

#define TEST_ASSERT(condition)                                          \
    do {                                                                \
        if (!(condition)) {                                             \
            fprintf(stderr, "FAIL: %s:%d: %s\n",                        \
                    __FILE__, __LINE__, #condition);                    \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance)                   \
    do {                                                                \
        double _a = (double)(actual);                                   \
        double _e = (double)(expected);                                 \
        if (fabs(_a - _e) > (tolerance)) {                              \
            fprintf(stderr,                                             \
                    "FAIL: %s:%d: actual=%f expected=%f tol=%f\n",      \
                    __FILE__, __LINE__, _a, _e, (double)(tolerance));   \
            return 1;                                                   \
        }                                                               \
    } while (0)

#endif
```

This header is the single most reused file in the test suite. Get it right once, use it everywhere.

---

# Part VI — Real Handwriting Evaluation

## 6.1 Why MNIST Accuracy Isn't Enough

MNIST accuracy is 99%+. That does not mean the model works on your handwriting.

The MNIST test set was collected from the same distribution as the training set: census workers and students, drawn with a stylus at a specific stroke width, anti-aliased in a specific way, centered in the frame. Your mouse-drawn digits are a *different* distribution:

- Different stroke width.
- Different anti-aliasing.
- Different centering (you might draw anywhere on the canvas).
- Different size (you might draw tiny or large).
- Different shape (your digits might be slanted or stylized).

This is called *domain shift*. The model has never seen your specific drawing distribution. Its accuracy on MNIST tells you about its accuracy on MNIST-like images; it tells you *nothing* about its accuracy on your drawings.

The gap can be huge:

```
MNIST test accuracy: 99.2%
Your drawings: 65.0%
```

Or it can be small, if your drawings happen to match MNIST's distribution. The only way to know is to measure.

## 6.2 Building a Handwriting Dataset

Collect 20–50 drawings per digit (200–500 total to start). Use the actual application — draw in the Raylib canvas, save the drawing, label it.

Directory structure:

```
data/handwriting/
├── 0/
│   ├── 00001.bin
│   ├── 00002.bin
│   └── ...
├── 1/
├── ... 
└── 9/
```

Each `.bin` file contains the **28×28 preprocessed tensor**, not the raw 280×280 canvas. This is the actual model input, and it is what you want to compare across models and preprocessing variants.

**Why save the preprocessed tensor, not the raw canvas**:

- It is what the model sees, so it is what determines the model's output.
- It is much smaller (784 floats = 3136 bytes vs. 280×280 = 313,600 bytes).
- It is directly comparable across preprocessing versions (you can compare "same digit, different preprocessing" by looking at the preprocessed files).

A helper in `main.c` saves the drawing:

```c
static void save_drawing(const AppState *app, int label) {
    float mnist[28 * 28];
    canvas_to_mnist_input(app, mnist);

    char path[256];
    static int counter = 0;
    snprintf(path, sizeof(path),
             "data/handwriting/%d/%05d.bin", label, counter++);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fwrite(mnist, sizeof(float), 28 * 28, f);
    fclose(f);

    printf("saved %s\n", path);
}
```

Hook it into keyboard shortcuts:

```c
for (int digit = 0; digit <= 9; digit++) {
    if (IsKeyPressed(KEY_ZERO + digit)) {
        save_drawing(&app, digit);
        canvas_clear(&app);
    }
}
```

Draw a 7, press 7, clear, repeat.

## 6.3 The Batch Evaluator

Once you have a dataset, you need a tool that runs inference over the whole directory and produces accuracy metrics.

```c
#include "../include/nn.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "models/weights.bin";
    const char *data_dir = argc > 2 ? argv[2] : "data/handwriting";

    CnnModel model;
    if (model_load(&model, weights) != 0) return 1;

    int confusion[10][10] = {0};
    int total_per_class[10] = {0};
    int total_correct = 0;
    int total = 0;

    for (int true_label = 0; true_label <= 9; true_label++) {
        char class_dir[256];
        snprintf(class_dir, sizeof(class_dir), "%s/%d", data_dir, true_label);

        DIR *d = opendir(class_dir);
        if (!d) continue;

        struct dirent *entry;
        while ((entry = readdir(d)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            if (strstr(entry->d_name, ".bin") == NULL) continue;

            char path[512];
            snprintf(path, sizeof(path), "%s/%s", class_dir, entry->d_name);

            Tensor input = tensor_alloc(1, 28, 28);
            FILE *f = fopen(path, "rb");
            if (!f) { tensor_free(&input); continue; }
            fread(input.data, sizeof(float), 28 * 28, f);
            fclose(f);

            float logits[10];
            model_forward(&model, &input, logits);
            tensor_free(&input);

            int pred = argmax(logits, 10);
            confusion[true_label][pred]++;
            total_per_class[true_label]++;
            total++;
            if (pred == true_label) total_correct++;
        }
        closedir(d);
    }

    printf("Total: %d\n", total);
    printf("Correct: %d\n", total_correct);
    printf("Accuracy: %.2f%%\n",
           100.0 * total_correct / (total > 0 ? total : 1));

    printf("\nPer-class accuracy:\n");
    for (int i = 0; i < 10; i++) {
        if (total_per_class[i] > 0) {
            printf("  %d: %d/%d (%.1f%%)\n",
                   i, confusion[i][i], total_per_class[i],
                   100.0 * confusion[i][i] / total_per_class[i]);
        }
    }

    printf("\nConfusion matrix (rows=actual, cols=predicted):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d:", i);
        for (int j = 0; j < 10; j++) {
            printf(" %3d", confusion[i][j]);
        }
        printf("\n");
    }

    return 0;
}
```

### 6.3.1 What the report tells you

**Total and accuracy** — the headline number.

**Per-class accuracy** — which digits the model handles well and which it does not. Some digits may be much harder than others.

**Confusion matrix** — which digits are confused with which. Example:

```
  0: 30  0  0  0  0  0  0  0  0  0
  1:  0 28  0  0  0  0  0  2  0  0
  3:  0  0  0 25  0  3  0  0  0  2
```

Row `1`, column `7` shows 2 samples of digit `1` were predicted as `7`. That is a specific, actionable failure pattern.

## 6.4 Metrics — Accuracy, Confusion Matrix, Per-Class

**Accuracy** is `correct / total`. Simple but incomplete.

**Per-class accuracy** is `correct_i / total_i` for each class. Reveals imbalance: a model that is 99% overall but 30% on digit `1` is not equally good everywhere.

**Confusion matrix**: `confusion[i][j]` is the count of samples with true label `i` predicted as `j`. The diagonal is correct predictions; off-diagonal entries are errors.

**Precision** for class `i`: of all samples predicted as `i`, what fraction are actually `i`?

```
precision_i = confusion[i][i] / sum_j confusion[j][i]
```

**Recall** for class `i`: of all samples that are actually `i`, what fraction are predicted correctly?

```
recall_i = confusion[i][i] / sum_j confusion[i][j]
```

**F1 score**: harmonic mean of precision and recall.

For balanced classes, accuracy is enough. For imbalanced classes, per-class metrics are essential.

## 6.5 Confidence Calibration

A softmax output of 0.9 does *not* mean "90% chance this prediction is correct." Softmax gives you a number that sums to 1, but the actual accuracy at that confidence level is an empirical question.

To measure calibration: bucket predictions by confidence and compute empirical accuracy within each bucket.

```
Bucket      | Samples | Correct | Empirical Accuracy
0.5 - 0.6   | 20      | 12      | 60%
0.6 - 0.7   | 30      | 22      | 73%
0.7 - 0.8   | 40      | 34      | 85%
0.8 - 0.9   | 60      | 54      | 90%
0.9 - 1.0   | 150     | 148     | 98.7%
```

A well-calibrated model has empirical accuracy ≈ bucket confidence. If the 0.9–1.0 bucket is only 70% correct, the model is overconfident, and you should not trust its high-confidence predictions as much as the numbers suggest.

**Why this matters**: a UI that says "99.8% confident" when the model is actually only 70% accurate at that confidence level is misleading. A trustworthy UI either uses calibrated confidence or avoids displaying raw softmax values.

## 6.6 Error Analysis

For every bad prediction, ask:

1. **Was the input bad?** Look at the preprocessed 28×28 image. If it is unrecognizable to a human, the model is not at fault — the drawing was.
2. **Was preprocessing bad?** If the preprocessed image is recognizable but looks nothing like what training samples look like (e.g., stroke is way too thick or thin), preprocessing is at fault.
3. **Was the model uncertain?** Look at the confidence. If it is near 0.5, the model is unsure, and the wrong prediction is a "close call."
4. **Was the model confidently wrong?** This is the most interesting failure. The model was sure it saw a `7` when it was actually a `1`. These examples are valuable for understanding model biases.

Collect failure examples. Eventually build a "failure gallery" that lets you visually scan through misclassified drawings and identify patterns.

## 6.7 The Failure Gallery

Build a tool that renders:

- The raw 280×280 canvas (if you saved it).
- The preprocessed 28×28 input.
- The logits/probabilities.
- The true label and predicted label.

Arrange failures in a grid, sorted by confidence (most confidently wrong first). The patterns you see will suggest concrete preprocessing or model changes.

---

# Part VII — ML Improvement

## 7.1 The Experiment Harness

The single most important change in this part: turn "I changed the model and it feels better" into a structured experiment with a hypothesis, a controlled change, and a measurement.

Every experiment should record:

```
experiment_id
date
git commit
model architecture
optimizer
learning rate
batch size
epochs
seed
training dataset version
preprocessing version
MNIST test accuracy
own-drawing accuracy
confusion matrix
notes
```

Example:

```
EXP-001
baseline CNN
Adam, lr=0.001, batch=64, epochs=5
seed=42
MNIST test: 99.2%
own-drawing: 91.3%
```

Then a change:

```
EXP-002
EXP-001 + bounding-box preprocessing
MNIST test: 99.2%  (no change — MNIST already centered)
own-drawing: 94.7%  (+3.4%)
```

Now you have evidence. "Bounding-box preprocessing improves own-drawing accuracy by 3.4 percentage points."

## 7.2 Preprocessing Experiments

The specific pipelines to compare:

**A: Current pipeline**
- Bounding box, square, margin, resize to 20×20, center in 28×28.

**B: Bounding box without centering**
- Bounding box, resize to 20×20 (aspect ratio not preserved), place in top-left of 28×28.

**C: Crop + center-of-mass**
- Compute the intensity-weighted center of mass.
- Translate the digit so the center of mass is at the image center.

**D: Aspect-ratio-preserving**
- Bounding box, scale preserving aspect ratio, pad to 28×28.

**E: Stroke normalization**
- Compute average stroke width, scale to make it match MNIST's average.

For each, run the full evaluation pipeline on the same dataset. Compare accuracy, per-class accuracy, and confusion matrix.

The order of experiments matters: do not try E before you have confirmed A works. Start simple.

## 7.3 Data Augmentation

Augmentation = generating additional training samples by transforming existing ones.

For handwriting, useful transformations:

- **Translation**: shift by ±2 pixels. Handwriting is not always centered.
- **Rotation**: ±5 degrees. Handwriting is not always upright.
- **Scale**: ±10%. Different writing sizes.
- **Shear**: small horizontal or vertical distortion. Different handwriting styles.
- **Stroke width**: dilate or erode the strokes slightly.
- **Noise**: small random perturbations.

**Critical caveat**: augmentation must preserve the digit's identity. A 45-degree rotation might turn a `1` into a `7`. A 90-degree rotation turns a `6` into a `9`. Only apply transformations that keep the digit recognizable.

Typical PyTorch setup:

```python
transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=5,
        translate=(2/28, 2/28),
        scale=(0.9, 1.1),
        shear=5,
    ),
    transforms.ToTensor(),
])
```

The gains from augmentation are usually modest (1-3 percentage points on well-tuned models) but reliable.

## 7.4 Training Improvements

**Learning rate**:

- Too high: loss oscillates, diverges.
- Too low: training is painfully slow.
- Typical: 1e-3 for Adam, 1e-2 for SGD with momentum.

**Batch size**:

- Smaller (32, 16): noisier gradients, sometimes better generalization, slower per epoch.
- Larger (128, 256): more stable gradients, faster per epoch, sometimes worse generalization.

**Optimizer**:

- SGD: simple, well-understood, requires careful learning-rate tuning.
- SGD + momentum: adds velocity, smoother convergence.
- Adam: adaptive per-parameter learning rate, usually good defaults.

**Regularization**:

- Weight decay: penalizes large weights.
- Dropout: randomly zeros activations during training.
- Early stopping: stop when validation loss stops improving.

**Learning-rate schedules**:

- Step decay: reduce LR by factor every N epochs.
- Cosine annealing: smooth decay following a cosine curve.
- Reduce-on-plateau: reduce LR when validation loss plateaus.

## 7.5 Architecture Experiments

Later, once preprocessing and data are sorted:

- **More channels**: 32 → 64. More capacity, slower inference.
- **More layers**: add a third block. More capacity, risk of overfitting.
- **Different kernel size**: 3×3 → 5×5. Larger receptive field, more parameters.
- **Batch normalization**: normalizes layer inputs, speeds training, sometimes improves accuracy.
- **Residual connections**: skip connections between blocks, easier to train deeper networks.

Architecture changes should be late because they are the most expensive to test (retrain from scratch) and the least likely to fix a domain-shift problem.

## 7.6 The Experiment Log Format

Store each experiment as a markdown file in `experiments/`:

```markdown
# EXP-003: Augmentation with RandomAffine

**Date**: 2024-01-15
**Commit**: a1b2c3d
**Status**: Complete

## Hypothesis
Adding small random affine transforms during training will improve
real-handwriting accuracy by making the model more robust to natural
variation in drawn digits.

## Configuration
- Architecture: baseline CNN (4 conv, 2 pool, 1 fc)
- Optimizer: Adam, lr=1e-3
- Batch size: 64
- Epochs: 5
- Seed: 42
- Preprocessing: version 2 (bounding box + margin + 20x20 + center)
- Augmentation: RandomAffine(degrees=5, translate=(2/28), scale=(0.9,1.1), shear=5)

## Results
- MNIST test accuracy: 99.1% (baseline: 99.2%)
- Own-handwriting accuracy: 95.3% (baseline: 94.7%)
- Delta: +0.6% on own handwriting, -0.1% on MNIST

## Analysis
Small improvement on own handwriting at negligible cost on MNIST.
Worth keeping.

## Next Steps
Try stronger augmentation (degrees=10, scale=(0.85,1.15)).
```

This file format is the record. Write it before the experiment (as a hypothesis), update it after (as a result). The record is what makes the project cumulative rather than forgetful.

---

# Part VIII — Observability

## 8.1 Activation Visualization

The C implementation already has explicit intermediate tensors. Exposing them for visualization is straightforward:

- After each conv/ReLU/pool, dump the tensor to a displayable format.
- In the UI, render each channel as a small grayscale image (normalize values to [0, 1] for display).

For `conv1` (32 channels of 28×28), display a 4×8 grid of 28×28 images.

For `conv4` (32 channels of 14×14), similar.

### 8.1.1 What you will see

**Conv1**: the earliest layer typically responds to edges and short strokes. Different channels highlight different orientations.

**Later layers**: more complex, more abstract patterns. Some channels respond to curves, some to intersections, some to whole-digit features.

**ReLU**: usually shows sparsity — many activations are zero (negative values clamped). The fraction of zeros is a measure of "how active" the layer is.

**Pool**: visibly smaller than the input, and smoother (max preserves the strongest activation in each window).

## 8.2 Layer Timing

Wrap each layer in a timer:

```c
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

double t0 = now_seconds();
Tensor a = conv2d(input, m->conv1_w, m->conv1_b, 32, 3, 1, 1);
double t1 = now_seconds();
printf("conv1: %.3f ms\n", (t1 - t0) * 1000.0);
```

Run the forward pass many times, average the timings.

Expected profile (rough estimates, verify on your machine):

```
conv1: 0.4 ms   (small in_channels)
conv2: 5.6 ms   (32 in_channels)
conv3: 1.4 ms   (smaller spatial)
conv4: 1.4 ms   (smaller spatial)
linear: 0.05 ms
total:  ~9 ms
```

The later convs dominate runtime because they have 32 input channels each. Optimizing the earlier convs will not help; optimizing the later convs will.

## 8.3 The Debugging Playbook

When something is wrong, classify the failure before fixing it.

**Compilation error** — check the file, line, symbol. Missing include? Wrong type? Typo?

**Linker error** — missing definition? Wrong library linked? Name mismatch between declaration and definition?

**Crash at runtime** — null pointer? Out-of-bounds? Use-after-free? Wrong file path? Run under ASan.

**Wrong prediction** — do NOT immediately change the model. Check:
1. Input preprocessing.
2. Weights (loaded correctly? correct file?).
3. Tensor shapes (do they match the architecture?).
4. Layer order (does `model_forward` match `forward`?).
5. Conv indexing.
6. Pool indexing.
7. Flatten order.
8. Softmax.

**C differs from Python** — find the *first* divergent layer (Part IV). Fix the layer that first diverges.

**Model trains but performance is bad** — check:
1. Is training loss actually decreasing? (If not, learning rate or optimizer.)
2. Is validation loss much higher than training loss? (Overfitting.)
3. Is validation accuracy stagnating? (Learning rate too high, or model capacity.)

## 8.4 Logits and Softmax Inspection

The logits are the raw scores. Inspect them:

```python
logits = model(x)
print(logits)
print(torch.softmax(logits, dim=1))
```

If the logits are all approximately equal, the model is uncertain. If one logit dominates, the model is confident. The *gap* between the top logit and the second is the confidence measure.

For a stable display in the UI, compute softmax over the logits and show the top-3 probabilities.

---

# Part IX — Performance Engineering

## 9.1 Measure First

The only rule of optimization: **measure before you change.**

Premature optimization is worse than no optimization at all, because it makes the code more complex without producing a measurable improvement.

The workflow:

1. **Establish a baseline**: run the current implementation, measure.
2. **Profile**: find the bottleneck.
3. **Optimize the bottleneck**: change one thing, measure again.
4. **Verify correctness**: parity comparison must still pass.
5. **Repeat**: continue until satisfied.

## 9.2 Memory Reuse

The current `model_forward` allocates and frees each tensor. This is clean and correct, but it does involve allocator calls.

An optimization: pre-allocate a workspace and reuse buffers.

```c
typedef struct {
    Tensor a;
    Tensor b;
    Tensor p1;
    Tensor c;
    Tensor d;
    Tensor p2;
} InferenceWorkspace;

void workspace_init(InferenceWorkspace *ws);
void model_forward_workspace(const CnnModel *m, const Tensor *input,
                             float *logits, InferenceWorkspace *ws);
void workspace_free(InferenceWorkspace *ws);
```

The idea: allocate once, reuse across calls. The `a` buffer is overwritten with the next layer's output.

**Is this worth it?** Depends on how much time the allocator takes. For a small model with only a few allocations per forward pass, the answer might be "no." For a large model with many allocations, or for a very high-throughput application, it might be significant.

**Measure first**. If the profile shows `malloc`/`free` accounting for a meaningful fraction of runtime, then reuse. If not, the current approach is fine.

## 9.3 Cache-Aware Convolution

The current conv2d loops over:

```
for oc:
  for oy:
    for ox:
      for ic:
        for ky:
          for kx:
```

This is a natural order but not necessarily cache-optimal. The question is: does the inner loop touch memory in a pattern that fits in L1/L2 cache?

Potential reorderings:

- **Input-stationary**: loop over input positions in the outer loop, accumulate into outputs.
- **Output-stationary**: current approach, loop over output positions in the outer loop.
- **Weights-stationary**: loop over filter positions in the outer loop.

Different loops are optimal for different shapes. For this project, the difference may be small enough not to matter. Measure, do not guess.

## 9.4 SIMD, Eventually

SIMD (Single Instruction, Multiple Data) uses vector instructions to process multiple floats at once.

Example: dot product of two float32 arrays.

Scalar:
```c
float sum = 0;
for (int i = 0; i < n; i++) sum += a[i] * b[i];
```

SIMD (AVX2):
```c
__m256 sum = _mm256_setzero_ps();
for (int i = 0; i < n; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    sum = _mm256_fmadd_ps(va, vb, sum);
}
// ... horizontal sum ...
```

SIMD can give 4-8× speedups for well-vectorized code. But:

- Requires aligned memory or unaligned loads (slower).
- Requires the compiler to not reorder your code.
- Adds platform-specific code (`<immintrin.h>` on x86).
- Breaks portability to ARM, where AVX does not exist.

**Only after profiling shows convolution is the bottleneck.** And only after the simpler optimizations (loop reordering, memory reuse) have been tried.

## 9.5 Quantization

Quantization = representing weights and/or activations with fewer bits, typically int8 instead of float32.

Approach:

```
scale = max(|x|) / 127
q = round(x / scale)        # float to int8
x_approx = q * scale        # int8 to float (for comparison)
```

Convolution with int8 weights:
- Multiply int8 × int8 → int32 (exact, no overflow for reasonable magnitudes).
- Accumulate int32.
- Dequantize to float at the end.

Benefits: 4× smaller model file, potentially faster inference (integer arithmetic is faster on some hardware).

Costs: accuracy loss (small but nonzero), implementation complexity (need int8 arithmetic, scale handling).

**When to consider it**: after the model works, after profiling shows inference is too slow, and after you have tried simpler optimizations.

## 9.6 Compiler Flags

The compiler can do a lot of work for you, if you let it.

- `-O2` or `-O3`: aggressive optimization. `-O3` may vectorize loops automatically.
- `-march=native`: use the CPU's full instruction set (AVX, FMA, etc.). Non-portable — the binary will only run on CPUs with those instructions.
- `-ffast-math`: allow the compiler to reorder floating-point operations. **Dangerous**: it can change numerical results, breaking parity. Do not use unless you are aware of the consequences.
- `-funroll-loops`: unroll loops for better ILP. Usually automatic at `-O3`.
- `-flto`: link-time optimization. Can help with inlining across translation units.

For this project: start with `-O2`, and only move to `-O3` or `-march=native` after measuring.

---

# Part X — Two-Digit Recognition

## 10.1 The Two-Digit Problem

Given an image containing two handwritten digits, output the number.

Naive approach: train a 100-class classifier for "00" through "99". Bad for two reasons:

1. It does not scale — 3-digit numbers need 1000 classes, 4-digit numbers need 10,000.
2. It is wasteful — the model has to learn each two-digit number separately, when the underlying structure (digit 1, digit 2) is compositional.

Better approach: recognize individual digits, combine them.

```
image → segment → digit crops → CNN → digits → combine → number
```

## 10.2 Connected-Component Segmentation

The simplest segmentation: find connected components of foreground pixels.

```c
typedef struct {
    int min_x, min_y, max_x, max_y;
    int area;
} Component;

int segment_digits(const float *image, int width, int height,
                   float threshold, int min_area,
                   Component *components_out, int max_components);
```

Algorithm:

1. Threshold to binary: `mask[i] = (image[i] > threshold) ? 1 : 0`.
2. For each unvisited foreground pixel, start a BFS/DFS flood fill.
3. The flood fill marks every connected pixel and updates the bounding box.
4. Repeat until all pixels are visited.
5. Filter components by area (ignore tiny noise).
6. Sort by `min_x` (leftmost first).

## 10.3 Flood Fill in C

```c
typedef struct {
    int x, y;
} Pixel;

static int flood_fill(const uint8_t *mask, uint8_t *visited,
                      int width, int height,
                      int sx, int sy,
                      Component *out) {
    int capacity = 256;
    int size = 0;
    Pixel *queue = malloc(capacity * sizeof(Pixel));
    if (!queue) return 0;

    queue[size++] = (Pixel){sx, sy};
    visited[sy * width + sx] = 1;

    int min_x = sx, max_x = sx, min_y = sy, max_y = sy, area = 1;

    const int dx[] = {0, 0, -1, 1};
    const int dy[] = {-1, 1, 0, 0};

    while (size > 0) {
        Pixel p = queue[--size];

        for (int i = 0; i < 4; i++) {
            int nx = p.x + dx[i];
            int ny = p.y + dy[i];

            if (nx < 0 || nx >= width) continue;
            if (ny < 0 || ny >= height) continue;

            int idx = ny * width + nx;
            if (visited[idx]) continue;
            if (!mask[idx]) continue;

            visited[idx] = 1;
            if (nx < min_x) min_x = nx;
            if (nx > max_x) max_x = nx;
            if (ny < min_y) min_y = ny;
            if (ny > max_y) max_y = ny;
            area++;

            if (size >= capacity) {
                capacity *= 2;
                Pixel *newq = realloc(queue, capacity * sizeof(Pixel));
                if (!newq) { free(queue); return 0; }
                queue = newq;
            }
            queue[size++] = (Pixel){nx, ny};
        }
    }

    free(queue);

    out->min_x = min_x;
    out->max_x = max_x;
    out->min_y = min_y;
    out->max_y = max_y;
    out->area = area;
    return 1;
}
```

BFS rather than recursive DFS: recursion depth is bounded by the size of the component, which can be thousands of pixels. A deep recursion could overflow the stack. BFS uses an explicit queue, which is bounded by the size of the component but lives on the heap.

**Alternative: 8-connected components** — including diagonals. Sometimes better for handwriting, where diagonal strokes may only touch at corners. Worth experimenting with.

## 10.4 Bounding Boxes and Sorting

After segmentation, sort components by `min_x`:

```c
static int compare_components(const void *a, const void *b) {
    return ((const Component *)a)->min_x - ((const Component *)b)->min_x;
}

qsort(components, n_components, sizeof(Component), compare_components);
```

Then for each component, crop the bounding box, resize to 28×28, run the single-digit CNN, and combine the predictions.

The `Component` type:

```c
typedef struct {
    int min_x;
    int min_y;
    int max_x;
    int max_y;
    int area;
} BoundingBox;

static int box_width(const BoundingBox *box) {
    return box->max_x - box->min_x + 1;
}

static int box_height(const BoundingBox *box) {
    return box->max_y - box->min_y + 1;
}
```

Line by line:

- `typedef struct { ... } BoundingBox;` — Defines the structure.
- `min_x, min_y, max_x, max_y` — Stores the corners of the bounding box.
- `area` — Stores the number of pixels belonging to the component.
- `box_width` — Computes inclusive width.
- `box_height` — Computes inclusive height.

## 10.5 Number Decoding

For two digits:

```c
int digit_a = argmax(logits_a, 10);
int digit_b = argmax(logits_b, 10);
int number = digit_a * 10 + digit_b;
```

For N digits:

```c
int number = 0;
for (int i = 0; i < n; i++) {
    number = number * 10 + digit_i;
}
```

This is a clean, compositional way to handle any number of digits — as long as segmentation works.

## 10.6 Failure Cases

The simple segmentation baseline has known failure modes:

- **Touching digits**: `44` might become one component. Fix: analyze projection profiles, or fall back to a sequence model.
- **Noise**: small specks might be detected as components. Fix: filter by area.
- **Dots on `i`, `j`**: not relevant for digits, but "decimal point" in a number might be misdetected.
- **Very close spacing**: two digits might merge into one component.
- **Very wide spacing**: might be interpreted as three components if there is noise.

The fix for most of these is: collect failure examples, analyze them, and choose a more advanced method (Part XI) when the simple method breaks.

## 10.7 Projection-Based Segmentation

A simpler two-digit experiment can use vertical projection.

For each x:

```
column_sum[x] = Σ_y pixel[y,x]
```

Then:

```
column_sum[x] > threshold
```

means the column contains ink.

A long zero region can separate digits.

This fails when digits touch, but it is excellent for learning.

Do not throw away simple methods just because they are not production OCR.

```c
void vertical_projection(
    const float *image,
    int width,
    int height,
    float *projection
) {
    for (int x = 0; x < width; ++x) {
        projection[x] = 0.0f;

        for (int y = 0; y < height; ++y) {
            projection[x] += image[y * width + x];
        }
    }
}
```

Line by line:

- `void vertical_projection(` — Defines a function that receives a flat grayscale image.
- `const float *image,` — The image width is needed for indexing.
- `int width,` — The image height controls the row loop.
- `int height,` — Output array where each x coordinate gets one summed value.
- `float *projection` — Begins the function body.
- `) {` — Loops over columns.
- `for (int x = 0; x < width; ++x) {` — Resets the current column sum.
- `projection[x] = 0.0f;` — Initializes.
- `for (int y = 0; y < height; ++y) {` — Loops over every row in this column.
- `projection[x] += image[y * width + x];` — Adds the current pixel to the column's projection.

## 10.8 When Segmentation Stops Working

If digits touch:

```
12
```

may become one connected component.

A simple split may fail.

Possible next methods:

- watershed-style separation
- contour analysis
- learned object detection
- sequence recognition
- CTC

For a serious OCR direction, sequence recognition is more scalable than creating a class for every possible number.

Instead of:

```
00
01
02
...
99
```

use:

```
digit vocabulary = 10
sequence length = variable
```

This is the conceptual bridge from digit classification to OCR.

---

# Part XI — Variable-Length OCR

## 11.1 Why Segmentation Stops Working

Segmentation works well when digits are clearly separated. It breaks when:

- Digits touch.
- Digits overlap.
- Strokes are ambiguous.
- The number is very long.

These are common in real handwriting. A production OCR system cannot rely on segmentation.

## 11.2 Sliding Windows and Feature Sequences

The alternative approach: treat the image as a *sequence* of features, and let the model learn the alignment.

Think of it this way: the image is `H × W`. Slide a window across the width, extract features at each x-position. You get a sequence of feature vectors:

```
x=0   → feature_0
x=1   → feature_1
...
x=W-1 → feature_{W-1}
```

Each feature is `C × H` values (for a CNN with C channels and spatial height H).

This sequence can be fed into a sequence model (RNN, transformer) that produces one output per position:

```
position 0 → class probabilities over {0, ..., 9, blank}
position 1 → ...
```

## 11.3 CTC — The Intuition

CTC (Connectionist Temporal Classification) handles the alignment problem: the model outputs one prediction per position, but the target string may be shorter. CTC defines a loss that marginalizes over all alignments.

Example: target string "472" (3 characters), sequence length 8 (say). The model might output:

```
_ 4 4 _ 7 _ 2 2
```

where `_` is a special "blank" token. Collapse repeated non-blank characters and remove blanks:

```
4 7 2
```

Or:

```
4 _ 7 7 _ _ 2 _
```

Also collapses to:

```
4 7 2
```

All these alignments are valid for the target. CTC loss sums over all of them (using dynamic programming), and training optimizes the total probability.

## 11.4 CTC in Practice

PyTorch has `nn.CTCLoss` built in. Typical usage:

```python
log_probs = F.log_softmax(logits, dim=-1)  # (T, N, C)
input_lengths = torch.full((N,), T, dtype=torch.long)
target_lengths = torch.tensor([len(t) for t in targets])
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

Requirements:
- `log_probs` is log-softmax over `C = 11` classes (10 digits + blank).
- `targets` is a concatenated tensor of all target characters.
- `input_lengths` and `target_lengths` give the length of each item in the batch.

For inference, use greedy decoding:

```python
def greedy_ctc_decode(log_probs):
    tokens = log_probs.argmax(dim=-1)  # (T,)
    result = []
    prev = -1  # blank or unset
    for t in tokens:
        t = t.item()
        if t != 0 and t != prev:  # 0 is blank
            result.append(t)
        prev = t
    return result
```

The decoder is simple: argmax at each position, skip blanks, collapse repeats.

## 11.5 Decoding — Greedy and Beam Search

**Greedy decoding**: argmax at each timestep, then collapse. Fast, usually good enough.

**Beam search**: maintain the top-k candidate sequences, expand each one, prune. Slower but can produce better results, especially when the model's outputs are ambiguous.

For a first implementation, greedy is fine. Beam search is a later optimization.

## 11.6 The Full OCR Architecture

A complete OCR model:

```
Input: H × W grayscale
        ↓
CNN feature extractor (Conv, ReLU, Pool, ...)
        ↓
Feature map: C × H' × W'
        ↓
Sequence extraction: reshape to T = W' positions, each with C·H' features
        ↓
Sequence model (RNN, transformer, or just a linear layer)
        ↓
Output: T × (10+1) logits
        ↓
CTC loss (training) or CTC decode (inference)
        ↓
String output
```

For a minimal first implementation:
- CNN with a few conv layers, pooling only in height (keep width as the sequence dimension).
- Linear layer to map features to `(10+1)` classes.
- CTC loss.

This can be surprisingly effective for simple OCR tasks.

---

# Part XII — C Engineering for OCR

## 12.1 Sequence Types

OCR needs new types beyond `Tensor`:

```c
typedef struct {
    int length;
    int vocab_size;
    float *logits;   /* shape (length, vocab_size) */
} SequenceLogits;

typedef struct {
    int length;
    int *tokens;
} TokenSequence;
```

With corresponding alloc/free:

```c
SequenceLogits sequence_alloc(int length, int vocab_size);
void sequence_free(SequenceLogits *s);
```

The ownership pattern from `Tensor` extends to these new types.

## 12.2 Error Propagation

Current functions return `int` (0 for success, non-zero for failure) or crash on error. For a growing project, an error enum is cleaner:

```c
typedef enum {
    NN_OK = 0,
    NN_ERR_ALLOC,
    NN_ERR_FILE,
    NN_ERR_FORMAT,
    NN_ERR_SHAPE,
    NN_ERR_INVALID_ARGUMENT,
    NN_ERR_NUMERICAL
} NNStatus;

NNStatus model_load(CnnModel *model, const char *path);
```

Then callers check the status:

```c
NNStatus status = model_load(&model, path);
if (status != NN_OK) {
    fprintf(stderr, "model_load failed: %s\n", nn_status_string(status));
    return 1;
}
```

This is worth doing once the project has multiple model files (single-digit, two-digit, OCR), where the failure modes are more interesting.

## 12.3 Model Format v2

Once there is more than one model, the raw format becomes insufficient. A v2 format:

```
MAGIC (4 bytes: "NGM1")
VERSION (uint32)
DTYPE (uint32: 0=float32, 1=float16, 2=int8)
NUM_TENSORS (uint32)
for each tensor:
    NAME_LENGTH (uint32)
    NAME (utf-8 bytes)
    RANK (uint32)
    DIMENSIONS (RANK × uint32)
    DATA (product(dimensions) × dtype_size)
CHECKSUM (uint32 or uint64)
```

Benefits:
- Self-describing: you can inspect a model file without the source code.
- Version-aware: reject incompatible files with a clear error.
- Architecture-aware: reject files for the wrong architecture.
- Checksum: detect corruption.

The v2 format is a significant engineering effort. Do it once the project has multiple models, not before.

### 12.3.1 A minimal model header

```c
#include <stdint.h>

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t dtype;
    uint32_t layer_count;
    uint32_t payload_bytes;
} ModelHeader;
```

Line by line:

- `#include <stdint.h>` — Imports fixed-width integer types.
- `typedef struct {` — Defines a metadata structure that precedes the raw model payload.
- `char magic[4];` — Stores a four-byte identifier such as `NN01`.
- `uint32_t version;` — Stores the serialization version.
- `uint32_t dtype;` — Stores the numeric data type identifier.
- `uint32_t layer_count;` — Stores the number of serialized layers.
- `uint32_t payload_bytes;` — Stores the payload size so the loader can validate the file.

## 12.4 Determinism and Experiment Metadata

Every model file should record:
- Architecture name/version.
- Input shape.
- Dtype.
- Preprocessing version.
- Training commit hash.

Then a model file is not just weights; it is a *recipe*. Loading a model tells you exactly which preprocessing to use, which architecture to expect, which training run produced it.

## 12.5 Model Loader Hardening

The current `model_load` has one check: file size. A hardened version adds:

- **Magic validation**: reject files that do not start with the expected magic bytes.
- **Version validation**: reject files whose version is not supported.
- **Payload length**: check that the declared payload length matches the actual file size.
- **Checksum**: detect corruption.
- **Truncated file**: detect files that end in the middle of a tensor.
- **Wrong architecture**: reject files that were trained for a different architecture.
- **Wrong dtype**: reject files whose dtype does not match.

Each of these is independently testable. Implement one, test it, integrate it, document it, move on to the next.

---

# Part XIII — Backpropagation (Optional Keystone)

## 13.1 Why You Might Want To

You do not need to implement backprop for this project. PyTorch does it. The C side is inference-only.

But if you want to *understand* what PyTorch does — if you want to be able to look at any model and know, at the level of arithmetic, what is happening — implementing backprop is the way.

It is also a natural extension: the project already implements the forward pass in C. Adding the backward pass turns it into a full neural network library, in the "write your own framework" sense.

## 13.2 Linear Layer, Forward and Backward

Forward:

```
z = x @ W + b
```

Backward, given `dL/dz`:

```
dL/dW = dL/dz^T @ x          (or x^T @ dL/dz, depending on batch layout)
dL/db = sum(dL/dz, axis=batch)
dL/dx = dL/dz @ W^T
```

Verify against PyTorch with `torch.autograd`.

### 13.2.1 A tiny Python linear layer from scratch

```python
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad_output):
        self.grad_W = grad_output.T @ self.x
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W
```

Line by line:

- `import numpy as np` — Imports NumPy.
- `class Linear:` — Defines a trainable linear layer.
- `def __init__(self, in_features, out_features):` — Initializes weights with small random values.
- `self.W = np.random.randn(out_features, in_features) * 0.01` — Random weights.
- `self.b = np.zeros(out_features)` — Zero biases.
- `def forward(self, x):` — Defines the forward pass.
- `self.x = x` — Stores the input because backpropagation needs it.
- `return x @ self.W.T + self.b` — Computes Wx+b.
- `def backward(self, grad_output):` — Defines the backward pass.
- `self.grad_W = grad_output.T @ self.x` — Gradient of the weights.
- `self.grad_b = grad_output.sum(axis=0)` — Sums output gradients across the batch to get the bias gradient.
- `return grad_output @ self.W` — Propagates the gradient backward.

## 13.3 ReLU

Forward:

```
y = max(0, x)
```

Backward:

```
dL/dx = dL/dy * (x > 0 ? 1 : 0)
```

Element-wise.

## 13.4 MaxPool

Forward: output is the max in each window. Record the argmax position.

Backward: gradient flows only to the argmax position.

```
dL/dx[i][j] = dL/dy[k] if (i, j) is the argmax of window k
              0 otherwise
```

## 13.5 Conv2D

The hardest case. For an output `(oc, oy, ox)`:

```
dL/dW[oc][ic][ky][kx] += input[ic][iy][ix] * dL/doutput[oc][oy][ox]
```

where `iy, ix` depend on `oy, ox`. Sum over all output positions.

For the input gradient:

```
dL/dinput[ic][iy][ix] += W[oc][ic][ky][kx] * dL/doutput[oc][oy][ox]
```

Sum over all output positions.

The implementation mirrors the forward pass with the loops in reverse and the multiplications replaced by accumulations.

## 13.6 Gradient Checking

Before trusting your backprop implementation, verify it against numerical differentiation:

```
numerical_grad[i] ≈ (loss(params[i] + ε) - loss(params[i] - ε)) / (2ε)
```

Compare against the analytical gradient (from backprop). If they match within a small relative tolerance, the backprop is correct.

Typical tolerance: `1e-4` relative error. Small deviations are fine; large ones indicate a bug.

Gradient checking is a critical step — it is the single best way to catch backprop bugs, which are otherwise very hard to find.

## 13.7 A Complete Tiny Training Loop from Scratch

Before writing CNN backpropagation, implement a two-layer fully connected network on a tiny synthetic dataset.

Do not start with MNIST.

Use XOR.

The goal is to learn:

```
forward
loss
backward
update
repeat
```

Once XOR works, move to a tiny digit subset.

Only then attempt convolution backpropagation.

### 13.7.1 `python/from_scratch_xor.py`

```python
import numpy as np

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

rng = np.random.default_rng(0)

W1 = rng.normal(0, 0.5, (2, 4))
b1 = np.zeros((1, 4))

W2 = rng.normal(0, 0.5, (4, 1))
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

for step in range(10000):
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)

    z2 = a1 @ W2 + b2
    pred = sigmoid(z2)

    eps = 1e-7
    loss = -np.mean(
        y * np.log(pred + eps)
        + (1 - y) * np.log(1 - pred + eps)
    )

    dz2 = pred - y
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1 ** 2)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    lr = 0.1
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if step % 1000 == 0:
        print(step, loss)

print((pred > 0.5).astype(int).ravel())
```

Line by line:

- `import numpy as np` — Imports NumPy.
- `X = np.array([...])` — Creates the XOR inputs.
- `y = np.array([...])` — Creates the XOR labels.
- `rng = np.random.default_rng(0)` — Creates a deterministic random generator.
- `W1 = rng.normal(0, 0.5, (2, 4))` — Initializes the first layer weights.
- `b1 = np.zeros((1, 4))` — Initializes first-layer bias.
- `W2 = rng.normal(0, 0.5, (4, 1))` — Initializes second-layer weights.
- `b2 = np.zeros((1, 1))` — Initializes second-layer bias.
- `def sigmoid(x):` — Defines sigmoid.
- `return 1.0 / (1.0 + np.exp(-x))` — Converts a raw value into the range 0 to 1.
- `for step in range(10000):` — Runs gradient descent for 10,000 iterations.
- `z1 = X @ W1 + b1` — Computes the first linear transformation.
- `a1 = np.tanh(z1)` — Applies tanh as the hidden nonlinearity.
- `z2 = a1 @ W2 + b2` — Computes the second linear transformation.
- `pred = sigmoid(z2)` — Converts the output into a probability.
- `eps = 1e-7` — Adds a small epsilon for numerical stability.
- `loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))` — Computes binary cross entropy.
- `dz2 = pred - y` — Derivative of sigmoid-plus-binary-cross-entropy simplifies to prediction minus target.
- `dW2 = a1.T @ dz2` — Computes second-layer weight gradients.
- `db2 = dz2.sum(axis=0, keepdims=True)` — Computes second-layer bias gradients.
- `da1 = dz2 @ W2.T` — Propagates gradients into the hidden layer.
- `dz1 = da1 * (1 - a1 ** 2)` — Derivative of tanh is 1-a^2.
- `dW1 = X.T @ dz1` — Computes first-layer weight gradients.
- `db1 = dz1.sum(axis=0, keepdims=True)` — Computes first-layer bias gradients.
- `lr = 0.1` — Sets the learning rate.
- The update lines apply `parameter -= lr * gradient`.
- `if step % 1000 == 0: print(step, loss)` — Prints progress every 1000 iterations.
- `print((pred > 0.5).astype(int).ravel())` — Converts probabilities to binary predictions.

**Predict the final predictions before you run it.** They should be `[0, 1, 1, 0]`.

---

# Part XIV — Reference

## 14.1 The Complete File Listing

(For the mature version. Not all files exist yet.)

```
Number-Guesser/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── requirements.txt
├── python/
│   ├── dataset.py           ✓
│   ├── model.py             ✓
│   ├── train.py             ✓
│   ├── evaluate.py          ✓
│   ├── export.py            ✓
│   ├── dump_intermediate.py ✓
│   ├── helper_functions.py  ✓
│   ├── ocr/                 (future)
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── decode.py
│   └── tools/
│       ├── make_debug_input.py
│       └── compare.py
├── c/
│   ├── Makefile             ✓
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── nn.h             ✓
│   │   └── ui.h             ✓
│   ├── src/
│   │   ├── nn.c             ✓
│   │   ├── ui.c             ✓
│   │   └── main.c           ✓
│   └── tools/
│       ├── verify.c         ✓
│       └── eval_handwriting.c
├── tests/
│   ├── test_assert.h        (TARGET)
│   ├── test_nn.c            (TARGET)
│   ├── test_ui.c            (TARGET)
│   ├── test_relu.c          (TARGET)
│   ├── test_linear.c        (TARGET)
│   ├── test_pool.c          (TARGET)
│   ├── test_conv.c          (TARGET)
│   ├── tensor_index_demo.c  (TARGET)
│   ├── make_preprocessing_fixtures.py (TARGET)
│   └── fixtures/
│       ├── blank_28x28.bin
│       └── centered_7.bin
├── models/
│   ├── number_guesser_model.pth  ✓
│   ├── weights.bin               ✓
│   └── debug_input.bin           (TARGET)
├── data/
│   ├── MNIST/                    (downloaded)
│   └── handwriting/              (future)
│       ├── 0/
│       ├── 1/
│       └── ...
├── benchmark/
│   ├── input_000.bin
│   └── expected_000.bin
├── notes/
│   ├── pytorch_stages.txt
│   ├── c_stages.txt
│   └── expected_verify_output.txt
├── experiments/
│   ├── EXP-001.md
│   ├── EXP-002.md
│   └── ...
├── docs/
│   ├── C-IMPLEMENTATION-GUIDE.md
│   └── (this book)
└── .github/
    └── workflows/
        └── ci.yml
```

## 14.2 Mathematics Reference

### Convolution output shape

```
out = floor((N + 2P - K) / S) + 1
```

### Pooling output shape

```
out = floor((N - K) / S) + 1
```

### Linear layer

```
y = Wx + b
```

### ReLU

```
ReLU(x) = max(0, x)
```

### Softmax

```
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

Numerically stable:

```
softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

### Cross-entropy

```
L = -log(p_y)
```

### Gradient of cross-entropy with respect to logits

```
dL/dz_i = p_i - 1(i = y)
```

### Matrix multiplication

```
C = A @ B:  C[i][j] = sum_k A[i][k] * B[k][j]
```

### Dot product

```
y = x · w:  y = sum_i x[i] * w[i]
```

### Sigmoid

```
sigmoid(x) = 1 / (1 + exp(-x))
```

### Tanh

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

## 14.3 Numerical Reference

### float32 precision

- 24 bits of mantissa (roughly 7 decimal digits).
- `ε ≈ 1.2 × 10^-7` (machine epsilon).

### Accumulated rounding

For a sum of `N` terms, error ~ `N · ε · max|term|`.

For a dot product of 1568 terms, error ~ `1.9 × 10^-4 · max|term|`.

### Typical tolerances

- `1e-4` absolute for float32 comparisons.
- `1e-3` relative for float32 comparisons.

### How the error grows with depth

Each layer amplifies the previous layer's error by roughly the magnitude of its weights. In practice, for this model, error grows from about `2e-6` at conv1 to about `1e-4` at logits. If you see error growing much faster than that, you have a bug, not just rounding.

## 14.4 Debugging Playbook

**Compilation error** → check file, line, symbol, includes.

**Linker error** → check definition, library, name matching.

**Segfault** → run under ASan, check tensor shapes.

**Wrong prediction** → check input, weights, tensor shapes, layer order, flatten order, softmax.

**PyTorch vs. C mismatch** → find first divergent layer.

**Model trains poorly** → check learning rate, loss curve, overfitting.

**Model overfits** → add regularization, augmentation, or reduce capacity.

**Parity fails at conv1** → check weight order, kernel indexing, padding.

**Parity fails at pool1** → check `-INFINITY` initialization, window placement.

**Parity fails at logits** → check flatten order, FC weight layout.

**Input file is wrong size** → should be 784 floats × 4 bytes = 3136 bytes.

**Model file is wrong size** → should be 175016 bytes.

## 14.5 Glossary

**Tensor**: a runtime-sized, heap-allocated float buffer with shape metadata (channels, height, width).

**Channel-first layout**: memory layout where all of channel 0 comes first, then all of channel 1, etc. Also called C-contiguous or row-major with C as the second-fastest-varying dimension.

**Logits**: raw scores output by the final linear layer, before softmax.

**Softmax**: converts logits into probabilities, invariant to adding a constant to all logits.

**ReLU**: `max(0, x)`, element-wise.

**MaxPool**: keeps the maximum value in each k×k window.

**Forward pass**: the sequence of operations that turns an input into an output.

**Backprop**: the algorithm that computes gradients of the loss with respect to each parameter.

**Parity**: the property that two implementations produce the same result, within tolerance.

**MNIST**: a dataset of 28×28 grayscale handwritten digits.

**Domain shift**: the difference between the training distribution and the inference distribution.

**Segmentation**: separating an image into individual components (e.g., individual digits).

**Connected components**: groups of adjacent pixels of the same value.

**CTC**: Connectionist Temporal Classification, a loss and decoding scheme for sequence models.

**Greedy decoding**: argmax at each position, then collapse.

**Quantization**: representing weights or activations with fewer bits.

**Sanitizer**: a compiler feature that instruments code to catch runtime bugs.

**ASan**: AddressSanitizer.

**UBSan**: UndefinedBehaviorSanitizer.

---

# Part XV — Long-Term Roadmap

## 15.1 Milestones, in Order

**Milestone 1**: real-weight verification. The C inference matches PyTorch on real weights.

**Milestone 2**: real handwriting accuracy measured. A number on your own drawings.

**Milestone 3**: experiment harness built. "I changed the model" becomes evidence.

**Milestone 4**: observability (activation viz). You can see what the network sees.

**Milestone 5**: model improvement (preprocessing, augmentation). Measured improvement.

**Milestone 6**: performance engineering (profiling, optimization). You know where the milliseconds go.

**Milestone 7**: automated quality gates (sanitizer targets, CI). Regressions are caught automatically.

**Milestone 8**: two-digit recognition (segmentation + single-digit CNN). A measured baseline.

**Milestone 9**: variable-length OCR (sequence model, CTC). Arbitrary-length numbers.

**Milestone 10**: C implementation of OCR inference. The OCR model runs in C.

**Milestone 11**: backprop (optional). You understand what PyTorch does.

**Milestone 12**: research extensions (quantization, distillation, etc.).

## 15.2 What "Done" Means at Each Level

**Level 1 — Correctness**: PyTorch and C agree within tolerance at every layer, on real weights.

**Level 2 — Usability**: works on drawings from the actual application.

**Level 3 — Measured ML quality**: accuracy, confusion matrix, per-class metrics, on a real dataset.

**Level 4 — Explainability**: activations, logits, calibration, all viewable.

**Level 5 — Engineering quality**: tests, sanitizers, CI, profiling, documented.

**Level 6 — Generalization**: works on data it was not trained on.

**Level 7 — Sequence recognition**: handles arbitrary-length numbers.

## 15.3 What Not to Do

- Do not restart from scratch.
- Do not add Docker before you need it.
- Do not add CI before you have meaningful regression tests.
- Do not jump to Transformers before solving segmentation.
- Do not train 100-class classifiers for 2-digit numbers.
- Do not optimize before profiling.
- Do not change five things at once.
- Do not trust accuracy alone.
- Do not trust confidence blindly.
- Do not assume MNIST accuracy means handwriting accuracy.
- Do not assume the model is correct because it compiles.
- Do not assume it is correct because it predicts something.
- Do not assume the C implementation is correct because it matches one output.

---

# Part XVI — Labs

Each lab is a small, self-contained experiment. Run it, change one value, predict the result, run it again. This is how the material becomes yours.

## 16.1 C Array Lab

Create `lab_arrays.c`:

```c
#include <stdio.h>

int main(void) {
    float values[3] = {1.0f, 2.0f, 3.0f};

    for (int i = 0; i < 3; ++i) {
        printf("%f\n", values[i]);
    }

    return 0;
}
```

Line by line:

- `#include <stdio.h>` — Imports a C standard-library header.
- `int main(void) {` — Program entry point.
- `float values[3] = {1.0f, 2.0f, 3.0f};` — Creates a fixed-size array.
- `for (int i = 0; i < 3; ++i) {` — Starts a loop over the array.
- `printf("%f\n", values[i]);` — Prints the current value.
- `return 0;` — Returns success.

**Change one value and predict the result before running again.**

## 16.2 C Struct Lab

Create `lab_struct.c`:

```c
#include <stdio.h>

typedef struct {
    int width;
    int height;
} Shape;

int main(void) {
    Shape s = {28, 28};
    printf("%d x %d\n", s.width, s.height);
    return 0;
}
```

Line by line:

- `typedef struct { ... } Shape;` — Defines a struct type.
- `Shape s = {28, 28};` — Creates and initializes a struct value.
- `printf("%d x %d\n", s.width, s.height);` — Accesses fields via `.`.

**Change one field and predict the output.**

## 16.3 C Heap Lab

Create `lab_heap.c`:

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float *data = calloc(10, sizeof(float));

    if (data == NULL) {
        return 1;
    }

    data[3] = 7.0f;
    printf("%f\n", data[3]);

    free(data);
    return 0;
}
```

Line by line:

- `#include <stdlib.h>` — Imports `calloc`, `free`.
- `float *data = calloc(10, sizeof(float));` — Allocates 10 floats, zero-initialized.
- `if (data == NULL) { return 1; }` — Checks for allocation failure.
- `data[3] = 7.0f;` — Writes through the pointer.
- `printf("%f\n", data[3]);` — Reads through the pointer.
- `free(data);` — Frees the memory.

**Change the index and predict the output.**

## 16.4 C File Lab

Create `lab_file.c`:

```c
#include <stdio.h>

int main(void) {
    FILE *f = fopen("example.bin", "wb");

    if (f == NULL) {
        return 1;
    }

    float value = 3.14f;
    fwrite(&value, sizeof(value), 1, f);

    fclose(f);
    return 0;
}
```

Line by line:

- `FILE *f = fopen("example.bin", "wb");` — Opens a file for writing in binary mode.
- `if (f == NULL) { return 1; }` — Checks for failure.
- `float value = 3.14f;` — Creates a value to write.
- `fwrite(&value, sizeof(value), 1, f);` — Writes the bytes of the value.
- `fclose(f);` — Closes the file.

**Check the file size with `ls -l example.bin`.** It should be 4 bytes.

## 16.5 Python NumPy Lab

Create `lab_numpy.py`:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, 0.5, 0.5])
b = 1.0

y = x @ w + b

print(y)
```

Line by line:

- `import numpy as np` — Imports NumPy.
- `x = np.array([1.0, 2.0, 3.0])` — Creates a 1D array.
- `w = np.array([0.5, 0.5, 0.5])` — Creates another 1D array.
- `b = 1.0` — A scalar.
- `y = x @ w + b` — Dot product plus bias.
- `print(y)` — Prints the result.

**Predict the output before running.** `1*0.5 + 2*0.5 + 3*0.5 + 1 = 0.5 + 1 + 1.5 + 1 = 4.0`.

## 16.6 Python Shape Lab

Create `lab_shape.py`:

```python
import torch

x = torch.zeros(2, 3, 4)

print(x.shape)
print(x.numel())
```

**Predict both outputs before running.** Shape `(2, 3, 4)`, numel = 24.

## 16.7 Python ReLU Lab

Create `lab_relu.py`:

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 3.0])
print(torch.relu(x))
```

**Predict the output before running.** `[-2, -1, 0, 3]` → `[0, 0, 0, 3]`.

## 16.8 Python Softmax Lab

Create `lab_softmax.py`:

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.0])
probs = torch.softmax(logits, dim=0)

print(probs)
print(probs.sum())
```

**Predict the output before running.** Softmax of `[2, 1, 0]` is approximately `[0.6652, 0.2447, 0.0900]`, and it sums to 1.

## 16.9 Python Gradient Lab

Create `lab_gradient.py`:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * x
y.backward()

print(x.grad)
```

**Predict the output before running.** `d(x²)/dx = 2x`, so at `x=3`, grad = 6.

## 16.10 Python Conv Lab

Create `lab_conv.py`:

```python
import torch
from torch import nn

layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)

x = torch.zeros(1, 1, 28, 28)
y = layer(x)

print(y.shape)
```

**Predict the output before running.** With padding=1 and kernel=3, spatial dims are preserved, so `(1, 2, 28, 28)`.

---

# Part XVII — Mastery Drills

These drills exist to build fluency. Do not rush. For each question, write the answer in full, then compare against the book.

## Round 1

### Question
What is a pointer?

**Answer framework:**

1. Define the concept in one sentence.
2. Write the mathematical form if one exists.
3. Give a tiny numeric example.
4. Point to where it appears in Number Guesser.
5. Name one bug that would happen if it were implemented incorrectly.
6. Write one test that could detect that bug.

**Sample answer**: A pointer is a variable whose value is a memory address.

- Math: `p : address`, and `*p` is the value stored at that address.
- Example: `int x = 42; int *p = &x; *p` is `42`.
- In Number Guesser: `Tensor.data` is a `float *`.
- Bug: if `tensor_free` did not NULL the pointer, a subsequent `tensor_get` would dereference freed memory.
- Test: after `tensor_free(&t)`, check `t->data == NULL`.

### Question
What is the difference between an array and a pointer?

**Sample answer**: An array is a contiguous block of elements; a pointer is a variable that holds an address. In an expression, an array name *decays* to a pointer to its first element, but the array object itself is not a pointer.

- Math: `arr[i] ≡ *(arr + i)`.
- Example: `int a[4]; int *p = a;` — `p` and `a` both point to the first element, but `sizeof(a) == 16` while `sizeof(p) == 8`.
- In Number Guesser: `float conv1_w[288]` is an array; `Tensor.data` is a pointer.
- Bug: returning a pointer to a local array (dangling pointer).
- Test: ASan catches use-after-return.

### Question
Why does array indexing work through pointer arithmetic?

**Sample answer**: Because `arr[i]` is *defined* as `*(arr + i)`, and `arr + i` moves by `i * sizeof(element)` bytes.

- Math: address of `arr[i]` is `base + i * sizeof(T)`.
- Example: `int a[4] = {10,20,30,40}; *(a + 2) == 30`.
- In Number Guesser: `tensor_get` computes a flat index, then indexes `t->data[index]`, which is the same idea at runtime.
- Bug: incorrect element size (e.g., casting `float*` to `int*`) would move by the wrong stride.
- Test: check that `tensor_get(t, c, y, x)` matches manual `(c*H + y)*W + x`.

### Question
Why does Tensor store a float pointer?

**Sample answer**: Because the size of the tensor is not known until runtime, so the data must live on the heap, and a pointer is the handle to that heap memory.

- Math: `Tensor = (float *, int, int, int)`.
- Example: a 32×28×28 tensor has 25,088 floats, allocated via `calloc`.
- In Number Guesser: `tensor_alloc` returns a `Tensor` with `data` pointing to a fresh `calloc`.
- Bug: forgetting to `free(t->data)` leaks memory.
- Test: run the test suite under ASan with leak detection enabled.

### Question
Why is channel-first ordering important?

**Sample answer**: Because it must match PyTorch's default `[C, H, W]` layout for the binary weight file to be a straight byte copy.

- Math: `offset(c, y, x) = (c * H + y) * W + x`.
- Example: a `(2, 3, 3)` tensor's `(1, 2, 0)` element is at flat index 15.
- In Number Guesser: `tensor_get` and `tensor_set` both use this formula.
- Bug: writing `(c * H + y) * (W + x)` shifts every element by `x` whole rows, silently corrupting adjacent cells.
- Test: "neighbor unaffected" checks in `test_tensor`.

### Question
What is a convolution kernel?

**Sample answer**: A small matrix of weights that is slid across the input, computing a dot product at each position.

- Math: `output[oc][oy][ox] = sum_{ic, ky, kx} input[ic][iy][ix] * W[oc][ic][ky][kx]`.
- Example: a 3×3 kernel over a 3×3 input produces a 1×1 output (without padding).
- In Number Guesser: `conv2d` takes `const float *weights` and loops over `oc, oy, ox, ic, ky, kx`.
- Bug: transposed weight index produces wrong sums.
- Test: hand-computed conv on a 3×3 input with a 2×2 kernel.

### Question
Why does padding preserve spatial size here?

**Sample answer**: Because for K=3, P=1, S=1, the output shape formula gives `N` exactly.

- Math: `out = floor((N + 2P - K) / S) + 1 = N + 2 - 3 + 1 = N`.
- Example: `N=28` → `out=28`.
- In Number Guesser: this is why every conv in the model preserves spatial size.
- Bug: using P=0 would shrink the tensor by 2 per conv, breaking the `1568` flatten size.
- Test: check output shape from `conv2d(&input, ..., k=3, stride=1, pad=1)` matches input shape.

### Question
Why does MaxPool reduce 28 to 14?

**Sample answer**: Because pooling with K=2, S=2 halves the spatial dimension.

- Math: `out = floor((N - K) / S) + 1 = floor((28-2)/2) + 1 = 14`.
- Example: same formula, `N=14` → `out=7`.
- In Number Guesser: two maxpools bring 28×28 to 7×7.
- Bug: wrong sentinel (`0.0f` instead of `-INFINITY`) silently miscomputes on all-negative windows.
- Test: pool a 2×2 window of `[-3, -5; -2, -7]` and check the answer is `-2`.

### Question
Why is the flattened size 1568?

**Sample answer**: Because after two maxpools, the tensor is `32 × 7 × 7`, and `32 * 7 * 7 = 1568`.

- Math: `C * H * W = 32 * 7 * 7 = 1568`.
- Example: this is the input size of the FC layer.
- In Number Guesser: `int in_features = p2.channels * p2.height * p2.width;`.
- Bug: changing pooling or conv channels changes this number; forgetting to update `fc_w`'s shape or `LAYER_KEYS` breaks the contract.
- Test: assert `p2.channels * p2.height * p2.width == 1568`.

### Question
Why does the final layer have 10 outputs?

**Sample answer**: Because MNIST has 10 digit classes (0–9).

- Math: `output_shape = 10`.
- Example: logits shape `(batch, 10)`.
- In Number Guesser: `nn.Linear(flatten_size, output_shape)` with `output_shape=10`.
- Bug: changing to 11 without retraining produces garbage (the weights would not match).
- Test: check `fc_b` has 10 elements.

### Question
What is a logit?

**Sample answer**: A raw score output by the final linear layer, before any softmax.

- Math: `z = W x + b`.
- Example: `[2.1, -0.5, 4.7, ...]`.
- In Number Guesser: `model_forward(..., logits)` writes these raw scores.
- Bug: applying softmax *before* the final layer would give wrong values (softmax is only meaningful after the last linear layer).
- Test: check `argmax(logits) == argmax(softmax(logits))`.

### Question
Why is softmax applied after the final linear layer?

**Answer**: Because the linear layer produces raw scores, and softmax turns them into a probability distribution.

### Question
Why subtract the maximum logit?

**Answer**: To prevent overflow in `exp` without changing the result (softmax is invariant to adding a constant).

### Question
What is cross-entropy?

**Answer**: `L = -log(p_y)` where `p_y` is the probability assigned to the correct class.

### Question
What is a gradient?

**Answer**: The vector of partial derivatives of the loss with respect to each parameter.

### Question
Why subtract the gradient during SGD?

**Answer**: Because the gradient points in the direction of increasing loss; subtracting moves against it, toward lower loss.

### Question
What does the learning rate control?

**Answer**: The step size for each parameter update.

### Question
Why can two correct floating-point programs differ slightly?

**Answer**: Because floating-point operations are not associative; different orderings of the same sums give slightly different results.

### Question
What is numerical parity?

**Answer**: The property that two implementations produce results that agree within tolerance.

### Question
Why is the first divergent layer important?

**Answer**: Because everything before it is correct; the bug is in the layer that first diverges or the layer right before it.

### Question
Why can preprocessing break an otherwise good model?

**Answer**: Because the model was trained on one input distribution and is being tested on another; preprocessing is the mechanism to bring them into alignment.

### Question
Why should tests be deterministic?

**Answer**: So that a failure can be reproduced exactly, and so that a pass means the same thing every time.

### Question
Why use sanitizers?

**Answer**: Because they catch memory bugs that would otherwise be silent until they corrupt data or crash in production.

### Question
Why profile before optimizing?

**Answer**: Because intuition about performance is often wrong, and optimizing the wrong thing wastes effort.

### Question
Why is a model file an API?

**Answer**: Because two systems agree on its format, and changing the format on one side without the other silently breaks the contract.

### Question
Why should the serialization format have a version?

**Answer**: So that a loader can detect incompatible files and give a clear error instead of producing garbage.

### Question
Why are connected components useful for OCR?

**Answer**: Because they give a natural way to separate individual digits without needing a learned detector.

### Question
Why can two touching digits defeat projection segmentation?

**Answer**: Because if their strokes overlap, the projection profile never drops to zero between them.

### Question
Why is sequence OCR different from classification?

**Answer**: Because the output is a variable-length sequence, not a single label from a fixed set.

### Question
What problem does CTC solve?

**Answer**: It aligns a fixed-length sequence of per-position predictions with a variable-length target sequence, by marginalizing over all possible alignments.

### Question
Why should C backpropagation be attempted only after inference is stable?

**Answer**: Because if you cannot verify the forward pass, you cannot verify the backward pass, and backprop bugs are much harder to find.

## Rounds 2–120

Rounds 2 through 120 repeat the same 30 questions with the same answer framework. The repetition is the point: each round should take less time, and you should need to consult the book less and less. By round 120, you should be able to answer all 30 from memory, with numeric examples, in under five minutes.

The rounds are identical in structure. Do them. Do not skip them because they "feel repetitive." The repetition is the training.

*(For brevity, the full text of rounds 2–120 is identical to Round 1. The intent is: 30 questions × 120 rounds = 3,600 reps. Mastery comes from the reps, not from reading the questions once.)*

---

# Part XVIII — The Study Contract

## 18.1 How to Read This Book

Do not read this book passively.

For every implementation section, use this loop:

```
1. Read the objective.
2. Read the "why".
3. Inspect the current file.
4. Create the exact file requested.
5. Paste the smallest working version.
6. Read every line.
7. Compile.
8. Run the focused test.
9. Intentionally understand at least one failure mode.
10. Commit only after verification.
11. Write a short Obsidian note.
12. Continue.
```

Paper is for derivations.
Code is for experiments.
Obsidian is for durable knowledge.

A useful rule:

> If you cannot explain a line, do not hide it behind an abstraction.

## 18.2 The Workflow

The repository is the laboratory.
The code is the experiment.
The math is the explanation.
The tests are the proof.

```
READ
↓
WRITE CODE
↓
COMPILE
↓
TEST
↓
BREAK IT
↓
DEBUG
↓
MEASURE
↓
EXPLAIN
↓
COMMIT
↓
NEXT
```

## 18.3 The Final Rulebook

Never forget these:

### Rule 1
If the code works but you cannot explain it, you do not own the code yet.

### Rule 2
If Python and C disagree, find the first layer where they disagree.

### Rule 3
If a test fails, do not edit five files.

### Rule 4
If performance is slow, profile first.

### Rule 5
If handwriting fails, inspect preprocessing before changing the CNN.

### Rule 6
If memory is corrupted, run sanitizers.

### Rule 7
If a model file is binary, treat its layout as an API.

### Rule 8
If you change architecture, update:
- Python model
- exporter
- C struct
- C loader
- C forward pass
- tests
- benchmark
- documentation

### Rule 9
Do not confuse "same predicted digit" with numerical parity.

### Rule 10
Build understanding in layers.

```
C fundamentals
→ tensor memory
→ neural-network math
→ inference
→ parity
→ tests
→ preprocessing
→ systems engineering
→ optimization
→ OCR
→ backpropagation
```

That is the complete learning path for this project.

---

# Part XIX — Closing

## 19.1 What "Done" Means

When this project is done — really done, at the level this book describes — you will be able to say:

> I trained a CNN in PyTorch from scratch. I exported its weights to a documented binary format. I implemented the entire inference engine in C, by hand, without ML libraries. I verified that every layer matches PyTorch's output within numerical tolerance, on real weights. I built a Raylib application that draws digits and predicts them. I collected a dataset of my own handwriting and measured the model's accuracy on it. I diagnosed the failures. I improved the preprocessing, and I measured the improvement. I built a segmentation pipeline for two-digit recognition, and I measured its accuracy. I studied CTC and built a sequence model for arbitrary-length numbers. And I understand every operation, in both languages, at the level of arithmetic.

That is a rare thing. Most people who use machine learning cannot explain it. Most people who write C cannot explain convolutions. Most people who do both cannot build the whole pipeline from dataset to deployment.

The goal of this book is to make you one of the people who can.

## 19.2 The Point of This Project

This project is not just about recognizing digits. It is a laboratory for learning:

- Python
- C
- memory
- arrays
- pointers
- structs
- files
- binary serialization
- linear algebra
- convolution
- neural networks
- gradients
- backpropagation
- optimization
- probability
- evaluation
- numerical parity
- testing
- debugging
- sanitizers
- build systems
- CI
- profiling
- performance
- computer vision
- OCR
- sequence modeling

Every one of these is something you will have built with your hands and proved with a test. That is the difference between having read about machine learning and having *done* machine learning.

---

## Appendix A — The One-Page Summary

If you only remember one page, remember this:

```
PYTORCH                       C
─────────                     ────────
model.py                      nn.c
  Conv2d(1,32,3,p=1)   →       conv2d(input, conv1_w, conv1_b, 32, 3, 1, 1)
  ReLU                 →       relu_tensor(&a)
  Conv2d(32,32,3,p=1)  →       conv2d(&a, conv2_w, conv2_b, 32, 3, 1, 1)
  ReLU                 →       relu_tensor(&b)
  MaxPool2d(2,2)       →       maxpool2d(&b, 2, 2)
  Conv2d(32,32,3,p=1)  →       conv2d(&p1, conv3_w, conv3_b, 32, 3, 1, 1)
  ReLU                 →       relu_tensor(&c)
  Conv2d(32,32,3,p=1)  →       conv2d(&c, conv4_w, conv4_b, 32, 3, 1, 1)
  ReLU                 →       relu_tensor(&d)
  MaxPool2d(2,2)       →       maxpool2d(&d, 2, 2)
  Flatten              →       (implicit in p2.data)
  Linear(1568,10)      →       linear(fc_w, fc_b, p2.data, logits, 1568, 10)

CONTRACT:
  - Channel-first layout: offset(c,y,x) = (c*H + y)*W + x
  - Weight order: [out, in, ky, kx]
  - Dtype: float32
  - File size: 175016 bytes

VERIFICATION:
  - Run both implementations on the same input
  - Dump every intermediate tensor
  - Find the first divergence
  - Fix the layer that diverges
  - Re-run until all layers pass

THE RULE:
  Do not claim done until you have the command that proves it.
```

---

## Appendix B — Quick Reference Card

### Build

```bash
# Python
python python/train.py
python python/evaluate.py
python python/export.py

# C
cd c && make test
cd c && make app
cd c && make verify
```

### Verify

```bash
python python/dump_intermediate.py > notes/py_stages.txt
cd c/tools && ./verify ../models/weights.bin ../models/debug_input.bin > ../../notes/c_stages.txt
python tools/compare.py
```

### Sizes

```
weights.bin: 175016 bytes
debug_input.bin: 3136 bytes (784 floats × 4)
```

### Shapes

```
Input:   (1, 28, 28)
conv1:   (32, 28, 28)
conv2:   (32, 28, 28)
pool1:   (32, 14, 14)
conv3:   (32, 14, 14)
conv4:   (32, 14, 14)
pool2:   (32, 7, 7)
flat:    (1568,)
logits:  (10,)
```

### Tolerances

```
Layer-by-layer: 1e-4 absolute
```

### Key Files

```
python/model.py           — the CNN
python/export.py          — writes weights.bin
c/include/nn.h            — the C API
c/src/nn.c                — the C implementation
c/src/ui.c                — the canvas
c/src/main.c              — the Raylib app
c/tools/verify.c          — the C dump tool
tests/test_assert.h       — the assertion macros
```

---

**End of the master reference.**

This document is intentionally open-ended. There are sections that will grow as the project does (the OCR parts especially), and there are sections that will need revising as you learn more. That is fine. The book is a living document, and it should reflect the project's actual state, not some idealized finished version.

If you work through this book — really work through it, running every command, breaking every piece of code, measuring every result — you will know more about how machine learning actually works, at the level of arithmetic and memory, than most people who use it professionally.

That is the point.