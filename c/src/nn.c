#include "../include/nn.h"

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

void relu(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

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

void tensor_free(Tensor *t) {
    free(t->data);
    t->data = NULL;
    t->channels = t->height = t->width = 0;
}

float tensor_get(const Tensor *t, int c, int y, int x) {
    size_t index = ((size_t)c * (size_t)t->height + (size_t)y) *
                   (size_t)t->width + (size_t)x;
    return t->data[index];
}

void tensor_set(Tensor *t, int c, int y, int x, float value) {
    size_t index = ((size_t)c * (size_t)t->height + (size_t)y) *
                   (size_t)t->width + (size_t)x;
    t->data[index] = value;
}

void tensor_info(const Tensor *t, const char *label) {
    int n = t->channels * t->height * t->width;
    int show = n < 5 ? n : 5;
    printf("%s: [%d, %d, %d] first %d values: [",
           label, t->channels, t->height, t->width, show);
    for (int i = 0; i < show; ++i) {
        printf("%.6f%s", t->data[i], i == show - 1 ? "" : ", ");
    }
    printf("]\n");
}

void relu_tensor(Tensor *t) {
    relu(t->data, t->channels * t->height * t->width);
}

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

static int read_floats(FILE *f, float *dst, size_t count, const char *what) {
    size_t n = fread(dst, sizeof(float), count, f);
    if (n != count) {
        fprintf(stderr,
                "model_load: short read on %s (got %zu of %zu floats)\n",
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

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long file_size = ftell(f);
    if (file_size < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

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
    err |= read_floats(f, m->fc_w, sizeof m->fc_w / sizeof m->fc_w[0], "fc_w");
    err |= read_floats(f, m->fc_b, sizeof m->fc_b / sizeof m->fc_b[0], "fc_b");

    fclose(f);
    return err ? -1 : 0;
}
