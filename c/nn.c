#include "nn.h"

void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features) {
    for (int o = 0; o < out_features; o++) {
        float sum = b[o];
        const float *row = W + o * in_features;  /* pointer arithmetic: */
                                                   /* start of row o */
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
