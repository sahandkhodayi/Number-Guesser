#ifndef NN_H
#define NN_H

/*
 * Fully-connected layer: y = Wx + b
 *
 * W: flat [out_features * in_features] array, row-major.
 *    Row o (weights for output o) starts at W[o * in_features].
 * b: [out_features] array.
 * x: [in_features] input.
 * y: [out_features] output buffer — caller allocates it, this function
 *    only writes into it. Nothing here is freed or owned by linear().
 */
void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);

/* In-place ReLU: x[i] = max(0, x[i]) for i in [0, n). */
void relu(float *x, int n);

/* Index of the largest value in x[0..n). */
int argmax(const float *x, int n);

#endif
