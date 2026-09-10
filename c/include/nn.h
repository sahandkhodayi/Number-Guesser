#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#define WEIGHTS_FILE_BYTES 175016
/*
 * C inference model.
 *
 * IMPORTANT: this struct mirrors the exact tensor shapes exported by
 * python/export.py. C is only doing inference; training remains in PyTorch.
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
