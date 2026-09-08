#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WEIGHTS_FILE_BYTES 175016



typedef struct {
    float conv1_w[288],  conv1_b[32];
    float conv2_w[9216], conv2_b[32];
    float conv3_w[9216], conv3_b[32];
    float conv4_w[9216], conv4_b[32];
    float fc_w[15680],   fc_b[10];
} CnnModel;




typedef struct 
{
    float *data;
    int channels;       /* our Tensor structore*/
    int height;
    int width;
}Tensor ;

void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);
void relu(float *x, int n);
int argmax(const float *x, int n);



int model_load(CnnModel *m, const char *path);
void model_forward(const CnnModel *m, const Tensor *input, float *logits_out);



Tensor tensor_alloc(int channels,int height,int width);

float tensor_get(const Tensor *t,int c,int y, int x );

void tensor_set(Tensor *t,int c,int y, int x,float value ); // why not const??

void tensor_info(const Tensor *t,const char *label);

void relu_tensor(Tensor *t); // why pointer to tensor?

Tensor conv2d(const Tensor *input, const float *weights, const float *bias,
              int out_channels, int k, int stride, int pad); // our converter with filter !!


Tensor maxpool2d(const Tensor *input, int k, int stride); // 9x9 --> 3x3


void model_forward(const CnnModel *m, const Tensor *input, float *logits_out);
void tensor_free(Tensor *t);


int model_load(CnnModel *m, const char *path);

#endif

