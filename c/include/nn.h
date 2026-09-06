#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


void linear(const float *W, const float *b, const float *x, float *y,
            int in_features, int out_features);
void relu(float *x, int n);
int argmax(const float *x, int n);



typedef struct 
{
    float *data;
    int channels;       /* our Tensor structore*/
    int height;
    int width;
}Tensor ;

Tensor tensor_alloc(int channels,int height,int width);

float tensor_get(const Tensor *t,int c,int y, int x );

void tensor_set(Tensor *t,int c,int y, int x,float value ); // why not const??

void tensor_info(const Tensor *t,const char *label);

void relu_tensor(Tensor *t) // why pointer to tensor?


#endif

