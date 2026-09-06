#include "../include/nn.h"

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


Tensor tensor_alloc(int channels, int height, int width) {
    Tensor t;
    t.channels = channels;
    t.height = height;     // our tesnore info 
    t.width = width;
    size_t n = (size_t)channels * height * width;
    t.data = calloc(n, sizeof(float)); // creating a heap for our data
    if (t.data == NULL) {
        fprintf(stderr, "tensor_alloc: calloc failed for %d x %d x %d\n",
                channels, height, width);
        exit(1); // checking for our errors
    }
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data); // free or clearing our buffer 
    t->data = NULL;
    t->channels = t->height = t->width = 0; // deleting our tensore 
}


float tensor_get(const Tensor *t , int c, int y , int x){

    return t->data[(c* t->height + y ) * (t->width + x)]; // wtf?



} 


void tesnor_set(Tensor *t, int c, int y , int x , float value){

    t->data[(c* t->height + y ) * (t->width + x)] = value; // setting a value at xy postion with a value!


}



void tensor_info(const Tensor *t , const char *label){ //label (str)
    
    printf("%s: [%d, %d, %d]", label, t->channels, t->height, t->width);// Tensor structore info 
    
    int n = t->channels * t->height * t->width;
    
    int show = n < 5 ? n : 5;// if n <5 then n if not n = 5
    
    
    printf("  first %d values: [", show);
    
    for (int i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], (i == show - 1) ? "" : ", "); // printing some datas as a n example
    }
    printf("]\n");

}


void relu_tensor(Tensor *t){ // activation function for tensor 

    relu(t->data, t->channels * t->width * t->height)

}       // data                position of that data in structore