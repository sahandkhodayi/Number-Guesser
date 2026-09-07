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

    relu(t->data, t->channels * t->width * t->height);

}       // data                position of that data in structore


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
                            int iy = oy * stride - pad + ky; // wtf my eyes nigga holy shit 
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






Tensor maxpool2d(const Tensor *input, int k, int stride) {
    int out_h = (input->height - k) / stride + 1;
    int out_w = (input->width  - k) / stride + 1;        // our output block or Tensors info
    Tensor out = tensor_alloc(input->channels, out_h, out_w);

    
    
    for (int c = 0; c < input->channels; c++) { // loop for every rgb or conv2d dims
        
        for (int oy = 0; oy < out_h; oy++) { // every row 
            
            
            for (int ox = 0; ox < out_w; ox++) { // ever column
                
                
                float best = -1e30f; // sentinel
                
                
                
                for (int ky = 0; ky < k; ky++) {
                    for (int kx = 0; kx < k; kx++) {  /*        our main winodws 
                                                                                        */
                        int iy = oy * stride + ky;
                        int ix = ox * stride + kx; // cordiante calculation
                        float v = tensor_get(input, c, iy, ix); // getting the value
                        if (v > best) best = v; // if it is bigger than our temp then it is the max in that window 

                    }
                }
                tensor_set(&out, c, oy, ox, best); // we set that cordinate compare to our output tensor
            }
        }
    }
    return out;
}




// Final step our forward pass 


// input is a Tensor (1x28x28) ---> 10 logits (classes)


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

    int in_features = p2.channels * p2.height * p2.width; // 1568
    linear(m->fc_w, m->fc_b, p2.data, logits_out, in_features, 10);
    tensor_free(&p2);
}