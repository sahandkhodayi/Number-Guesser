#include "../include/nn.h"
#include <stdio.h>

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
    const char *weights_path = argc > 1 ? argv[1] : "../../models/weights.bin";
    const char *input_path   = argc > 2 ? argv[2] : "models/debug_input.bin";

    CnnModel m;
    if (model_load(&m, weights_path) != 0) return 1;

    Tensor input = tensor_alloc(1, 28, 28);
    FILE *f = fopen(input_path, "rb");
    if (f != NULL) {
        fread(input.data, sizeof(float), 28 * 28, f);
        fclose(f);
    } /* else: leave it zeroed */
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
    linear(m.fc_w, m.fc_b, p2.data, logits, p2.channels * p2.height * p2.width, 10);
    tensor_free(&p2);

    printf("logits   shape=(1, 10)  first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           logits[0], logits[1], logits[2], logits[3], logits[4]);
    printf("\npredicted digit: %d\n", argmax(logits, 10));
    return 0;
}