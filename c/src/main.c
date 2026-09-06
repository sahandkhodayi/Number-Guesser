#include <stdio.h>
#include "nn.h"

/*
 * Hand-verified test case:
 *
 * x = [1, 2, 3]
 * W = [[ 1.0,  0.0, -1.0],      (row 0 -> output 0)
 *      [ 0.5,  0.5,  0.5]]      (row 1 -> output 1)
 * b = [0.0, -1.0]
 *
 * y0 = 1*1 + 0*2 + -1*3 + 0  = -2
 * y1 = 0.5*1 + 0.5*2 + 0.5*3 - 1 = 2
 *
 * after relu: [0, 2]
 * argmax:     1
 */
static void test_linear_relu_argmax(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};

    /* Flat, row-major: W[o * in_features + i] */
    float W[2 * 3] = {
        1.0f,  0.0f, -1.0f,
        0.5f,  0.5f,  0.5f
    };
    float b[2] = {0.0f, -1.0f};
    float y[2];

    linear(W, b, x, y, /*in_features=*/3, /*out_features=*/2);
    printf("linear output: [%.2f, %.2f] (expected [-2.00, 2.00])\n", y[0], y[1]);

    relu(y, 2);
    printf("after relu:    [%.2f, %.2f] (expected [0.00, 2.00])\n", y[0], y[1]);

    int pred = argmax(y, 2);
    printf("argmax:        %d (expected 1)\n", pred);
}

int main(void) {
    test_linear_relu_argmax();
    return 0;
}
