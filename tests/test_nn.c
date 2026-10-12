

#include "../c/include/nn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_tensor_set_get(void) {
    Tensor t = tensor_alloc(2, 3, 4);
    assert(t.data != NULL);

    tensor_set(&t, 1, 2, 3, 42.5f);
    assert(fabsf(tensor_get(&t, 1, 2, 3) - 42.5f) < 1e-6f);

    /* A separate channel must not overwrite this element. */
    assert(tensor_get(&t, 0, 2, 3) == 0.0f);

    tensor_free(&t);
    assert(t.data == NULL);
    assert(t.channels == 0);
    assert(t.height == 0);
    assert(t.width == 0);
}

static void test_relu(void) {
    float values[] = {-3.0f, 0.0f, 2.5f, -0.25f};
    relu(values, 4);

    assert(values[0] == 0.0f);
    assert(values[1] == 0.0f);
    assert(values[2] == 2.5f);
    assert(values[3] == 0.0f);
}

static void test_argmax(void) {
    float values[] = {-2.0f, 0.5f, 9.0f, 3.0f};
    assert(argmax(values, 4) == 2);
}

static void test_linear(void) {
    /*
       W = [1 2]    b = [10]    x = [5]
           [3 4]        [20]        [6]

       y0 = 1*5 + 2*6 + 10 = 27
       y1 = 3*5 + 4*6 + 20 = 59
    */
    const float W[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {10.0f, 20.0f};
    const float x[] = {5.0f, 6.0f};
    float y[2] = {0.0f, 0.0f};

    linear(W, b, x, y, 2, 2);

    assert(fabsf(y[0] - 27.0f) < 1e-6f);
    assert(fabsf(y[1] - 59.0f) < 1e-6f);
}

static void test_maxpool(void) {
    Tensor input = tensor_alloc(1, 4, 4);
    const float values[16] = {
        1, 8, 2, 4,
        3, 5, 7, 6,
        9, 0, 1, 2,
        4, 3, 8, 5
    };

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            tensor_set(&input, 0, y, x, values[y * 4 + x]);
        }
    }

    Tensor output = maxpool2d(&input, 2, 2);
    assert(output.data != NULL);
    assert(output.channels == 1);
    assert(output.height == 2);
    assert(output.width == 2);

    assert(tensor_get(&output, 0, 0, 0) == 8.0f);
    assert(tensor_get(&output, 0, 0, 1) == 7.0f);
    assert(tensor_get(&output, 0, 1, 0) == 9.0f);
    assert(tensor_get(&output, 0, 1, 1) == 8.0f);

    tensor_free(&output);
    tensor_free(&input);
}

int main(void) {
    test_tensor_set_get();
    test_relu();
    test_argmax();
    test_linear();
    test_maxpool();

    puts("All C unit tests passed.");
    return 0;
}