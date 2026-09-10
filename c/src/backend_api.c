#include "../include/nn.h"

#include <math.h>
#include <string.h>

/*
 * Small FFI boundary for the Python GUI.
 *
 * Python provides exactly 784 normalized float32 pixels. C runs the existing
 * CNN inference code and returns ten logits plus the predicted class.
 *
 * The model is loaded once and reused, so every mouse prediction does not
 * reopen and reread the weights file.
 */
int number_guesser_predict(const float *input_784,
                           float *logits_out,
                           int *prediction_out) {
    static CnnModel model;
    static int loaded = 0;

    if (input_784 == NULL || logits_out == NULL || prediction_out == NULL) {
        return -1;
    }

    if (!loaded) {
        if (model_load(&model, "models/weights.bin") != 0) {
            return -2;
        }
        loaded = 1;
    }

    Tensor input = tensor_alloc(1, 28, 28);
    memcpy(input.data, input_784, 784 * sizeof(float));

    model_forward(&model, &input, logits_out);
    *prediction_out = argmax(logits_out, 10);

    tensor_free(&input);
    return 0;
}
