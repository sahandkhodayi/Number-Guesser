#include "../c/include/nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_input(const char *path, float *input, size_t count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Could not open input: %s\n", path);
        return -1;
    }
    size_t got = fread(input, sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "Expected %zu floats, got %zu\n", count, got);
        return -1;
    }
    return 0;
}

static int write_tensor(const char *name, const Tensor *t) {
    char path[256];
    snprintf(path, sizeof(path), "benchmark/c_%s.bin", name);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Could not write %s\n", path);
        return -1;
    }

    size_t count = (size_t)t->channels * t->height * t->width;
    size_t written = fwrite(t->data, sizeof(float), count, f);
    fclose(f);

    return written == count ? 0 : -1;
}

static int write_logits(const char *name, const float *logits, size_t count) {
    char path[256];
    snprintf(path, sizeof(path), "benchmark/c_%s.bin", name);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(logits, sizeof(float), count, f);
    fclose(f);
    return written == count ? 0 : -1;
}

int main(void) {
    CnnModel model;
    float input_data[28 * 28];

    if (model_load(&model, "models/weights.bin") != 0) {
        return 1;
    }
    if (read_input("benchmark/input.bin", input_data, 28 * 28) != 0) {
        return 1;
    }

    /*
     * Keep these operations explicit instead of calling model_forward().
     * That lets us save every intermediate tensor and find the FIRST layer
     * where C and PyTorch disagree.
     */
    Tensor input = tensor_alloc(1, 28, 28);
    memcpy(input.data, input_data, sizeof(input_data));
    write_tensor("input", &input);

    Tensor conv1 = conv2d(&input, model.conv1_w, model.conv1_b, 32, 3, 1, 1);
    write_tensor("conv1", &conv1);
    relu_tensor(&conv1);
    write_tensor("relu1", &conv1);

    Tensor conv2 = conv2d(&conv1, model.conv2_w, model.conv2_b, 32, 3, 1, 1);
    write_tensor("conv2", &conv2);
    relu_tensor(&conv2);
    write_tensor("relu2", &conv2);
    tensor_free(&conv1);

    Tensor pool1 = maxpool2d(&conv2, 2, 2);
    write_tensor("pool1", &pool1);
    tensor_free(&conv2);

    Tensor conv3 = conv2d(&pool1, model.conv3_w, model.conv3_b, 32, 3, 1, 1);
    write_tensor("conv3", &conv3);
    relu_tensor(&conv3);
    write_tensor("relu3", &conv3);
    tensor_free(&pool1);

    Tensor conv4 = conv2d(&conv3, model.conv4_w, model.conv4_b, 32, 3, 1, 1);
    write_tensor("conv4", &conv4);
    relu_tensor(&conv4);
    write_tensor("relu4", &conv4);
    tensor_free(&conv3);

    Tensor pool2 = maxpool2d(&conv4, 2, 2);
    write_tensor("pool2", &pool2);
    tensor_free(&conv4);

    float logits[10];
    linear(model.fc_w, model.fc_b, pool2.data, logits, 32 * 7 * 7, 10);
    write_logits("logits", logits, 10);

    printf("C prediction: %d\n", argmax(logits, 10));
    tensor_free(&pool2);
    tensor_free(&input);
    return 0;
}
