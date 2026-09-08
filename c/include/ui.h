#ifndef UI_H
#define UI_H

#include <stddef.h>
#include <string.h>
#include <math.h>

#define CANVAS_SIZE 280      /* on‑screen drawing area, pixels (= 28*10) */
#define MNIST_SIZE  28
#define BRUSH_RADIUS 8.0f

typedef struct {
    float pixels[CANVAS_SIZE * CANVAS_SIZE]; /* [0,1] grayscale, row‑major */
    int predicted_digit;   /* -1 = no prediction yet */
    float confidence;      /* softmax probability */
} AppState;

void canvas_clear(AppState *app);
void canvas_draw_at(AppState *app, int px, int py);
void canvas_to_mnist_input(const AppState *app, float *out28x28);

#endif