#ifndef UI_H
#define UI_H

#include <stddef.h>
#include <string.h>
#include <math.h>

#define CANVAS_SIZE 280
#define MNIST_SIZE 28
#define BRUSH_RADIUS 12.0f
#define BRUSH_STRENGTH 0.85f

typedef struct {
    float pixels[CANVAS_SIZE * CANVAS_SIZE];
    int predicted_digit;
    float confidence;
    float probs[10];
    int has_prediction;
    float last_mouse_x;
    float last_mouse_y;
    int is_drawing;
} AppState;

void canvas_clear(AppState *app);
void canvas_draw_line(AppState *app, float x1, float y1, float x2, float y2);
void canvas_draw_point(AppState *app, float px, float py);
void canvas_to_mnist_input(const AppState *app, float *out28x28);

#endif