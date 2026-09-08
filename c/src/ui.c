#include "ui.h"

void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
}

void canvas_draw_at(AppState *app, int px, int py) {
    int r = (int)BRUSH_RADIUS;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = px + dx;
            int y = py + dy;
            if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;

            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > BRUSH_RADIUS) continue;

            float strength = 1.0f - (dist / BRUSH_RADIUS) * 0.3f;
            float *pixel = &app->pixels[y * CANVAS_SIZE + x];
            *pixel = fmaxf(*pixel, strength);
        }
    }
}

void canvas_to_mnist_input(const AppState *app, float *out28x28) {
    int block = CANVAS_SIZE / MNIST_SIZE; // 10
    for (int oy = 0; oy < MNIST_SIZE; oy++) {
        for (int ox = 0; ox < MNIST_SIZE; ox++) {
            float sum = 0.0f;
            for (int by = 0; by < block; by++) {
                for (int bx = 0; bx < block; bx++) {
                    int iy = oy * block + by;
                    int ix = ox * block + bx;
                    sum += app->pixels[iy * CANVAS_SIZE + ix];
                }
            }
            out28x28[oy * MNIST_SIZE + ox] = sum / (float)(block * block);
        }
    }
}