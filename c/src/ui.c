#include "../include/ui.h"
#include <stdio.h>

void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
    app->has_prediction = 0;
    memset(app->probs, 0, sizeof(app->probs));
    app->is_drawing = 0;
}

static void draw_circle_brush(AppState *app, float cx, float cy, float radius, float strength) {
    int r = (int)(radius + 1);
    int center_x = (int)cx;
    int center_y = (int)cy;
    
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = center_x + dx;
            int y = center_y + dy;
            if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) continue;
            
            float dist = sqrtf((float)(dx * dx + dy * dy));
            if (dist > radius) continue;
            
            // Gaussian-like falloff
            float falloff = 1.0f - (dist * dist) / (radius * radius);
            float final_strength = strength * falloff;
            
            float *pixel = &app->pixels[y * CANVAS_SIZE + x];
            *pixel = fmaxf(*pixel, final_strength);
        }
    }
}

void canvas_draw_point(AppState *app, float px, float py) {
    draw_circle_brush(app, px, py, BRUSH_RADIUS, BRUSH_STRENGTH);
}

void canvas_draw_line(AppState *app, float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dist = sqrtf(dx * dx + dy * dy);
    
    if (dist < 0.1f) {
        canvas_draw_point(app, x1, y1);
        return;
    }
    
    int steps = (int)(dist * 1.5f) + 1;
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / steps;
        float px = x1 + dx * t;
        float py = y1 + dy * t;
        draw_circle_brush(app, px, py, BRUSH_RADIUS * 0.8f, BRUSH_STRENGTH);
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