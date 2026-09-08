#include "../include/ui.h"
#include <math.h>
#include <string.h>

void canvas_clear(AppState *app) {
    memset(app->pixels, 0, sizeof(app->pixels));
    app->predicted_digit = -1;
    app->confidence = 0.0f;
    app->has_prediction = 0;
    memset(app->probs, 0, sizeof(app->probs));
    app->is_drawing = 0;
}

static void draw_circle_brush(AppState *app, float cx, float cy,
                              float radius, float strength) {
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
        float t = (float)i / (float)steps;
        draw_circle_brush(app, x1 + dx * t, y1 + dy * t,
                          BRUSH_RADIUS * 0.8f, BRUSH_STRENGTH);
    }
}

/* Bilinear resize keeps the preprocessing closer to a real image resize than
   simply averaging fixed 10x10 blocks. */
static float sample_bilinear(const float *src, int width, int height,
                             float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > width - 1) x = (float)(width - 1);
    if (y > height - 1) y = (float)(height - 1);

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1 < width ? x0 + 1 : x0;
    int y1 = y0 + 1 < height ? y0 + 1 : y0;
    float fx = x - x0;
    float fy = y - y0;

    float a = src[y0 * width + x0];
    float b = src[y0 * width + x1];
    float c = src[y1 * width + x0];
    float d = src[y1 * width + x1];
    float top = a + (b - a) * fx;
    float bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

void canvas_to_mnist_input(const AppState *app, float *out28x28) {
    memset(out28x28, 0, sizeof(float) * MNIST_SIZE * MNIST_SIZE);

    /*
     * CHANGE: normalize the user's drawing before inference.
     *
     * The old implementation always mapped the full 280x280 canvas directly
     * to 28x28. A user can draw a tiny digit in one corner, while MNIST digits
     * are roughly centered. This creates a large train/inference distribution
     * mismatch and can make an otherwise good model look inaccurate.
     *
     * We therefore:
     *   1. find the non-empty bounding box,
     *   2. crop it while preserving aspect ratio,
     *   3. resize it to a 20x20 region using bilinear sampling,
     *   4. center that region inside the 28x28 input.
     *
     * This is an inference-side preprocessing improvement; it does not change
     * the trained CNN or its weights.
     */
    const float threshold = 0.02f;
    int min_x = CANVAS_SIZE, min_y = CANVAS_SIZE;
    int max_x = -1, max_y = -1;

    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            float v = app->pixels[y * CANVAS_SIZE + x];
            if (v > threshold) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    /* Empty canvas: keep the all-zero MNIST input. */
    if (max_x < 0 || max_y < 0) return;

    int box_w = max_x - min_x + 1;
    int box_h = max_y - min_y + 1;
    int side = box_w > box_h ? box_w : box_h;

    /* Add a small margin around the drawn digit before resizing. */
    int margin = side / 10;
    side += 2 * margin;
    int center_x = (min_x + max_x) / 2;
    int center_y = (min_y + max_y) / 2;
    int crop_x = center_x - side / 2;
    int crop_y = center_y - side / 2;

    const int target = 20;
    const int offset = (MNIST_SIZE - target) / 2; /* 4 pixels on each side */

    for (int oy = 0; oy < target; oy++) {
        for (int ox = 0; ox < target; ox++) {
            float src_x = crop_x + ((ox + 0.5f) * side / target) - 0.5f;
            float src_y = crop_y + ((oy + 0.5f) * side / target) - 0.5f;
            out28x28[(oy + offset) * MNIST_SIZE + (ox + offset)] =
                sample_bilinear(app->pixels, CANVAS_SIZE, CANVAS_SIZE,
                                src_x, src_y);
        }
    }
}
