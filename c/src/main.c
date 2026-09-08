#include "raylib.h"
#include "../include/nn.h"
#include "../include/ui.h"
#include <math.h>
#include <stdio.h>

#define WINDOW_W 900
#define WINDOW_H 600
#define CANVAS_X 50
#define CANVAS_Y 80
#define CANVAS_SIZE 280
#define BUTTON_W 100
#define BUTTON_H 50
#define BAR_CHART_X 380
#define BAR_CHART_Y 100
#define BAR_CHART_W 200
#define BAR_CHART_H 300

typedef struct {
    Rectangle rect;
    const char *label;
    Color color;
} Button;

static Color prob_color(float p) {
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    unsigned char r = (unsigned char)(255 * (1.0f - p));
    unsigned char g = (unsigned char)(255 * p);
    return (Color){r, g, 30, 255};
}

static void softmax(const float *logits, float *probs, int n) {
    /* Subtracting max_val prevents expf() from overflowing for large logits. */
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

static void draw_probability_bars(const float *probs, int x, int y, int w, int h) {
    int bar_w = w / 10;
    int max_h = h - 30;

    DrawLine(x, y + max_h, x + w, y + max_h, LIGHTGRAY);
    DrawLine(x, y, x, y + max_h, LIGHTGRAY);

    for (int i = 0; i < 10; i++) {
        int bar_x = x + i * bar_w + 2;
        int bar_h = (int)(probs[i] * max_h);
        int bar_y = y + max_h - bar_h;
        Color c = prob_color(probs[i]);

        DrawRectangle(bar_x, bar_y, bar_w - 4, bar_h, c);

        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        DrawText(label, bar_x + 2, y + max_h + 5, 12, DARKGRAY);
    }
}

static void run_prediction(AppState *app, const CnnModel *model) {
    float mnist_input[MNIST_SIZE * MNIST_SIZE];
    canvas_to_mnist_input(app, mnist_input);

    Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);
    for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
        input.data[i] = mnist_input[i];
    }

    float logits[10];
    model_forward(model, &input, logits);
    tensor_free(&input);

    softmax(logits, app->probs, 10);
    app->predicted_digit = argmax(logits, 10);
    app->confidence = app->probs[app->predicted_digit];
    app->has_prediction = 1;
}

int main(void) {
    CnnModel model;
    int model_ok = (model_load(&model, "models/weights.bin") == 0);
    if (!model_ok) {
        fprintf(stderr, "Warning: could not load models/weights.bin\n");
    }

    InitWindow(WINDOW_W, WINDOW_H, "Number Guesser Pro");
    SetTargetFPS(60);

    AppState app;
    canvas_clear(&app);

    Rectangle canvas_rect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};
    Button clear_btn = {
        {CANVAS_X, CANVAS_Y + CANVAS_SIZE + 20, BUTTON_W, BUTTON_H},
        "CLEAR", LIGHTGRAY
    };
    Button predict_btn = {
        {CANVAS_X + BUTTON_W + 20, CANVAS_Y + CANVAS_SIZE + 20, BUTTON_W, BUTTON_H},
        "PREDICT", model_ok ? SKYBLUE : GRAY
    };

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        /* CHANGE: button actions use Pressed instead of Down.
           Holding the mouse button no longer triggers the action every frame. */
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            if (CheckCollisionPointRec(mouse, clear_btn.rect)) {
                canvas_clear(&app);
            } else if (model_ok && CheckCollisionPointRec(mouse, predict_btn.rect)) {
                run_prediction(&app, &model);
            }
        }

        /* Drawing uses Down because a brush should continue while dragging. */
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) &&
            CheckCollisionPointRec(mouse, canvas_rect)) {
            float px = mouse.x - CANVAS_X;
            float py = mouse.y - CANVAS_Y;

            if (!app.is_drawing) {
                app.is_drawing = 1;
                app.last_mouse_x = px;
                app.last_mouse_y = py;
                canvas_draw_point(&app, px, py);
            } else {
                canvas_draw_line(&app, app.last_mouse_x, app.last_mouse_y, px, py);
                app.last_mouse_x = px;
                app.last_mouse_y = py;
            }
        } else {
            app.is_drawing = 0;
        }

        if (IsKeyPressed(KEY_C)) canvas_clear(&app);
        if (IsKeyPressed(KEY_ENTER) && model_ok) run_prediction(&app, &model);

        BeginDrawing();
        ClearBackground(GetColor(0x1a1a2eFF));

        DrawText("Draw a Digit", 20, 10, 28, RAYWHITE);
        DrawText("Press 'C' to clear | Enter to predict", 20, 45, 16, LIGHTGRAY);

        DrawRectangleRec(canvas_rect, BLACK);
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                float v = app.pixels[y * CANVAS_SIZE + x];
                if (v > 0.01f) {
                    unsigned char g = (unsigned char)(v * 255.0f);
                    DrawPixel(CANVAS_X + x, CANVAS_Y + y, (Color){g, g, g, 255});
                }
            }
        }
        DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);

        DrawRectangleRec(clear_btn.rect, clear_btn.color);
        DrawText(clear_btn.label, (int)(clear_btn.rect.x + 25),
                 (int)(clear_btn.rect.y + 15), 18, BLACK);

        DrawRectangleRec(predict_btn.rect, predict_btn.color);
        DrawText(predict_btn.label, (int)(predict_btn.rect.x + 15),
                 (int)(predict_btn.rect.y + 15), 18, BLACK);

        if (app.has_prediction) {
            draw_probability_bars(app.probs, BAR_CHART_X, BAR_CHART_Y,
                                  BAR_CHART_W, BAR_CHART_H);

            char buf[128];
            snprintf(buf, sizeof(buf), "Prediction: %d", app.predicted_digit);
            DrawText(buf, BAR_CHART_X, BAR_CHART_Y + BAR_CHART_H + 30, 28, RAYWHITE);

            snprintf(buf, sizeof(buf), "Confidence: %.1f%%", app.confidence * 100.0f);
            Color conf_color = (app.confidence > 0.8f) ? GREEN :
                               (app.confidence > 0.5f) ? YELLOW : RED;
            DrawText(buf, BAR_CHART_X, BAR_CHART_Y + BAR_CHART_H + 60, 18, conf_color);
        } else if (!model_ok) {
            DrawText("No model loaded", BAR_CHART_X, 200, 18, MAROON);
        } else {
            DrawText("Draw a digit and press PREDICT", BAR_CHART_X, 200, 18, GRAY);
        }

        DrawText("Canvas: 280x280 -> MNIST-style 28x28", CANVAS_X,
                 CANVAS_Y + CANVAS_SIZE + BUTTON_H + 60, 12, GRAY);
        DrawFPS(WINDOW_W - 80, 10);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
