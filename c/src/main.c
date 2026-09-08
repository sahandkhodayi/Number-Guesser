#include "../include/raylib.h"
#include "../include/ui.h"
#include "../include/nn.h"
#include <stdio.h>
#include <math.h>

#define WINDOW_W 320
#define WINDOW_H 420
#define CANVAS_X 20
#define CANVAS_Y 20
#define BUTTON_H 50

typedef struct { Rectangle rect; const char *label; } Button;

static void softmax(const float *logits, float *probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

static void run_prediction(AppState *app, const CnnModel *model) {
    float mnist_input[MNIST_SIZE * MNIST_SIZE];
    canvas_to_mnist_input(app, mnist_input);

    Tensor input = tensor_alloc(1, MNIST_SIZE, MNIST_SIZE);
    for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) input.data[i] = mnist_input[i];

    float logits[10];
    model_forward(model, &input, logits);
    tensor_free(&input);

    float probs[10];
    softmax(logits, probs, 10);
    int pred = argmax(logits, 10);
    app->predicted_digit = pred;
    app->confidence = probs[pred];
}

int main(void) {
    CnnModel model;
    int model_ok = (model_load(&model, "models/weights.bin") == 0);
    if (!model_ok) {
        fprintf(stderr, "Warning: could not load models/weights.bin — "
                        "Predict button will be disabled. Run python/export.py first.\n");
    }

    InitWindow(WINDOW_W, WINDOW_H, "Number Guesser");
    SetTargetFPS(60);

    AppState app;
    canvas_clear(&app);

    Rectangle canvas_rect = { CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE };
    Button clear_btn   = { { CANVAS_X, CANVAS_Y + CANVAS_SIZE + 10, CANVAS_SIZE / 2.0f - 5, BUTTON_H }, "CLEAR" };
    Button predict_btn = { { CANVAS_X + CANVAS_SIZE / 2.0f + 5, CANVAS_Y + CANVAS_SIZE + 10, CANVAS_SIZE / 2.0f - 5, BUTTON_H }, "PREDICT" };

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (CheckCollisionPointRec(mouse, canvas_rect)) {
                int px = (int)(mouse.x - CANVAS_X);
                int py = (int)(mouse.y - CANVAS_Y);
                canvas_draw_at(&app, px, py);
            } else if (CheckCollisionPointRec(mouse, clear_btn.rect)) {
                canvas_clear(&app);
            } else if (model_ok && CheckCollisionPointRec(mouse, predict_btn.rect)) {
                run_prediction(&app, &model);
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawRectangleRec(canvas_rect, BLACK);
        for (int y = 0; y < CANVAS_SIZE; y++) {
            for (int x = 0; x < CANVAS_SIZE; x++) {
                float v = app.pixels[y * CANVAS_SIZE + x];
                if (v > 0.0f) {
                    unsigned char g = (unsigned char)(v * 255.0f);
                    DrawPixel(CANVAS_X + x, CANVAS_Y + y, (Color){ g, g, g, 255 });
                }
            }
        }
        DrawRectangleLinesEx(canvas_rect, 2, DARKGRAY);

        DrawRectangleRec(clear_btn.rect, LIGHTGRAY);
        DrawText(clear_btn.label, (int)(clear_btn.rect.x + 30), (int)(clear_btn.rect.y + 15), 18, BLACK);
        DrawRectangleRec(predict_btn.rect, model_ok ? SKYBLUE : GRAY);
        DrawText(predict_btn.label, (int)(predict_btn.rect.x + 15), (int)(predict_btn.rect.y + 15), 18, BLACK);

        if (app.predicted_digit >= 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Prediction: %d", app.predicted_digit);
            DrawText(buf, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 20, 22, BLACK);
            snprintf(buf, sizeof(buf), "Confidence: %.0f%%", app.confidence * 100.0f);
            DrawText(buf, CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 50, 18, DARKGRAY);
        } else if (!model_ok) {
            DrawText("No model loaded", CANVAS_X, CANVAS_Y + CANVAS_SIZE + BUTTON_H + 20, 18, MAROON);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}