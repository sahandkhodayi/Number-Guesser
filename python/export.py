"""Export trained PyTorch weights to a flat binary file for C inference."""

import torch
from pathlib import Path
from model import _MainModel

MODEL_PATH = Path("models/number_guesser_model.pth")
OUTPUT_PATH = Path("models/weights.bin")
EXPECTED_BYTES = 175016

# Order MUST match c/src/nn.c's model_load() exactly!
LAYER_KEYS = [
    "block_1.0.weight", "block_1.0.bias",   # conv1: 1 -> 32
    "block_1.2.weight", "block_1.2.bias",   # conv2: 32 -> 32
    "block_2.0.weight", "block_2.0.bias",   # conv3: 32 -> 32
    "block_2.2.weight", "block_2.2.bias",   # conv4: 32 -> 32
    "classifier.1.weight", "classifier.1.bias",  # linear: 1568 -> 10
]

def main():
    print("[1] Checking for trained model...")
    if not MODEL_PATH.exists():
        raise SystemExit(f"❌ {MODEL_PATH} not found — train and save a model first.")
    
    print("[2] Loading model...")
    model = _MainModel(input_shape=1, hidden_units=32, output_shape=10)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("[3] Checking for missing keys...")
    missing = [k for k in LAYER_KEYS if k not in state_dict]
    if missing:
        raise SystemExit(
            f"❌ state_dict is missing expected keys: {missing}\n"
            f"   Actual keys: {list(state_dict.keys())}\n"
            f"   Update LAYER_KEYS to match your model.py architecture."
        )
    print("   ✅ All keys found!")

    print("[4] Writing weights.bin...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        for key in LAYER_KEYS:
            tensor = state_dict[key]
            # Ensure contiguous (just in case) and write raw bytes
            f.write(tensor.contiguous().numpy().tobytes())

    actual_bytes = OUTPUT_PATH.stat().st_size
    if actual_bytes == EXPECTED_BYTES:
        print(f"✅ Wrote {OUTPUT_PATH} ({actual_bytes} bytes, expected {EXPECTED_BYTES}) [OK]")
    else:
        print(f"⚠️  Wrote {OUTPUT_PATH} ({actual_bytes} bytes, expected {EXPECTED_BYTES}) [MISMATCH]")
        print("   Check that the architecture hasn't changed!")

    print("[5] Done! You can now run C inference with `./verify`.")

if __name__ == "__main__":
    main()