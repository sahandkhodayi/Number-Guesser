
# 🧠 AI Number Guesser

> A polyglot desktop application that recognizes hand-drawn digits (0-9) in real-time, powered by a custom-trained neural network.

[![Language: C](https://img.shields.io/badge/Language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Language: Python](https://img.shields.io/badge/Language-Python-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

The **AI Number Guesser** is an interactive desktop application that demonstrates a complete machine learning pipeline, from model training to a native desktop integration. Users can draw a digit on a canvas, and the application will accurately predict the number in real-time.

This project showcases a **polyglot architecture**, combining the high-performance, low-level capabilities of **C** for the user interface and system interactions with the powerful, flexible machine learning ecosystem of **Python** and **PyTorch** for the inference engine.

## ✨ Key Features

*   **Interactive Drawing Canvas:** A lightweight, custom-built UI in C allows users to draw digits naturally with their mouse.
*   **Deep Learning Core:** Utilizes a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify hand-drawn images with high accuracy.
*   **Real-time Inference:** The C backend seamlessly communicates with the Python model for instant predictions.
*   **Modular & Extensible:** The project is cleanly separated into a C frontend, a Python ML pipeline, and a shared data layer, making it easy to modify or extend.

## 🏗️ Architecture

The system is designed with a clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      C Application (UI)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Captures user input from a drawing canvas.          │  │
│  │  • Processes and normalizes the image.                 │  │
│  │  • Displays the predicted digit.                       │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │ (Image Data)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python Inference Engine                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Loads the trained PyTorch model.                    │  │
│  │  • Performs a forward pass for classification.         │  │
│  │  • Returns the predicted class label.                  │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │ (Prediction)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • `models/` : Stores trained model weights (.pt).     │  │
│  │  • `data/`   : Handles dataset loading and caching.   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | C | High-performance, native desktop UI and event handling. |
| **ML Framework** | PyTorch | Building, training, and deploying the neural network. |
| **Data Handling** | NumPy, Pandas | Dataset manipulation and preprocessing. |
| **Model** | CNN | Convolutional Neural Network for image classification. |
| **Environment** | Python 3.10+ | Managing the machine learning pipeline. |

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

*   **C Compiler:** GCC or Clang.
*   **Python:** Version 3.8 or higher.
*   **Build Tools:** Make or CMake.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sahandkhodayi/Number-Guesser.git
    cd Number-Guesser
    ```

2.  **Set up the Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Compile the C application:**
    ```bash
    cd c
    make
    ```

### Usage

1.  **Ensure the trained model is present:**
    The `models/` directory should contain the exported model file (e.g., `mnist_model.pt`). If not, you can train it by running:
    ```bash
    python python/train.py
    ```

2.  **Run the application:**
    ```bash
    ./c/number_guesser
    ```

3.  **Interact with the UI:**
    *   A window will appear with a drawing canvas.
    *   Use your mouse to draw a single digit (0-9).
    *   The application will display its prediction instantly.

## 📂 Project Structure

```
Number-Guesser/
├── c/                     # C source code for the UI
│   ├── include/           # Header files
│   ├── src/               # Source files
│   ├── tools/             # Utility scripts
│   └── number_guesser     # Compiled executable
├── data/                  # Dataset files (e.g., MNIST)
├── models/                # Trained model weights
├── python/                # Python ML pipeline
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # CNN architecture definition
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation script
│   └── export.py          # Script to export model for C inference
├── tests/                 # Unit and integration tests
├── main.py                # Main Python entry point
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🧪 Testing

The project includes a `tests/` directory for ensuring code quality. To run the tests:

```bash
# Run Python tests
pytest

# Run C tests (if applicable)
cd c && make test
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 📧 Contact

**Sahand Khodayi** - [GitHub Profile](https://github.com/sahandkhodayi)

Project Link: [https://github.com/sahandkhodayi/Number-Guesser](https://github.com/sahandkhodayi/Number-Guesser)

