.DEFAULT_GOAL := help

PYTHON ?= python
CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build

.PHONY: help menu train evaluate export collect configure build run test parity c-test clean

help:
	@echo "Number Guesser commands:"
	@echo "  make menu      - open the interactive menu"
	@echo "  make train     - train and save the PyTorch model"
	@echo "  make evaluate  - evaluate the saved checkpoint (no training)"
	@echo "  make export    - export checkpoint weights for C inference"
	@echo "  make collect   - collect handwritten digit images"
	@echo "  make configure - configure CMake with the desktop GUI enabled"
	@echo "  make build     - build the C/Raylib desktop app"
	@echo "  make run       - build the app and open the interactive menu"
	@echo "  make test      - run headless C tests and Python/C parity checks"
	@echo "  make parity    - run Python/C numerical parity test"
	@echo "  make c-test    - build and run headless C unit tests"
	@echo "  make clean     - remove generated build artifacts"

menu:
	$(PYTHON) main.py

train:
	$(PYTHON) python/train.py

evaluate:
	$(PYTHON) python/evaluate.py

export:
	$(PYTHON) python/export.py

collect:
	$(PYTHON) python/digit_collector.py

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DBUILD_GUI=ON

build: configure
	$(CMAKE) --build $(BUILD_DIR)

run: build
	$(PYTHON) main.py

c-test:
	$(CMAKE) -S . -B $(BUILD_DIR) -DBUILD_GUI=OFF
	$(CMAKE) --build $(BUILD_DIR) --target test_nn
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

parity:
	$(PYTHON) tests/test_parity.py

test: c-test parity

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
