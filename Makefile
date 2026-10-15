.DEFAULT_GOAL := help

PYTHON ?= python
CMAKE ?= cmake
BUILD_DIR := build
APP := $(BUILD_DIR)/number_guesser

.PHONY: help menu train evaluate export collect build run test parity c-test clean

help:
	@echo "Number Guesser commands:"
	@echo "  make menu      - open the interactive menu"
	@echo "  make train     - train and save the PyTorch model"
	@echo "  make evaluate  - evaluate the saved checkpoint (no training)"
	@echo "  make export    - export checkpoint weights for C inference"
	@echo "  make collect   - collect handwritten digit images"
	@echo "  make build     - configure and build the C/Raylib app"
	@echo "  make run       - build if needed and launch the drawing app"
	@echo "  make test      - run parity and C unit tests"
	@echo "  make parity    - run Python/C numerical parity test"
	@echo "  make c-test    - compile and run C unit tests"
	@echo "  make clean     - remove generated build/test artifacts"

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

build:
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR)

run: build
	cd $(BUILD_DIR) && ./number_guesser

parity:
	$(PYTHON) tests/test_parity.py

c-test:
	mkdir -p $(BUILD_DIR)
	gcc -std=c11 -Wall -Wextra -Wpedantic tests/test_nn.c c/src/nn.c -o $(BUILD_DIR)/test_nn -lm
	./$(BUILD_DIR)/test_nn

test: parity c-test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)