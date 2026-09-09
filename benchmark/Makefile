CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic

TARGET = benchmark_c

all: $(TARGET)

$(TARGET): c_benchmark.c ../c/src/nn.c
	$(CC) $(CFLAGS) -I../c/include $^ -o $@ -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) c_*.bin pytorch_*.bin input.bin label.txt

.PHONY: all run clean
