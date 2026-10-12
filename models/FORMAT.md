# Number-Guesser model format {#number-guesser-model-format  data-source-line="450"}

## Version 1 {#version-1  data-source-line="452"}

- Magic: four ASCII bytes NGNN
- Version: unsigned 32-bit integer, little-endian
- Architecture ID: unsigned 32-bit integer
- Payload length: unsigned 64-bit integer, little-endian
- Payload: IEEE-754 float32 values, little-endian

Payload order:
1. conv1 weights
2. conv1 bias
3. conv2 weights
4. conv2 bias
5. conv3 weights
6. conv3 bias
7. conv4 weights
8. conv4 bias
9. fully connected weights
10. fully connected bias

The loader rejects an unknown magic, unsupported version,
unknown architecture ID, wrong payload length, or truncated file.