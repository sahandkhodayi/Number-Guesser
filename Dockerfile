FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Headless targets need a C compiler and CMake, but do not need Raylib or a display.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cmake -S . -B build -DBUILD_GUI=OFF \
    && cmake --build build --config Release --target test_nn c_benchmark \
    && mkdir -p models data

# Default behavior is a fast, headless C unit-test run.
CMD ["ctest", "--test-dir", "build", "-C", "Release", "--output-on-failure"]
