# PuppyCam. Default: NVIDIA GPU image (CUDA 12.8 supports GTX 16xx through RTX 50xx).
# CPU-only build: see docker-compose.cpu.yml.
ARG BASE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/data/.ultralytics \
    DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only used by the CPU build (the GPU base image already has PyTorch).
ARG TORCH_INDEX=""
RUN if [ -n "$TORCH_INDEX" ]; then pip install torch torchvision --index-url "$TORCH_INDEX"; fi

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY puppycam ./puppycam

EXPOSE 8080
VOLUME /data
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8080/api/health || exit 1
CMD ["python", "-m", "puppycam"]
