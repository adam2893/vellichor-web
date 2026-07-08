# Vellichor — Multi-backend Docker image
#
# Build for your GPU vendor:
#   docker build -t vellichor-web:latest .                        # NVIDIA CUDA (default)
#   docker build --build-arg TORCH_BACKEND=openvino -t vellichor-web:arc .   # Intel Arc
#   docker build --build-arg TORCH_BACKEND=vulkan  -t vellichor-web:amd .    # AMD / Vulkan
#   docker build --build-arg TORCH_BACKEND=cpu     -t vellichor-web:cpu .    # CPU only

ARG TORCH_BACKEND=cuda

FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    PYTHONPATH=/app

# ffmpeg = audio encoding/m4b assembly; espeak-ng = Kokoro G2P fallback /
# non-English phonemization; libsndfile1 = soundfile backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg espeak-ng libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------
# CUDA backend (NVIDIA) — default
# -----------------------------------------------------------------
FROM base AS cuda
# CUDA 12.4 PyTorch — still supports the GTX 1080 (Pascal, sm_61). Pinned to
# 2.6.0 to satisfy chatterbox-tts's exact torch pin.
RUN pip install --retries 10 --timeout 300 torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------------------
# OpenVINO backend (Intel Arc / iGPU)
# -----------------------------------------------------------------
FROM base AS openvino
# IPEX 2.6.10+xpu pulls the correct torch from its dependency chain and
# registers the 'xpu' device. No separate torch install needed — installing
# torch first from any other index would shadow IPEX's own torch and break XPU.
#
# libze1 provides the Level Zero loader that IPEX uses to talk to the GPU
# through /dev/dri. The actual compute runtime lives on the HOST.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libze1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --retries 10 --timeout 300 \
    intel-extension-for-pytorch==2.6.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# OpenVINO runtime for device detection / optional ONNX acceleration
RUN pip install openvino==2025.2.0

# IPEX bundles its own triton; kokoro's deps pull a standalone triton which
# triggers a duplicate TORCH_LIBRARY namespace registration. This env var
# tells triton not to register its own TORCH_LIBRARY, letting IPEX own it.
ENV TRITON_DISABLE_TORCH_LIBRARY=1

# -----------------------------------------------------------------
# Vulkan backend (AMD / cross-vendor) — experimental
# -----------------------------------------------------------------
FROM base AS vulkan
# Vulkan compute: needs libvulkan1, Mesa Vulkan drivers (radeonsi for AMD,
# intel for Arc), and a PyTorch build with USE_VULKAN.  Stock pip wheels
# don't ship with Vulkan; we install the standard CPU wheel and attempt to
# enable Vulkan at runtime via PyTorch's dynamic backend loading.
# For true Vulkan acceleration, rebuild PyTorch from source with USE_VULKAN=1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --retries 10 --timeout 300 torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------
# CPU-only fallback
# -----------------------------------------------------------------
FROM base AS cpu
RUN pip install --retries 10 --timeout 300 torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------
# Final stage — pick the selected backend
# -----------------------------------------------------------------
FROM ${TORCH_BACKEND} AS final

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ /app/

EXPOSE 7777
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7777/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]
