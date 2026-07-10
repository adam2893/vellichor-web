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
        ffmpeg espeak-ng libsndfile1 git curl ca-certificates gnupg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------
# CUDA backend (NVIDIA) — default
# -----------------------------------------------------------------
FROM base AS cuda
RUN pip install --retries 10 --timeout 300 torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------------------
# OpenVINO backend (Intel Arc / iGPU)
# -----------------------------------------------------------------
FROM base AS openvino
# PyTorch native XPU support is upstreamed into PyTorch 2.6+.
# We use the official PyTorch XPU wheel index (NOT Intel's old extension index).
# The 2.6.0+xpu wheel satisfies chatterbox-tts's torch==2.6.0 requirement.
#
# Intel oneAPI runtime libraries (libsycl.so.8, Level Zero) are still required
# for the XPU backend to talk to the GPU through /dev/dri.
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor > /usr/share/keyrings/intel-oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] \
    https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libze1 \
        intel-oneapi-runtime-compilers \
        intel-oneapi-runtime-opencl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --retries 10 --timeout 300 \
    torch==2.6.0+xpu \
    torchaudio==2.6.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu

RUN pip install openvino==2025.2.0

# -----------------------------------------------------------------
# Vulkan backend (AMD / cross-vendor) — experimental
# -----------------------------------------------------------------
FROM base AS vulkan
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

# kokoro depends on the standalone triton package, but both CUDA torch and
# PyTorch XPU already include triton internally. Having both causes a double
# TORCH_LIBRARY registration crash ("Only a single TORCH_LIBRARY can be used
# to register the namespace triton"). Remove the standalone package.
RUN pip uninstall -y triton 2>/dev/null || true

COPY app/ /app/

# Hugging Face Hub: disable symlinks (symlinks break on FUSE/shfs — Unraid
# and network mounts). Also enable the faster Rust downloader if available.
ENV HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 7777
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7777/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]
