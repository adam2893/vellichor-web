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
# PyTorch native XPU support via the official XPU wheel index.
#
# Two apt repos are needed:
#   1. oneAPI repo  — SYCL runtime (libsycl.so) for PyTorch's SYCL backend
#   2. Intel GPU repo — Level Zero loader + Intel GPU adapter + OpenCL ICD
#
# The Intel GPU adapter (libze_intel_gpu.so.1, from intel-level-zero-gpu) is
# REQUIRED for PyTorch XPU to enumerate devices. libze1 / level-zero is just
# the loader — without the adapter, torch.xpu.device_count() returns 0 and
# everything silently runs on CPU. /dev/dri passthrough only shares the kernel
# render node; the userspace driver stack must be installed inside the container.
#
# We use intel-opencl-icd (GPU repo) instead of intel-oneapi-runtime-opencl
# (oneAPI repo) to avoid a dpkg file-overwrite conflict over libze_intel_gpu.so.1.
#
# Battlemage (Arc B580, Xe2) requires PyTorch >= 2.7 for precompiled kernels.
# 2.6.0+xpu has no xe2 AOT kernels and will fail or fall back to slow JIT.

# oneAPI repo: SYCL runtime only
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor > /usr/share/keyrings/intel-oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] \
    https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list

# Intel GPU repo: Level Zero loader + GPU adapter + OpenCL ICD
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor > /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/gpu/ubuntu jammy client" \
    > /etc/apt/sources.list.d/intel.gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        level-zero \
        intel-level-zero-gpu \
        intel-opencl-icd \
        intel-oneapi-runtime-compilers \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --retries 10 --timeout 300 \
    torch==2.7.1+xpu \
    torchaudio==2.7.1+xpu \
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

ARG TORCH_BACKEND
COPY requirements.txt .
RUN pip install -r requirements.txt

# chatterbox-tts pins torch==2.6.0, which downgrades the 2.7.1+xpu wheel
# installed in the openvino stage. Reinstall the XPU torch for Battlemage
# support. Other backends are unaffected (their torch already satisfies the pin).
RUN if [ "$TORCH_BACKEND" = "openvino" ]; then \
        pip install --retries 10 --timeout 300 \
            torch==2.7.1+xpu torchaudio==2.7.1+xpu \
            --index-url https://download.pytorch.org/whl/xpu; \
    fi

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
