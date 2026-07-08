"""Serialize the two GPU consumers — Kokoro TTS and the Ollama Smart-cast model
— so they never fight over limited VRAM at the same time, and so the VRAM is
handed cleanly from one to the other.

Supports CUDA (NVIDIA), OpenVINO/XPU (Intel Arc), and Vulkan backends.  When
both models can't coexist in VRAM, whichever consumer is about to run takes
LOCK and evicts the other from VRAM first, so it gets a full GPU offload
instead of CPU fallback.
"""
import os
import threading

import requests

# Reentrant so a single thread can nest acquisitions safely; different threads
# (the conversion worker vs. a Smart-cast request) still serialize.
LOCK = threading.RLock()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL = os.environ.get("SMARTCAST_MODEL", "llama3.2:3b")


def release_kokoro():
    """Drop Kokoro's cached pipelines and free its VRAM (so Ollama can offload
    to the GPU instead of falling back to CPU)."""
    from tts import ENGINE
    ENGINE.unload()


def release_ollama():
    """Ask Ollama to evict the model from VRAM immediately (keep_alive=0) so
    Kokoro can reclaim the GPU. No-op/quick if the model isn't loaded."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": MODEL, "keep_alive": 0}, timeout=10)
    except requests.RequestException:
        pass


def empty_gpu_cache():
    """Try to free GPU memory across all supported backends."""
    try:
        import gc
        import torch
        import backends
        gc.collect()
        be = backends.current()
        if be.id == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
    except Exception:
        pass
