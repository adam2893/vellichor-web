"""GPU backend detection and device-compatibility layer.

Vellichor supports four backends, detected in priority order:
  cuda    – NVIDIA GPUs via CUDA (existing, mature)
  openvino – Intel Arc / iGPU via native PyTorch XPU + OpenVINO
  vulkan  – AMD / cross-vendor GPUs via PyTorch's Vulkan backend (experimental)
  cpu     – fallback

For non-CUDA backends, we provide a ``move_to_device`` helper that moves PyTorch
models to the target device after loading.  This is safer than a CUDA shim:
Kokoro and Chatterbox check torch.cuda.is_available() during load and choose
CPU, then we move everything to the real device — no lies, no crashes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Backend descriptors
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    id: str
    label: str
    torch_device: str
    icon: str = ""

    def __hash__(self):
        return hash(self.id)


CUDA     = Backend("cuda",     "NVIDIA CUDA",          "cuda",   "⚡")
OPENVINO = Backend("openvino", "Intel OpenVINO / XPU", "xpu",    "🔷")
VULKAN   = Backend("vulkan",  "Vulkan (experimental)", "vulkan", "🔶")
CPU      = Backend("cpu",     "CPU (no GPU)",          "cpu",    "💻")

DETECT_ORDER = [CUDA, OPENVINO, VULKAN]
_ACTIVE: Optional[Backend] = None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> bool:
    try:
        import torch
        if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def _detect_openvino() -> bool:
    """Intel Arc / iGPU via OpenVINO runtime.  Detects through /dev/dri."""
    try:
        import openvino as ov
        core = ov.Core()
        for d in core.available_devices:
            if "GPU" in d.upper():
                return True
    except Exception:
        pass
    return False


def _detect_vulkan() -> bool:
    try:
        import torch
        if hasattr(torch.backends, "vulkan") and torch.backends.vulkan.is_available():
            return True
    except Exception:
        pass
    return False


_DETECTORS = {CUDA: _detect_cuda, OPENVINO: _detect_openvino, VULKAN: _detect_vulkan}


def detect() -> Backend:
    for be in DETECT_ORDER:
        try:
            if _DETECTORS[be]():
                return be
        except Exception:
            continue
    return CPU


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def activate(backend_id: Optional[str] = None) -> Backend:
    global _ACTIVE
    if _ACTIVE is not None:
        return _ACTIVE

    forced = backend_id or os.environ.get("VELLICHOR_GPU_BACKEND", "").strip().lower()
    if forced:
        for be in [CUDA, OPENVINO, VULKAN, CPU]:
            if be.id == forced:
                _ACTIVE = be
                break
        if _ACTIVE is None:
            print(f"[backends] Unknown backend '{forced}', falling back to auto-detect", flush=True)
            _ACTIVE = detect()
    else:
        _ACTIVE = detect()

    print(f"[backends] Activated: {_ACTIVE.label} (device={_ACTIVE.torch_device})", flush=True)
    return _ACTIVE


def current() -> Backend:
    global _ACTIVE
    if _ACTIVE is None:
        activate()
    return _ACTIVE


# ---------------------------------------------------------------------------
# Model movement helper — move loaded models to the real GPU
# ---------------------------------------------------------------------------

def move_to_device(obj: object, device: str) -> None:
    """Recursively move every torch.nn.Module in *obj* to *device*.

    After a library like Kokoro loads a model on CPU (because it can't
    detect XPU/Vulkan), call this once to shift everything to the GPU.
    Only handles attributes of *obj* — does NOT walk global state.
    """
    import torch

    def _walk(o):
        if isinstance(o, torch.nn.Module):
            try:
                o.to(device)
            except Exception:
                pass
            for child in o.children():
                _walk(child)
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple, set)):
            for item in o:
                _walk(item)

    _walk(obj.__dict__ if hasattr(obj, "__dict__") else obj)
