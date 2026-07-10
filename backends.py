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

def move_to_device(obj: object, device: str) -> int:
    """Recursively move every torch.nn.Module in *obj* to *device*.

    After a library like Kokoro loads a model on CPU (because it can't
    detect XPU/Vulkan), call this once to shift everything to the GPU.
    Only handles attributes of *obj* — does NOT walk global state.

    Returns the number of modules successfully moved.
    """
    import torch

    moved = 0
    failed = 0
    visited = set()

    def _walk(o, path="obj"):
        nonlocal moved, failed
        obj_id = id(o)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(o, torch.nn.Module):
            try:
                current_device = None
                try:
                    p = next(o.parameters(), None)
                    if p is not None:
                        current_device = str(p.device)
                except Exception:
                    pass
                o.to(device)
                moved += 1
                # Only log successful moves for top-level or interesting modules
                if "." not in path or path.count(".") <= 2:
                    print(f"[backends] Moved {path} ({type(o).__name__}) to {device} (was {current_device})", flush=True)
            except Exception as e:
                failed += 1
                # SUPPRESS SPAM: only log failures for top-level modules
                if "." not in path or path.count(".") <= 1:
                    print(f"[backends] Failed to move {path} to {device}: {e}", flush=True)
            for name, child in o.named_children():
                _walk(child, f"{path}.{name}")
        elif isinstance(o, dict):
            for k, v in o.items():
                _walk(v, f"{path}[{k!r}]")
        elif isinstance(o, (list, tuple)):
            for i, item in enumerate(o):
                _walk(item, f"{path}[{i}]")
        elif hasattr(o, "__dict__"):
            for k, v in o.__dict__.items():
                _walk(v, f"{path}.{k}")

    _walk(obj)
    print(f"[backends] move_to_device: moved {moved} module(s), {failed} failed, to {device}", flush=True)
    return moved
