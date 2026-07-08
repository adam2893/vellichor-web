"""GPU backend detection and device-compatibility layer.

Vellichor supports four backends, detected in priority order:
  cuda    – NVIDIA GPUs via CUDA (existing, mature)
  openvino – Intel Arc / iGPU via Intel Extension for PyTorch (IPEX) + OpenVINO
  vulkan  – AMD / cross-vendor GPUs via PyTorch's Vulkan backend (experimental)
  cpu     – fallback

Because the Kokoro KPipeline and Chatterbox internals hard-code a CUDA check
(`torch.cuda.is_available()`), we provide a compatibility shim that redirects
`'cuda'` device ops to the actual backend (e.g. `'xpu'` for Intel).  This
avoids forking the model packages while letting inference run on any GPU.

Activation is automatic at import time if ``VELLICHOR_GPU_BACKEND`` is set, or
by calling ``activate(backend_id)`` explicitly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Backend descriptors
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    id: str            # "cuda" | "openvino" | "vulkan" | "cpu"
    label: str         # human-readable (shown in UI)
    torch_device: str  # PyTorch device string ("cuda", "xpu", "vulkan", "cpu")
    icon: str = ""     # optional emoji / short label

    def __hash__(self):
        return hash(self.id)


CUDA = Backend("cuda",     "NVIDIA CUDA",      "cuda",    "⚡")
OPENVINO = Backend("openvino", "Intel OpenVINO / XPU", "xpu", "🔷")
VULKAN  = Backend("vulkan",  "Vulkan (experimental)",  "vulkan", "🔶")
CPU     = Backend("cpu",     "CPU (no GPU)",    "cpu",     "💻")

DETECT_ORDER = [CUDA, OPENVINO, VULKAN]

_ACTIVE: Optional[Backend] = None
_SHIM_ACTIVE = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _detect_openvino() -> bool:
    """Intel Arc / iGPU available via IPEX (Intel Extension for PyTorch).

    IPEX registers ``torch.xpu`` and provides ``torch.xpu.is_available()``.
    Also works when the host has the Intel GPU compute-runtime installed and
    the Docker container has /dev/dri passed through.
    """
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    # Fallback: check the OpenVINO runtime directly (device may be visible even
    # without PyTorch XPU integration).
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
    """PyTorch Vulkan backend available.

    Requires a PyTorch build with ``USE_VULKAN=1`` (not default in pip wheels).
    On Linux the host must have Vulkan drivers + ICD loader; pass /dev/dri into
    the container.
    """
    try:
        import torch
        if hasattr(torch.backends, "vulkan") and torch.backends.vulkan.is_available():
            return True
    except Exception:
        pass
    return False


_DETECTORS = {
    CUDA:     _detect_cuda,
    OPENVINO: _detect_openvino,
    VULKAN:   _detect_vulkan,
}


def detect() -> Backend:
    """Run the detection chain and return the best available backend."""
    for be in DETECT_ORDER:
        try:
            if _DETECTORS[be]():
                return be
        except Exception:
            continue
    return CPU


# ---------------------------------------------------------------------------
# Activation / device shim
# ---------------------------------------------------------------------------

def _install_shim(backend: Backend):
    """Monkey-patch PyTorch so that ``.to('cuda')`` calls are redirected to the
    actual backend device (e.g. ``'xpu'`` for OpenVINO, ``'vulkan'`` for Vulkan).

    This is the key trick: Kokoro's ``KPipeline`` and Chatterbox's
    ``ChatterboxTTS`` both check ``torch.cuda.is_available()`` and call
    ``model.to('cuda')``.  We make that check pass and redirect the device
    move to the real backend, so both engines work on any GPU vendor without
    touching their source code.
    """
    global _SHIM_ACTIVE
    if _SHIM_ACTIVE:
        return
    if backend is CUDA or backend is CPU:
        _SHIM_ACTIVE = True
        return   # no shim needed — works natively

    import torch
    target_device = backend.torch_device

    # ---- torch.cuda.is_available / device_count ---------------------------
    _orig_cuda_available = torch.cuda.is_available
    _orig_cuda_device_count = torch.cuda.device_count

    def _shim_available():
        return True

    def _shim_device_count():
        return 1

    torch.cuda.is_available = _shim_available
    torch.cuda.device_count = _shim_device_count

    # ---- torch.Tensor.to ------------------------------------------------
    _orig_tensor_to = torch.Tensor.to

    def _shim_tensor_to(self, *args, **kwargs):
        # Redirect 'cuda' (or 'cuda:0') to the actual target device
        new_args = list(args)
        for i, a in enumerate(new_args):
            if isinstance(a, str) and a.startswith("cuda"):
                new_args[i] = a.replace("cuda", target_device, 1)
            elif isinstance(a, torch.device) and a.type == "cuda":
                new_args[i] = torch.device(target_device, a.index if a.index is not None else 0)
        return _orig_tensor_to(self, *new_args, **kwargs)

    torch.Tensor.to = _shim_tensor_to

    # ---- torch.nn.Module.to ---------------------------------------------
    _orig_module_to = torch.nn.Module.to

    def _shim_module_to(self, *args, **kwargs):
        new_args = list(args)
        for i, a in enumerate(new_args):
            if isinstance(a, str) and a.startswith("cuda"):
                new_args[i] = a.replace("cuda", target_device, 1)
            elif isinstance(a, torch.device) and a.type == "cuda":
                new_args[i] = torch.device(target_device, a.index if a.index is not None else 0)
        return _orig_module_to(self, *new_args, **kwargs)

    torch.nn.Module.to = _shim_module_to

    # ---- torch.cuda.empty_cache → no-op (backend handles its own memory) --
    _orig_empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: None

    # Store originals for potential cleanup / unshim
    _SHIM_ORIGINALS["is_available"] = _orig_cuda_available
    _SHIM_ORIGINALS["device_count"] = _orig_cuda_device_count
    _SHIM_ORIGINALS["tensor_to"] = _orig_tensor_to
    _SHIM_ORIGINALS["module_to"] = _orig_module_to
    _SHIM_ORIGINALS["empty_cache"] = _orig_empty_cache

    _SHIM_ACTIVE = True


_SHIM_ORIGINALS: dict = {}


def activate(backend_id: Optional[str] = None):
    """Select and activate a GPU backend.

    Call once at startup, before any models are loaded.  If *backend_id* is
    ``None``, auto-detect.  Set the env var ``VELLICHOR_GPU_BACKEND`` to force
    a specific backend (e.g. ``VELLICHOR_GPU_BACKEND=openvino``).
    """
    global _ACTIVE

    if _ACTIVE is not None:
        return _ACTIVE   # already activated

    forced = backend_id or os.environ.get("VELLICHOR_GPU_BACKEND", "").strip().lower()

    if forced:
        for be in [CUDA, OPENVINO, VULKAN, CPU]:
            if be.id == forced:
                _ACTIVE = be
                break
        if _ACTIVE is None:
            print(f"[backends] Unknown backend '{forced}', falling back to auto-detect",
                  flush=True)
            _ACTIVE = detect()
    else:
        _ACTIVE = detect()

    _install_shim(_ACTIVE)

    print(f"[backends] Activated: {_ACTIVE.label} (device={_ACTIVE.torch_device})",
          flush=True)
    return _ACTIVE


def current() -> Backend:
    """Return the currently active backend (activates on first call if needed)."""
    global _ACTIVE
    if _ACTIVE is None:
        activate()
    return _ACTIVE
