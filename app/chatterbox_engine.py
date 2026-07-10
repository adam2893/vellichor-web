"""Chatterbox (Resemble AI) TTS backend.

Expressive TTS: one "exaggeration" intensity dial plus zero-shot voice cloning
from a short reference clip. Loaded lazily and unloaded after a job so its VRAM
can go back to Kokoro / the Ollama Smart-cast model (they can't all coexist on
an 8 GB card). Output is resampled to the pipeline's 24 kHz.

NOTE: the exact package API is pinned via research; if the installed
`chatterbox-tts` differs, adjust `_load()` / `synth_chunk()` accordingly.
"""
import threading
import numpy as np

SAMPLE_RATE = 24000          # pipeline rate; Chatterbox output is resampled to this


def _resample(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or arr.size == 0:
        return arr.astype("float32")
    try:  # high quality if torchaudio is present (it is, via torch)
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(arr.astype("float32")).unsqueeze(0)
        out = AF.resample(t, src_sr, dst_sr).squeeze(0).numpy()
        return out.astype("float32")
    except Exception:  # noqa: BLE001 — fall back to linear interp
        n = int(round(arr.size * dst_sr / src_sr))
        if n <= 0:
            return np.zeros(0, dtype="float32")
        xp = np.linspace(0, 1, arr.size, endpoint=False)
        x = np.linspace(0, 1, n, endpoint=False)
        return np.interp(x, xp, arr).astype("float32")


class ChatterboxEngine:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self.device = "cpu"
        try:
            import backends
            backend = backends.current()
            self.device = backend.torch_device
        except Exception:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass

    def _move_submodules_to_device(self, model, device: str):
        """ChatterboxTTS is not an nn.Module, so we manually move its submodules."""
        import torch
        moved = 0
        # Known submodule names in ChatterboxTTS
        for attr in ("t3", "s3gen", "ve", "watermarker"):
            submodule = getattr(model, attr, None)
            if submodule is not None and isinstance(submodule, torch.nn.Module):
                try:
                    submodule.to(device)
                    moved += 1
                except Exception as e:
                    print(f"[chatterbox] Failed to move {attr} to {device}: {e}", flush=True)
        return moved

    def _load(self):
        with self._lock:
            if self._model is None:
                from chatterbox.tts import ChatterboxTTS
                print(f"[chatterbox] Loading model (target device: {self.device})...", flush=True)
                try:
                    # Try loading directly on target device first
                    self._model = ChatterboxTTS.from_pretrained(device=self.device)
                    print(f"[chatterbox] Loaded directly on {self.device}", flush=True)
                except Exception as e:
                    print(f"[chatterbox] Direct load on {self.device} failed: {e}", flush=True)
                    print(f"[chatterbox] Loading on CPU then moving submodules...", flush=True)
                    self._model = ChatterboxTTS.from_pretrained(device="cpu")
                    moved = self._move_submodules_to_device(self._model, self.device)
                    # from_pretrained(device="cpu") sets self.device="cpu" inside
                    # the model. generate() uses self.device to place input tensors,
                    # so we must fix it to the real target or we get a device
                    # mismatch (weights on xpu, inputs on cpu).
                    self._model.device = self.device
                    if self._model.conds is not None:
                        self._model.conds = self._model.conds.to(self.device)
                    print(f"[chatterbox] Moved {moved} submodules to {self.device}", flush=True)
            return self._model

    def synth_chunk(self, text: str, voice: str = None, speed: float = 1.0, *,
                    exaggeration: float = 0.5, reference_path: str = None,
                    **_) -> np.ndarray:
        """Synthesize one chunk -> float32 mono @ 24 kHz. `reference_path` clones
        that voice; without it Chatterbox uses its built-in default voice."""
        model = self._load()
        kwargs = {"exaggeration": float(exaggeration), "cfg_weight": 0.5}
        if reference_path:
            kwargs["audio_prompt_path"] = reference_path
        wav = model.generate(text, **kwargs)
        try:
            arr = wav.detach().cpu().numpy()
        except AttributeError:
            arr = np.asarray(wav)
        arr = np.asarray(arr, dtype="float32").reshape(-1)
        sr = int(getattr(model, "sr", SAMPLE_RATE) or SAMPLE_RATE)
        return _resample(arr, sr, SAMPLE_RATE)

    def unload(self):
        """Drop the model and free VRAM so Kokoro / Ollama can reclaim it."""
        with self._lock:
            self._model = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
        except Exception:  # noqa: BLE001
            pass


ENGINE = ChatterboxEngine()
