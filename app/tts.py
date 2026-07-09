"""Kokoro TTS engine wrapper: pipeline caching, text chunking, synthesis,
and on-demand voice sample generation."""
import os
import re
import threading
import numpy as np
import soundfile as sf

import voices as voicecat

SAMPLE_RATE = 24000
SAMPLES_DIR = "/data/samples"


class Engine:
    def __init__(self):
        self._pipelines = {}
        self._lock = threading.Lock()
        self.device = "cpu"
        try:
            import backends
            backend = backends.current()
            self.device = backend.torch_device
        except Exception as e:
            print(f"[tts] Backend detection failed: {e}, will try torch fallback", flush=True)
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass
        print(f"[tts] Engine initialized with device: {self.device}", flush=True)
        os.makedirs(SAMPLES_DIR, exist_ok=True)

    def _move_kokoro_to_device(self, pipe, device: str) -> bool:
        """Directly move Kokoro's internal model to the target device.

        Kokoro's KPipeline stores the model in various places depending on version.
        We try multiple known locations and fall back to walking all attributes.
        """
        import torch

        moved_any = False

        # Try known attribute paths first
        paths_to_try = [
            "model",
            "model.bert", 
            "model.decoder",
            "model.bert_model",
            "model.diffusion",
            "g2p",
        ]

        for attr_path in paths_to_try:
            try:
                parts = attr_path.split(".")
                obj = pipe
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None and isinstance(obj, torch.nn.Module):
                    obj.to(device)
                    print(f"[tts] Moved pipe.{attr_path} to {device}", flush=True)
                    moved_any = True
            except Exception as e:
                print(f"[tts] Could not move pipe.{attr_path}: {e}", flush=True)

        # Also try the generic walker as backup
        if not moved_any:
            try:
                import backends
                count = backends.move_to_device(pipe, device)
                if count > 0:
                    moved_any = True
            except Exception as e:
                print(f"[tts] move_to_device fallback failed: {e}", flush=True)

        # Last resort: walk all attributes of pipe.model
        if not moved_any and hasattr(pipe, 'model'):
            try:
                model = pipe.model
                for attr_name in dir(model):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(model, attr_name)
                        if isinstance(attr, torch.nn.Module):
                            attr.to(device)
                            print(f"[tts] Moved model.{attr_name} to {device}", flush=True)
                            moved_any = True
                    except Exception:
                        pass
            except Exception as e:
                print(f"[tts] Attribute walk failed: {e}", flush=True)

        return moved_any

    def pipeline(self, lang_code: str):
        with self._lock:
            if lang_code not in self._pipelines:
                print(f"[tts] Creating KPipeline for lang_code={lang_code}", flush=True)
                from kokoro import KPipeline
                pipe = KPipeline(lang_code=lang_code)
                print(f"[tts] KPipeline created", flush=True)

                # Debug: inspect what Kokoro loaded
                if hasattr(pipe, 'model'):
                    import torch
                    model = pipe.model
                    print(f"[tts] pipe.model type: {type(model).__name__}", flush=True)
                    if hasattr(model, 'bert'):
                        print(f"[tts] model.bert type: {type(model.bert).__name__}", flush=True)
                    if hasattr(model, 'decoder'):
                        print(f"[tts] model.decoder type: {type(model.decoder).__name__}", flush=True)

                # Move model to real GPU if available (Kokoro only does CUDA/CPU)
                if self.device != "cpu":
                    print(f"[tts] Attempting to move model to {self.device}...", flush=True)
                    success = self._move_kokoro_to_device(pipe, self.device)
                    if success:
                        print(f"[tts] Model successfully moved to {self.device}", flush=True)
                    else:
                        print(f"[tts] WARNING: Could not move model to {self.device}, will run on CPU", flush=True)

                self._pipelines[lang_code] = pipe
                print(f"[tts] Pipeline cached for {lang_code}", flush=True)
            return self._pipelines[lang_code]

    def unload(self):
        """Drop cached pipelines and release GPU memory, so another model
        (Ollama, for Smart cast) can claim the VRAM. Pipelines reload lazily
        on the next synth."""
        with self._lock:
            count = len(self._pipelines)
            self._pipelines.clear()
            print(f"[tts] Unloaded {count} pipeline(s)", flush=True)
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
                print("[tts] XPU cache emptied", flush=True)
        except Exception as e:
            print(f"[tts] Cache clear error: {e}", flush=True)

    @staticmethod
    def clean_speech_text(text: str) -> str:
        """Strip Markdown markup so the TTS never vocalizes it (e.g. reading a
        heading '#' as 'hashtag', or '*' as 'asterisk'). Conservative: only
        removes formatting characters, never words or sentence punctuation."""
        if not text:
            return text
        # ATX headings: leading #'s at the start of a line (with or w/o a space)
        text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)
        # setext heading underlines (=== / --- on their own line)
        text = re.sub(r"(?m)^[ \t]{0,3}[=\-]{3,}[ \t]*$", "", text)
        # emphasis / code markers and blockquote arrows
        text = text.replace("`", "")
        text = re.sub(r"\*{1,3}", "", text)
        text = re.sub(r"(?m)^[ \t]{0,3}>[ \t]?", "", text)
        text = re.sub(r"(?<![A-Za-z0-9])_(?![A-Za-z0-9])", "", text)  # _emphasis_, not in_words
        return text

    @staticmethod
    def chunk_text(text: str, max_chars: int = 500):
        """Split text into synthesis-sized chunks at sentence boundaries."""
        text = text.strip()
        if not text:
            return []
        # split into sentences while keeping terminal punctuation
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        chunks, cur = [], ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > max_chars:
                # hard-wrap an over-long sentence on commas/spaces
                for piece in re.findall(r".{1," + str(max_chars) + r"}(?:\s|$)", s):
                    piece = piece.strip()
                    if piece:
                        chunks.append(piece)
                continue
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks

    def synth_chunk(self, text: str, voice: str, speed: float = 1.0, **_) -> np.ndarray:
        """Synthesize one chunk, returning a float32 mono waveform at 24kHz.
        Extra keyword args (exaggeration, reference_path) are accepted for a
        uniform cross-engine interface and ignored by Kokoro."""
        text = self.clean_speech_text(text)
        print(f"[tts] synth_chunk: voice={voice}, speed={speed}, text_len={len(text)}", flush=True)
        pipe = self.pipeline(voicecat.lang_code(voice))
        audio_parts = []
        chunk_idx = 0
        for _, _, audio in pipe(text, voice=voice, speed=speed):
            chunk_idx += 1
            if audio is None:
                print(f"[tts] Chunk {chunk_idx}: audio is None, skipping", flush=True)
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            audio_parts.append(arr.astype("float32"))
            print(f"[tts] Chunk {chunk_idx}: got audio shape={arr.shape}", flush=True)
        if not audio_parts:
            print("[tts] No audio parts generated!", flush=True)
            return np.zeros(0, dtype="float32")
        result = np.concatenate(audio_parts)
        print(f"[tts] synth_chunk complete: total_samples={len(result)}", flush=True)
        return result

    def sample_path(self, voice: str) -> str:
        return os.path.join(SAMPLES_DIR, f"{voice}.mp3")

    def ensure_sample(self, voice: str) -> str:
        """Generate (and cache) a short preview clip for a voice. Returns mp3 path."""
        out = self.sample_path(voice)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        import gpu
        with gpu.LOCK:
            gpu.release_ollama()        # reclaim VRAM from Smart cast before TTS
            print(f"[tts] Generating sample for voice={voice}", flush=True)
            wav = self.synth_chunk(voicecat.SAMPLE_TEXT, voice, speed=1.0)
        tmp_wav = out.replace(".mp3", ".wav")
        sf.write(tmp_wav, wav, SAMPLE_RATE)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-c:a", "libmp3lame", "-q:a", "5", out],
            check=True, capture_output=True,
        )
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        print(f"[tts] Sample saved: {out}", flush=True)
        return out

    def prewarm(self, voice_ids):
        """Generate samples for a list of voices in the background."""
        for vid in voice_ids:
            try:
                self.ensure_sample(vid)
            except Exception as e:  # noqa: BLE001
                print(f"[prewarm] {vid} failed: {e}", flush=True)


ENGINE = Engine()
