# Vellichor 🎧

A self-hosted web app that turns your written stories (and ebooks) into
narrated audiobooks. Fast by default with the **Kokoro-82M** TTS model, plus an
optional **expressive engine (Chatterbox)** that adds emotional intensity and
**voice cloning** — clone a voice from a short clip, or record your own right in
the browser. GPU-accelerated and light enough to run on a modest card; falls
back to CPU. Open source (MIT).

### GPU Support

| Backend | Vendor | Status |
|---------|--------|--------|
| **CUDA** | NVIDIA (GeForce, Quadro, Tesla) | ✅ Mature, recommended |
| **OpenVINO / XPU** | Intel Arc A-series, Intel iGPU (Iris Xe, UHD) | ✅ Supported |
| **Vulkan** | AMD Radeon, cross-vendor | ⚠️ Experimental |
| **CPU** | Any | ✅ Always available |

## Requirements
- **Docker** and **Docker Compose**.
- **GPU (optional):**
  - **NVIDIA GPU** — requires the NVIDIA Container Toolkit on the host (Unraid:
    **Nvidia Driver** plugin).
  - **Intel Arc / iGPU** — requires Intel GPU drivers on the host + `/dev/dri`
    passthrough.
  - **AMD GPU** — requires AMDGPU drivers + Mesa Vulkan on the host.
  With no GPU it runs on CPU instead — see the CPU-only note in *Getting started*.
- **Disk:** ~7 GB for the Docker image, plus models downloaded on first use —
  Kokoro (small), the optional Ollama Smart-cast LLM (~2 GB), and the optional
  Chatterbox expressive model (~1–2 GB). Budget ~12 GB total to use everything.

## Getting started

### NVIDIA (default)
```bash
# 1. Clone
git clone https://github.com/woodscode/vellichor-web.git
cd vellichor-web

# 2. Create your .env: set a login password and a cookie-signing key
cp .env.example .env
sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$(openssl rand -hex 32)|" .env
$EDITOR .env                       # set VELLICHOR_PASSWORD

# 3. Edit docker-compose.yml for your box:
#    - Audiobookshelf export mount (…:/library) — repoint to your library, or
#      remove the volume if you don't use Audiobookshelf
#    - host port (default 7777:7777)
#    - NO GPU? set TORCH_BACKEND=cpu in .env and remove `runtime: nvidia` + `NVIDIA_*` lines

# 4. Build & start
docker compose up -d --build

# 5. (Optional) enable AI Smart cast — pull the local LLM once
docker exec vellichor-ollama ollama pull llama3.2:3b
```
Then open **http://<server-ip>:7777** and log in with your `VELLICHOR_PASSWORD`.

### Intel Arc / iGPU (OpenVINO)

Vellichor runs on Intel GPUs via two backends that share the card:
- **Kokoro + Chatterbox (TTS)** → Intel IPEX / OpenVINO (`xpu` device)
- **Ollama (Smart cast)** → Ollama's built-in Vulkan backend

Both get full GPU acceleration. The 12 GB on an Arc B580 comfortably fits
Kokoro (~2 GB) and the Smart cast LLM (~2 GB) with room to spare.

#### Step 1 — Install Intel GPU drivers on the host

**Ubuntu / Debian:**
```bash
# Add Intel's GPU package repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt update
sudo apt install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libze1 libmfx1

# Install Vulkan support (needed by Ollama)
sudo apt install -y mesa-vulkan-drivers vulkan-tools

# Verify the GPU is visible
sudo intel_gpu_top          # should show your Arc card
ls -la /dev/dri/renderD128  # must exist
vulkaninfo --summary | grep -i "GPU"   # Vulkan device listing
```

**Other distros:** install the equivalent of `intel-compute-runtime`,
`intel-level-zero-gpu`, `level-zero`, and `mesa-vulkan-drivers`.

**Kernel:** 6.2+ recommended for Arc dGPUs (Battlemage B580 works from 6.8+).
Check with `uname -r`.

#### Step 2 — Set up render group permissions

Docker containers access the GPU through `/dev/dri/renderD128`. Your user
and the `ollama` process need membership in the `render` and `video` groups:

```bash
sudo usermod -aG render,video $USER
# Check numeric group IDs (needed for docker-compose group_add)
stat -c '%g %n' /dev/dri/renderD128
stat -c '%g %n' /dev/dri/card0

# Log out and back in for group changes to take effect
```

#### Step 3 — Build and start Vellichor

```bash
cd vellichor-web

# Set the backend in .env
echo "TORCH_BACKEND=openvino" >> .env

# Build and start with the Arc profile
docker compose -f docker-compose.yml -f docker-compose.arc.yml build
docker compose -f docker-compose.yml -f docker-compose.arc.yml up -d

# Pull the Smart cast LLM (uses Ollama's Vulkan backend on the Arc card)
docker exec vellichor-ollama ollama pull llama3.2:3b
```

#### Step 4 — Confirm it's working

```bash
# Check that the backend activated correctly
docker logs vellichor-web 2>&1 | grep -i "backends"
# Expected: [backends] Activated: Intel OpenVINO / XPU (device=xpu)

# Check the health endpoint (shows which backend is active)
curl -s http://localhost:7777/healthz | python3 -m json.tool
# Expected: {"ok": true, "device": "xpu", "backend": {"id": "openvino", ...}}

# Verify Ollama sees the Vulkan GPU
docker exec vellichor-ollama ollama ps
docker exec vellichor-ollama ollama run llama3.2:3b "hello" --verbose 2>&1 | head -5
# Look for "Vulkan" in the output — confirms GPU offload
```

The web UI header will show `🔷 Intel OpenVINO / XPU` instead of the `⚡ GPU`
chip. Smart cast, live preview, voice cloning (Chatterbox), and multi-voice
narration all work on the Intel GPU.

**Troubleshooting:**
- No `/dev/dri` → GPU drivers aren't installed or loaded. Check
  `dmesg | grep -i i915` and `lspci | grep -i vga`.
- Backend falls back to CPU → the render group isn't set on the Docker
  user, or `intel-extension-for-pytorch` didn't find an XPU device.
  Run `docker exec vellichor-web python3 -c "import torch; print(torch.xpu.is_available())"`.
- Ollama runs on CPU → Vulkan drivers missing on the host. Install
  `mesa-vulkan-drivers` and verify with `vulkaninfo`.
- "Permission denied" on `/dev/dri` → check `group_add` numeric IDs in
  `docker-compose.arc.yml` match your system's `stat` output.

### AMD (Vulkan / ROCm)

Set `TORCH_BACKEND=vulkan` in `.env`, then use the AMD profile:
```bash
docker compose -f docker-compose.yml -f docker-compose.amd.yml build
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
```

**Host prerequisites:**
- `amdgpu` kernel module + Mesa (`mesa-vulkan-drivers` for radeonsi)
- `/dev/dri/renderD128` and `/dev/kfd` must exist
- For better performance with ROCm (Radeon RX 6000+), replace the Dockerfile's
  vulkan stage with a ROCm PyTorch build

> ⚠️ **Vulkan note:** PyTorch's Vulkan backend is experimental. Some TTS ops may
> fall back to CPU. For production AMD GPU use, consider building against ROCm
> PyTorch wheels (`pytorch/pytorch:2.6.0-rocm6.2.4`).

### CPU only
```bash
# Set TORCH_BACKEND=cpu in .env, or build directly:
docker build --build-arg TORCH_BACKEND=cpu -t vellichor-web:cpu .
# Then remove `runtime: nvidia` and `NVIDIA_*` from docker-compose.yml
```

## Features
- Built-in **story editor** (type/paste, `#` lines become chapters) + upload
  `.txt`, `.md`, `.epub`, `.pdf`, `.docx`.
- **Narration directives** — inline cues the studio interprets itself:
  `[pause 3s]` / `[pause]` / `[beat]` insert real silence, and `[slow]`,
  `[fast]`, `[normal]` change the pace of the following text. (These are
  reserved words — they're never read aloud or mistaken for a `[Name]` speaker
  tag.)
- **Voice picker** with 35 voices, grouped/filterable, each with a ▶ sample.
  Story-friendly voices are starred (★). `af_heart` is the default.
- **Live preview** — hear your chosen voice read the current text before
  committing to a full conversion.
- **Choose your TTS engine** (dropdown, per conversion):
  - **Kokoro** (default) — fast, lightweight, many preset voices.
  - **Chatterbox (expressive)** — richer, more lifelike delivery with an
    **Expressiveness** dial and **voice cloning**. Heavier (more VRAM, slower).
- **🎙️ Record / clone a voice** (Chatterbox) — record your own voice in the
  browser (read the on-screen script for ~15–20s) or upload a short clip, and
  save it to a reusable **My Voices** library. Narrate in *your* voice. (The mic
  needs a secure page — see *Expressive voices* below.)
- Reading-speed slider, optional cover art, author label.
- **Output loudness** control (Off / Standard / Loud / Extra loud) — applies
  EBU R128 loudness normalization so the finished book plays at a consistent,
  full volume, even on quiet speakers (e.g. a Toniebox). Defaults to *Loud*.
- **Live progress** (stage, segment count, ETA) + per-job log.
- **🎭 Multi-voice cast** — give each character their own voice. Three ways:
  - **🪄 Smart cast (AI)** — a local Ollama model (Llama 3.2 3B) reads the story,
    attributes each line to a speaker, and auto-inserts `[Name]` tags for you to
    review. Best for messy/untagged dialogue. Falls back to Quick detect if the
    model isn't ready.
  - **🔎 Quick detect** — fast rule-based: quotes + dialogue tags, with gender
    inference (honorifics + pronouns) to pick matching-gender voices.
  - **`[Name]` markup** — tag speakers yourself for exact control, e.g.
    `[Pip] "I can do it!"`.
  The cast panel lets you assign/preview a voice per character before converting.
  Works for **uploaded files too**: multi-voice auto-assigns distinct gender-matched
  voices to detected characters with zero setup, and 🔎/🪄 read the file's text so
  you can review/override the cast first.
- **🎵 Background ambience** — mix a bed under the narration: built-in
  license-free beds (Soft Rain, Gentle Night, Warm Hum, Dreamy Pad), or upload
  your own / drop files in `data/ambience/`. Volume slider + auto-ducking
  (music dips under speech).
- Output: **chaptered M4B** + **per-chapter MP3** (zip). Both downloadable.
- **Auto-export** to an Audiobookshelf library (path + owner UID/GID
  configurable; see `docker-compose.yml`).
- **🎨 Themes** — Dark, Light, Sepia, and Midnight, switchable from the header
  and remembered per browser.
- Password login; job history survives restarts.

## Managing it
```bash
cd vellichor-web                # your project directory
docker compose up -d            # start / apply changes
docker compose logs -f          # watch logs
docker compose down             # stop
docker compose up -d --build    # rebuild after editing app/ code
```

## AI Smart cast (Ollama)
Smart cast is **optional** — without it, multi-voice casting uses the
rule-based **Quick detect** instead. To enable it, the `ollama` service (in
docker-compose) runs a local LLM. After the first `docker compose up -d`, pull
the model once:
```bash
docker exec vellichor-ollama ollama pull llama3.2:3b
```
Both models share the GPU; `OLLAMA_KEEP_ALIVE=2m` unloads the LLM from VRAM
after use so Kokoro has room (on an 8 GB card they can't both stay resident).
To try a more accurate (heavier) model, pull it and set `SMARTCAST_MODEL` in
`.env` (e.g. `qwen2.5:7b`), then `up -d`.

## Expressive voices (Chatterbox)
Pick **Chatterbox (expressive)** from the *TTS engine* dropdown for more lifelike,
emotional narration. Two extra controls appear:
- **Expressiveness** — how animated the delivery is (0.5 is a good default).
- **Voice source** — clone a voice instead of using a preset:
  - **Preset** — clones the Kokoro voice picked on the left (zero setup).
  - **Record a voice** — record yourself in the browser (read the on-screen script
    for ~15–20s), review, then *Use for this book* or *Save* it to **My Voices**.
  - **Upload a clip** — a clean 5–30s single-speaker clip works best.

Saved voices form a **My Voices** library reusable across books, stored under
`data/voices/`. Cloning runs **locally on your GPU** — clips never leave the box.

Notes:
- **The microphone only works on a secure page** (`https://` or `localhost`). Over
  `http://<ip>:7777` the browser blocks the mic — put Vellichor behind an HTTPS
  reverse proxy (or use localhost). Uploading a clip works either way.
- Chatterbox is **heavier** than Kokoro (more VRAM, slower). It loads/unloads
  around Kokoro and the Ollama model to share an 8 GB card; its model (~1–2 GB)
  downloads on first use into `data/hf-cache`.
- Multi-voice cast is **Kokoro-only** for now.
- Chatterbox is MIT-licensed; outputs carry an inaudible Resemble "Perth"
  watermark by design (harmless for personal use).

## Configuration (`.env`)
- `VELLICHOR_PASSWORD` — login password (change anytime, then `up -d`).
- `SECRET_KEY` — session-cookie signing key (don't change or logins reset).
- `NOTIFY_URL` — optional. Set to an ntfy/gotify URL to get a push when a
  conversion finishes, e.g. `http://<server-ip>:8087/vellichor`.

## Data
- `./data/` — uploads, job workdirs, job history (`jobs.json`), cached voice
  samples (`samples/`), and the Hugging Face model cache (`hf-cache/`).
- Models download on first use and are cached in `./data/hf-cache`.

## Notes
- GPU is used automatically (`⚡ GPU` chip in the header). Falls back to CPU if
  the NVIDIA runtime is unavailable.
- Conversions run one at a time (single worker) so the GPU isn't oversubscribed.

## Security & deployment
- **Set a password.** Auth is a single shared password (`VELLICHOR_PASSWORD`).
  If it's left **blank, authentication is disabled entirely** — only do that on
  a trusted private network.
- **Don't expose it directly to the internet.** This is a self-hosted personal
  tool with a single-password gate, not a hardened multi-user service. If you
  need remote access, put it behind a reverse proxy (Nginx Proxy Manager,
  Traefik, Caddy) with HTTPS and ideally an extra auth layer (e.g. Authelia).
- **Keep `.env` private** (`chmod 600`). It holds your password and
  `SECRET_KEY` and is gitignored — never commit it.
- **`SECRET_KEY`** signs the session cookie. Generate one with
  `openssl rand -hex 32`. Changing it invalidates existing logins.
- **Uploaded files** (epub/pdf/docx) are parsed server-side; only allow
  uploads from people you trust.

## License
[MIT](LICENSE) — free to use, modify, and redistribute. TTS by
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0).
