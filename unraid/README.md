# Unraid (Community Applications)

Two templates are provided:
- **`vellichor-web.xml`** — NVIDIA GPU (default, CUDA)
- **`vellichor-web-arc.xml`** — Intel Arc GPU (OpenVINO / IPEX)

## Install it now (before it's in the CA store)

On your Unraid box:

1. **Docker** tab → **Add Container**.
2. Set **Template** to the raw URL of the template for your GPU:
   - NVIDIA: `https://raw.githubusercontent.com/woodscode/vellichor-web/main/unraid/vellichor-web.xml`
   - Intel Arc: `https://raw.githubusercontent.com/woodscode/vellichor-web/main/unraid/vellichor-web-arc.xml`
   …or paste the fields manually.
3. Set **Login password** (and ideally **Secret key** = `openssl rand -hex 32`),
   point **App data** at an appdata path, and **Audiobook library** at your
   library share (or clear it).
4. Apply, then browse to `http://<server-ip>:7777`.

---

## Intel Arc GPU setup (Unraid 7.0+)

### Prerequisites

Unraid 7.0+ ships with the `i915` kernel driver built in — your Arc card should
be detected automatically.  Verify:

```bash
# On your Unraid box (SSH or terminal):
ls -la /dev/dri/renderD128       # must exist
lspci | grep -i vga              # should list your Arc card
```

**Install the Intel GPU Top plugin** from Community Apps — it gives you live
GPU stats on the Unraid dashboard.

**No extra drivers needed.**  Unraid's kernel includes everything for Arc
cards (Alchemist and Battlemage).

### Install Vellichor (Intel Arc template)

1. **Docker tab** → **Add Container**
2. **Template:** paste
   `https://raw.githubusercontent.com/woodscode/vellichor-web/main/unraid/vellichor-web-arc.xml`
3. Fill in the required fields:
   - **Login password** — your web UI password
   - **App data (/data)** — e.g. `/mnt/user/appdata/vellichor-arc/data`
   - **GPU backend** — leave as `openvino`
4. **Extra Parameters** should already be set to `--device=/dev/dri`
   (this passes the Arc GPU through to the container)
5. Click **Apply**

The container will start.  Open `http://<unraid-ip>:7777` — the header should
show **🔷 Intel OpenVINO / XPU** confirming GPU acceleration is active.

### Optional: Smart cast (AI speaker attribution) with Ollama

Smart cast uses a separate Ollama container.  On Intel Arc, Ollama uses its
**Vulkan backend** for GPU acceleration.

**1. Install Ollama from Community Apps**, or set it up manually:

- Docker tab → Add Container
- **Name:** `ollama`
- **Repository:** `ollama/ollama:latest`
- **Extra Parameters:** `--device=/dev/dri --cap-add=CAP_PERFMON`
- **Network Type:** bridge (or host)
- Add a **Path:** `/mnt/user/appdata/ollama` → `/root/.ollama`
- Add a **Port:** `11434` → `11434` (if using bridge mode)
- Apply

**2. Pull the model** (after the container starts):

```bash
docker exec ollama ollama pull llama3.2:3b
```

**3. Point Vellichor at Ollama.**  Edit the Vellichor container, set
**Ollama URL** to `http://<unraid-ip>:11434` (use the host IP, not
`localhost`, since they're separate containers).  Apply.

**4. Verify GPU offload** — after pulling the model, run a test:

```bash
docker logs ollama 2>&1 | grep -i vulkan
# Looks for: "Vulkan" or "ggml_vulkan" — confirms GPU acceleration
```

> **Note:** If you have multiple GPUs (e.g. Intel iGPU + Arc dGPU), set
> `GGML_VK_VISIBLE_DEVICES=0` (or `1`) as an Ollama environment variable
> to pick the right one.

### Without a GPU (CPU-only)

1. Use the NVIDIA template and clear **Extra Parameters** (remove
   `--runtime=nvidia`)
2. Clear both `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`
3. Or on the Arc template: set **GPU backend** to `cpu` and clear
   **Extra Parameters**

All features work on CPU — just slower.

---

## AI Smart cast (optional) — NVIDIA

Smart cast uses a local **Ollama** LLM to attribute dialogue to speakers. The
Vellichor container does **not** include Ollama — you run it separately and
point Vellichor at it. Without it, multi-voice casting falls back to the
rule-based **Quick detect**, so this is entirely optional.

**1. Run an Ollama container.** Easiest on Unraid: install **Ollama** from
Community Applications. Or from the command line:
```bash
docker run -d --name ollama --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v /mnt/user/appdata/ollama:/root/.ollama \
  -p 11434:11434 ollama/ollama
```
Drop the `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` flags to run on CPU.

> **Pascal GPUs (GTX 10-series, e.g. the 1080):** recent Ollama dropped the
> Pascal CUDA build, so the model silently falls back to CPU. Pin a version
> that still bundles it — use `ollama/ollama:0.30.11` instead of `ollama/ollama`.

**2. Pull the model once** (use your Ollama container's name):
```bash
docker exec ollama ollama pull llama3.2:3b
```

**3. Point Vellichor at it:** set **Ollama URL** in the Vellichor template to
`http://<unraid-ip>:11434` — use the host IP, not `localhost`, since they're
separate containers.

On a single GPU, Vellichor hands VRAM back and forth between Kokoro and Ollama
automatically (it evicts whichever model is idle before the other runs). This
works across containers via Ollama's API, as long as the Ollama container also
has GPU access.

---

## Getting it into the Community Applications store
1. Make the GHCR package **public**: GitHub repo → **Packages** →
   `vellichor-web` → **Package settings** → change visibility to Public.
2. Add an **icon**: drop a square PNG at `unraid/icon.png` in this repo (the
   template's `<Icon>` already points there).
3. Create a **support thread** on the Unraid forums (the template references
   the GitHub issues page; CA prefers a forum support URL — update `<Support>`
   if you make one).
4. Submit the template repo to Community Applications via the official thread:
   <https://forums.unraid.net/topic/38582-plug-in-community-applications/> —
   follow "how to add your templates to CA". A moderator reviews it before it
   appears in the store.
