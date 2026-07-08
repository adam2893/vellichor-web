# Unraid (Community Applications)

Two templates are available:
- **NVIDIA GPU** → `vellichor-web.xml`
- **Intel Arc GPU** → `vellichor-web-arc.xml`

These XML files are for the Community Apps store (see "Getting into CA" below).
Until they're published, set up the container manually — follow the guide for your GPU below.

---

## Intel Arc GPU setup (Unraid 7.0+)

### Prerequisites

Unraid 7.0+ ships with the `i915` kernel driver — your Arc card should be
detected automatically.  Verify via SSH/terminal:

```bash
ls -la /dev/dri/renderD128       # must exist
lspci | grep -i vga              # should list your Arc card
uname -r                         # 6.6+ = good, 6.8+ for Battlemage B580
```

Install the **Intel GPU Top** plugin from Community Apps (gives you GPU stats
on the dashboard).  No other drivers needed.

### Step-by-step: Add Vellichor in the Unraid Docker GUI

Go to **Docker** tab → **Add Container**, then fill in these fields:

**Basic settings:**
| Field | Value |
|-------|-------|
| Name | `Vellichor-ARC` |
| Repository | `ghcr.io/woodscode/vellichor-web:latest` |
| Network Type | `Bridge` |
| Console shell command | `Shell` |

**Port mappings (Add):**
| Container Port | Host Port |
|---------------|-----------|
| 7777 | 7777 |

**Path mappings (Add):**
| Container Path | Host Path | Access |
|---------------|-----------|--------|
| `/data` | `/mnt/user/appdata/vellichor-arc/data` | Read/Write |
| `/library` | *(your Audiobookshelf library path, or leave empty)* | Read/Write |

**Variables (Add):**
| Key | Value | Notes |
|-----|-------|-------|
| `VELLICHOR_PASSWORD` | `your-password-here` | Web UI login |
| `SECRET_KEY` | *(generate with `openssl rand -hex 32`)* | Session key, optional |
| `VELLICHOR_GPU_BACKEND` | `openvino` | Tells Vellichor to use Intel GPU |
| `ABS_UID` | `99` | File ownership for exports (Unraid nobody) |
| `ABS_GID` | `100` | File ownership for exports (Unraid users) |

**Extra Parameters (toggle Advanced View top-right to see this):**
```
--device=/dev/dri
```

That's it — no `--runtime=nvidia`, no NVIDIA variables.  Click **Apply**.

Once running, browse to `http://<unraid-ip>:7777`.  The header should show
**🔷 Intel OpenVINO / XPU** — that confirms GPU acceleration is working.

### Optional: Smart cast (AI speaker attribution) with Ollama

Smart cast is the AI feature that reads your story and figures out who's
speaking each line, then auto-assigns voices.  It needs a separate Ollama
container running alongside Vellichor.

**1. Add the Ollama container** (Docker tab → Add Container):

| Field | Value |
|-------|-------|
| Name | `ollama` |
| Repository | `ollama/ollama:latest` |
| Network Type | `Bridge` |

| Container Port | Host Port |
|---------------|-----------|
| 11434 | 11434 |

| Container Path | Host Path |
|---------------|-----------|
| `/root/.ollama` | `/mnt/user/appdata/ollama` |

Extra Parameters:
```
--device=/dev/dri --cap-add=CAP_PERFMON
```

Click **Apply**.

**2. Pull the model.**  After the container starts, open a terminal on Unraid:
```bash
docker exec ollama ollama pull llama3.2:3b
```

**3. Point Vellichor at it.**  Edit the Vellichor container, add these variables:

| Key | Value |
|-----|-------|
| `OLLAMA_URL` | `http://<unraid-ip>:11434` |
| `SMARTCAST_MODEL` | `llama3.2:3b` |

Replace `<unraid-ip>` with your server's IP (e.g. `192.168.1.50`).

Click **Apply**.  Smart cast will now use the Arc card's Vulkan backend via Ollama.

> **Multiple GPUs?** If you have both an Intel iGPU and an Arc dGPU, add
> `GGML_VK_VISIBLE_DEVICES=0` as an Ollama variable to pick the right one
> (check `vulkaninfo --summary` to figure out which index is the dGPU).

### Without a GPU (CPU-only)

Same setup as above, but:
- Set `VELLICHOR_GPU_BACKEND` to `cpu`
- Clear **Extra Parameters** (leave empty)
- Everything works, just slower.

---

## NVIDIA GPU setup

### Prerequisites

Install the **Nvidia Driver** plugin from Community Apps.

### Step-by-step: Add Vellichor (NVIDIA)

Go to **Docker** tab → **Add Container**:

**Basic settings:**
| Field | Value |
|-------|-------|
| Name | `Vellichor` |
| Repository | `ghcr.io/woodscode/vellichor-web:latest` |
| Network Type | `Bridge` |

| Container Port | Host Port |
|---------------|-----------|
| 7777 | 7777 |

| Container Path | Host Path |
|---------------|-----------|
| `/data` | `/mnt/user/appdata/vellichor/data` |
| `/library` | *(your Audiobookshelf library, or leave empty)* |

**Variables:**
| Key | Value |
|-----|-------|
| `VELLICHOR_PASSWORD` | `your-password` |
| `SECRET_KEY` | *(generate: `openssl rand -hex 32`)* |
| `NVIDIA_VISIBLE_DEVICES` | `all` |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` |

**Extra Parameters:**
```
--runtime=nvidia
```

Click **Apply**.  Header shows `⚡ GPU`.

### Smart cast with Ollama (NVIDIA)

Add an Ollama container:

| Field | Value |
|-------|-------|
| Name | `ollama` |
| Repository | `ollama/ollama:latest` |
| Port | 11434 → 11434 |
| Path | `/root/.ollama` → `/mnt/user/appdata/ollama` |

Extra Parameters:
```
--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all
```

Pull model: `docker exec ollama ollama pull llama3.2:3b`

Point Vellichor at it: add `OLLAMA_URL` = `http://<unraid-ip>:11434`

> **GTX 10-series (Pascal)?** Recent Ollama dropped Pascal CUDA builds.
> Use `ollama/ollama:0.30.11` as the repository instead of `:latest`.

---

## Getting the templates into Community Applications

The XML files in this directory are CA store templates.  To get them into the
store so people can one-click install:

1. Make the GHCR package **public**: GitHub → Packages → `vellichor-web` →
   Package settings → change visibility to Public
2. Add an icon at `unraid/icon.png` (already done)
3. Create a support thread on the Unraid forums
4. Submit the repo to CA:
   <https://forums.unraid.net/topic/38582-plug-in-community-applications/>
