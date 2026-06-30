# Unraid (Community Applications)

`vellichor-web.xml` is an Unraid Docker template for installing Vellichor from
the published GHCR image (`ghcr.io/woodscode/vellichor-web`).

## Install it now (before it's in the CA store)
On your Unraid box:
1. **Docker** tab → **Add Container**.
2. Set **Template** to the raw URL of this file, or paste the fields manually:
   `https://raw.githubusercontent.com/woodscode/vellichor-web/main/unraid/vellichor-web.xml`
3. Set **Login password** (and ideally **Secret key** = `openssl rand -hex 32`),
   point **App data** at an appdata path, and **Audiobook library** at your
   library share (or clear it).
4. **GPU:** keep `--runtime=nvidia` in *Extra Parameters* only if you have the
   **Nvidia Driver** plugin installed; otherwise clear it to run on CPU.
5. Apply, then browse to `http://<server-ip>:7777`.

## AI Smart cast (optional)
Smart cast needs a separate **Ollama** container (install the official
`ollama` app from CA). Pull a model in it once:
```bash
docker exec <ollama-container> ollama pull llama3.2:3b
```
Then set **Ollama URL** in the Vellichor template to that container, e.g.
`http://<ollama-ip>:11434`. Without it, casting uses the rule-based Quick detect.

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
