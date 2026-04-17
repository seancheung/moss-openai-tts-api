# MOSS-TTS OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping the [MOSS-TTS family](https://github.com/OpenMOSS/MOSS-TTS) from OpenMOSS — flagship speech cloning (`MOSS-TTS`), multi-speaker dialogue (`MOSS-TTSD`), text-prompted voice design (`MOSS-VoiceGenerator`), and ambient sound effect generation (`MOSS-SoundEffect`). All four models share the same `MossTTSDelay` backbone and are served from a single image; the active model is selected with `MOSS_VARIANT`.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Four MOSS variants in one image** — switch between `tts` / `ttsd` / `voicegen` / `sfx` with `MOSS_VARIANT`; endpoints that do not apply to the active variant return `501`
- **Zero-shot voice cloning** (tts) — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory
- **Reference-free TTS** (tts) — `POST /v1/audio/direct` skips the reference audio entirely
- **Multi-speaker dialogue** (ttsd) — `POST /v1/audio/dialogue` with `[S1]`/`[S2]` tags and optional per-speaker voice cloning
- **Voice design from text** (voicegen) — `POST /v1/audio/design` with a natural-language voice `instruction`
- **Sound effect generation** (sfx) — `POST /v1/audio/effects` with an ambient sound description and optional `duration`
- **2 images** — `cuda` and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; HuggingFace cache is mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

> **Not covered by this service**: `MOSS-TTS-Realtime` (streaming voice agents) and `MOSS-TTS-Nano` (different architecture). `MOSS-TTS-Local-Transformer` (1.7B, `MossTTSLocal`) **is** supported via the `tts` variant — its processor/generate API is compatible with the 8B Delay checkpoint (see below).

## Model variants

Pick the variant that matches your use case; each variant only enables the endpoints it actually supports.

| `MOSS_VARIANT` | Default `MOSS_MODEL` | Size | Sample rate | Endpoints enabled |
|---|---|---:|---:|---|
| `tts` (default) | `OpenMOSS-Team/MOSS-TTS` | 8B | 24 kHz | `POST /v1/audio/speech`, `POST /v1/audio/direct` |
| `ttsd` | `OpenMOSS-Team/MOSS-TTSD-v1.0` | 8B | 24 kHz | `POST /v1/audio/dialogue` |
| `voicegen` | `OpenMOSS-Team/MOSS-VoiceGenerator` | 1.7B | 24 kHz | `POST /v1/audio/design` |
| `sfx` | `OpenMOSS-Team/MOSS-SoundEffect` | 8B | 24 kHz | `POST /v1/audio/effects` |

`/healthz`, `/v1/audio/voices` and `/v1/audio/voices/preview` are always available. `MOSS_VARIANT=auto` (the default) infers the variant from `MOSS_MODEL`; if both are unset the service loads `tts`.

The `tts` variant also accepts `MOSS_MODEL=OpenMOSS-Team/MOSS-TTS-Local-Transformer` (1.7B, `MossTTSLocal` architecture). It shares the same API as the 8B Delay checkpoint and is the recommended choice for CPU deployments or when you want to trade a bit of fidelity for speed. Local additionally supports RVQ depth control via `MOSS_N_VQ_FOR_INFERENCE` (or the `n_vq` request field); typical values are 4 / 8 / 16 / 32 — higher is slower but more faithful. The knob is silently ignored when running the 8B Delay checkpoint.

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/moss-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/moss-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the voices directory (tts / ttsd only)

```
voices/
├── alice.wav     # reference audio, mono, 16kHz+, ~3-20s recommended
├── alice.txt     # UTF-8 text: the transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. `voicegen` and `sfx` do not use the voices directory.

For `tts`, MOSS-TTS inference only uses the `.wav` for timbre conditioning — the `.txt` is surfaced via `/v1/audio/voices` as metadata but not passed to the model. For `ttsd`, the `.wav` of each listed voice is encoded and used as the per-speaker clone reference.

### 2. Run the container

Flagship TTS (GPU):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  -e MOSS_VARIANT=tts \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

Dialogue (MOSS-TTSD):

```bash
docker run --rm -p 8001:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  -e MOSS_VARIANT=ttsd \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

Voice design (MOSS-VoiceGenerator):

```bash
docker run --rm -p 8002:8000 --gpus all \
  -v $PWD/cache:/root/.cache \
  -e MOSS_VARIANT=voicegen \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

Sound effects (MOSS-SoundEffect):

```bash
docker run --rm -p 8003:8000 --gpus all \
  -v $PWD/cache:/root/.cache \
  -e MOSS_VARIANT=sfx \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

CPU (functional verification only, see caveats):

```bash
docker run --rm -p 8010:8000 \
  -v $PWD/cache:/root/.cache \
  -e MOSS_VARIANT=voicegen \
  ghcr.io/seancheung/moss-openai-tts-api:latest
```

Model weights are pulled from HuggingFace on first start (≈16 GB for the 8B models, ≈3.5 GB for VoiceGenerator). Mounting `/root/.cache` persists them across container restarts.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required. 8B variants need ~24 GB VRAM at bf16; VoiceGenerator needs ~6 GB.

### 3. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml) — it defines one service per variant.

## API usage

The service listens on port `8000` by default. Every endpoint below is only enabled when the active variant matches (`/healthz` will tell you which one is loaded). Endpoints that do not match the active variant return `501 Not Implemented`.

### GET `/v1/audio/voices`

List all usable voices (reads from `MOSS_VOICES_DIR`, available for every variant).

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`  *(variant=tts)*

OpenAI TTS-compatible endpoint — **zero-shot voice cloning**.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moss-tts",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (OpenAI SDK compat) |
| `input` | string | Text to synthesize, up to `MAX_INPUT_CHARS` characters |
| `voice` | string | Voice id from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Accepted but **ignored** (no native speed control in MOSS) |
| `tokens` | int | Optional target audio length, in MOSS audio tokens (~40 ms each) |
| `max_new_tokens` | int | Optional per-request override of `MOSS_MAX_NEW_TOKENS` |
| `n_vq` | int | Optional RVQ depth (1-32). Only effective when `MOSS_MODEL` is the `MossTTSLocal` checkpoint; ignored by Delay. |

### POST `/v1/audio/direct`  *(variant=tts)*

Reference-free TTS — MOSS samples a random voice. Fields: `input`, `response_format`, `tokens`, `max_new_tokens`, `n_vq` (Local-only).

### POST `/v1/audio/dialogue`  *(variant=ttsd)*

Multi-speaker dialogue synthesis. Use `[S1]` / `[S2]` / … tags in `input` to delimit turns. Pass a `voices` list to clone per-speaker timbres (`voices[0]` → `[S1]`, `voices[1]` → `[S2]`, …); omit `voices` for a fully generated dialogue.

```bash
curl -s http://localhost:8001/v1/audio/dialogue \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "[S1] Ready to start?\n[S2] Absolutely, let us go.",
    "voices": ["alice", "bob"],
    "response_format": "mp3"
  }' \
  -o dialogue.mp3
```

| Field | Type | Description |
|---|---|---|
| `input` | string | Dialogue text with `[S1]`/`[S2]`… tags |
| `voices` | list[string] | Optional; voice ids in speaker order |
| `response_format` | string | Same as `/speech` |
| `max_new_tokens` | int | Optional |

### POST `/v1/audio/design`  *(variant=voicegen)*

Voice design from a natural-language description — no reference audio required.

```bash
curl -s http://localhost:8002/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome aboard, please fasten your seatbelt.",
    "instruction": "warm female, calm pace, slightly low pitch",
    "response_format": "mp3"
  }' \
  -o design.mp3
```

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `instruction` | string | Voice description, e.g. `"warm female, calm pace"` |
| `response_format` | string | Same as `/speech` |
| `tokens` / `max_new_tokens` | int | Optional |

### POST `/v1/audio/effects`  *(variant=sfx)*

Ambient sound effect generation.

```bash
curl -s http://localhost:8003/v1/audio/effects \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Thunder rolls, heavy rain on a metal roof",
    "duration": 8,
    "response_format": "mp3"
  }' \
  -o sfx.mp3
```

| Field | Type | Description |
|---|---|---|
| `input` | string | Sound effect description |
| `duration` | float | Optional target duration in seconds; converted to `tokens` at 12.5 tokens/s |
| `tokens` | int | Explicit token budget; overrides `duration` when set |
| `response_format` | string | Same as `/speech` |
| `max_new_tokens` | int | Optional |

### Using the OpenAI Python SDK (variant=tts)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="moss-tts",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`tokens`, `max_new_tokens`) can be passed through `extra_body={...}`.

### GET `/healthz`

Returns model name, active variant, device, dtype, attention implementation, sample rate and status for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MOSS_VARIANT` | `auto` | `auto` infers from `MOSS_MODEL`; or force `tts` / `ttsd` / `voicegen` / `sfx`. |
| `MOSS_MODEL` | *(variant default)* | HuggingFace repo id or local path. When empty, defaults to the variant's flagship model. |
| `MOSS_DEVICE` | `auto` | `auto` → CUDA if available else CPU. Or `cuda` / `cpu`. |
| `MOSS_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `MOSS_DTYPE` | `auto` | `auto` → CUDA = bfloat16, CPU = float32. Or `bfloat16` / `float16` / `float32`. |
| `MOSS_CACHE_DIR` | — | Sets `HF_HOME` and the Transformers `cache_dir` before model load |
| `MOSS_ATTN_IMPLEMENTATION` | `auto` | `auto` → `flash_attention_2` if available (CUDA SM≥80 + fp16/bf16 + `flash_attn` installed), else `sdpa` on CUDA, else `eager`. Or force any of those. |
| `MOSS_MAX_NEW_TOKENS` | `4096` | Upper bound for `model.generate(max_new_tokens=...)` |
| `MOSS_N_VQ_FOR_INFERENCE` | — | MossTTSLocal-only RVQ depth (1-32, usually 4/8/16/32). Ignored by Delay models. |
| `MOSS_SFX_TOKENS_PER_SECOND` | `12.5` | Conversion factor used by `sfx` when `duration` is given instead of `tokens` |
| `MOSS_VOICES_DIR` | `/voices` | Voices directory |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t moss-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t moss-openai-tts-api:cpu .
```

## Caveats

- **One variant per container.** 8B MOSS models do not fit alongside each other; run separate containers for separate variants (the compose example does this) and route with a reverse proxy if you want a single URL.
- **CPU image is for functional verification / light use.** The 8B variants (`tts` with the default 8B Delay model, `ttsd`, `sfx`) need ~32 GB RAM at fp32 and take minutes per short clip on CPU. Practical CPU options: `MOSS_VARIANT=tts` + `MOSS_MODEL=OpenMOSS-Team/MOSS-TTS-Local-Transformer` (1.7B Local), or `MOSS_VARIANT=voicegen` (1.7B). For production use the CUDA image.
- **Model weights are large** — ~16 GB for 8B bf16, ~3.5 GB for VoiceGenerator. First-start downloads take a while; **always** mount `/root/.cache/huggingface`.
- **FlashAttention 2 is not bundled.** It has to be compiled from source, which is expensive on CI. The CUDA image falls back to PyTorch SDPA. To use FA2, install `flash-attn` into the running container and set `MOSS_ATTN_IMPLEMENTATION=flash_attention_2`.
- **cuDNN SDPA is force-disabled** at engine startup, following the MOSS-TTS upstream Quickstart. The flash / mem-efficient / math SDPA backends remain enabled as fallbacks.
- **`speed` is a no-op.** MOSS has no native speed control; the field is kept in `/v1/audio/speech` so that OpenAI's Python SDK default request body (`speed=1.0`) does not 422. Use `tokens` for length control, or post-process the returned audio.
- **TTSD speaker tags.** `POST /v1/audio/dialogue` expects `[S1]`/`[S2]`… tags in `input` (up to 5 speakers). Providing fewer `voices` entries than speakers in the text is fine — the missing speakers are fully generated. Every cloned speaker's reference wav is resampled to 24 kHz before encoding.
- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). MOSS is zero-shot; to get a stable voice under those names, drop `alloy.wav` + `alloy.txt` into `voices/`.
- **Concurrency**: a single MOSS instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Streaming is not supported** on the HTTP layer — each endpoint returns the complete audio when generation finishes. If you need streaming, look at MOSS-TTS-Realtime (a separate model, not wrapped by this service).
- **`.txt` is metadata-only for `tts`.** Unlike the sibling `voxcpm-openai-tts-api`, MOSS-TTS does not take a reference transcript; the `.txt` is surfaced in `/v1/audio/voices` for client display but is **not** passed to the model. TTSD does use transcripts internally, but this service currently invokes TTSD in the fully-audio continuation mode (the `.txt` is still unused).
- **`trust_remote_code=True` is required.** MOSS ships its model code inside each HuggingFace repo (upstream requirement). Pin `MOSS_MODEL` to a specific revision in production if you want to control which code runs.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.

## Project layout

```
.
├── MOSS-TTS/                   # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py               # endpoints + variant routing
│   ├── engine.py               # model loading + per-variant inference
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS).
