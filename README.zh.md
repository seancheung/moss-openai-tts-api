# MOSS-TTS OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 OpenMOSS 的 [MOSS-TTS 家族](https://github.com/OpenMOSS/MOSS-TTS) 做统一封装——旗舰语音克隆（`MOSS-TTS`）、多说话人对话（`MOSS-TTSD`）、文本描述音色设计（`MOSS-VoiceGenerator`）、音效生成（`MOSS-SoundEffect`）。四个模型都基于同一套 `MossTTSDelay` 架构，共享一个镜像；通过 `MOSS_VARIANT` 切换当前加载的模型。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体与 OpenAI SDK 一致
- **四个 MOSS 变体同一镜像**：`tts` / `ttsd` / `voicegen` / `sfx` 通过 `MOSS_VARIANT` 切换；当前变体不支持的端点会返回 `501`
- **零样本音色克隆**（tts）：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对
- **无参考 TTS**（tts）：`POST /v1/audio/direct`，完全不用参考音频
- **多说话人对话**（ttsd）：`POST /v1/audio/dialogue`，文本里用 `[S1]`/`[S2]` 标签，可按 speaker 顺序克隆音色
- **文本驱动音色设计**（voicegen）：`POST /v1/audio/design`，用自然语言 `instruction` 描述目标音色
- **音效生成**（sfx）：`POST /v1/audio/effects`，描述环境音并可指定 `duration`
- **2 个镜像**：`cuda` 与 `cpu`
- **模型运行时下载**：不打包进镜像，HuggingFace 缓存目录挂载后可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

> **本服务不覆盖**：`MOSS-TTS-Realtime`（流式语音 agent）与 `MOSS-TTS-Nano`（架构不同）。`MOSS-TTS-Local-Transformer`（1.7B，`MossTTSLocal`）**已支持**，通过 `tts` 变体使用——其 processor/generate API 与 8B Delay checkpoint 完全兼容（见下）。

## 模型变体

每个变体只启用与之对应的端点。

| `MOSS_VARIANT` | 默认 `MOSS_MODEL` | 参数量 | 采样率 | 启用端点 |
|---|---|---:|---:|---|
| `tts`（默认） | `OpenMOSS-Team/MOSS-TTS` | 8B | 24 kHz | `POST /v1/audio/speech`、`POST /v1/audio/direct` |
| `ttsd` | `OpenMOSS-Team/MOSS-TTSD-v1.0` | 8B | 24 kHz | `POST /v1/audio/dialogue` |
| `voicegen` | `OpenMOSS-Team/MOSS-VoiceGenerator` | 1.7B | 24 kHz | `POST /v1/audio/design` |
| `sfx` | `OpenMOSS-Team/MOSS-SoundEffect` | 8B | 24 kHz | `POST /v1/audio/effects` |

`/healthz`、`/v1/audio/voices`、`/v1/audio/voices/preview` 对所有变体开放。`MOSS_VARIANT=auto`（默认）会根据 `MOSS_MODEL` 字符串推断；两者都未设置时默认加载 `tts`。

`tts` 变体同样接受 `MOSS_MODEL=OpenMOSS-Team/MOSS-TTS-Local-Transformer`（1.7B，`MossTTSLocal` 架构）。它与 8B Delay checkpoint 共用同一套 API，是 CPU 部署的推荐选项，或者在 GPU 上用它换速度。Local 额外支持通过 `MOSS_N_VQ_FOR_INFERENCE` 或请求字段 `n_vq` 控制 RVQ 深度，常用 4 / 8 / 16 / 32——越大越慢但越保真。该参数在 8B Delay checkpoint 上会被静默忽略（Delay 架构的 `generate()` 不接受此 kwarg）。

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/moss-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/moss-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备音色目录（仅 tts / ttsd 使用）

```
voices/
├── alice.wav     # 参考音频，单声道，16kHz 以上，推荐 3-20 秒
├── alice.txt     # UTF-8 纯文本，alice.wav 的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才算有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。`voicegen` 和 `sfx` 不使用音色目录。

`tts` 变体推理**只用 `.wav`** 做音色条件，`.txt` 通过 `/v1/audio/voices` 作为元数据返给前端展示，不会传给模型。`ttsd` 变体会按 `voices` 顺序把每个说话人的 wav 编码为参考 audio codes。

### 2. 运行容器

旗舰 TTS（GPU）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e MOSS_VARIANT=tts \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

对话（MOSS-TTSD）：

```bash
docker run --rm -p 8001:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e MOSS_VARIANT=ttsd \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

音色设计（MOSS-VoiceGenerator）：

```bash
docker run --rm -p 8002:8000 --gpus all \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e MOSS_VARIANT=voicegen \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

音效（MOSS-SoundEffect）：

```bash
docker run --rm -p 8003:8000 --gpus all \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e MOSS_VARIANT=sfx \
  ghcr.io/seancheung/moss-openai-tts-api:cuda-latest
```

CPU 版本（仅用于功能验证，见局限）：

```bash
docker run --rm -p 8010:8000 \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e MOSS_VARIANT=voicegen \
  ghcr.io/seancheung/moss-openai-tts-api:latest
```

首次启动从 HuggingFace 下载模型权重（8B 模型 bf16 约 16 GB，VoiceGenerator 约 3.5 GB）。挂载 `/root/.cache/huggingface` 可让权重在容器重启后复用。

> **GPU 要求**：宿主机需 NVIDIA 驱动 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 用 Docker Desktop + WSL2 + NVIDIA Windows 驱动。8B 模型 bf16 约需 24 GB 显存，VoiceGenerator 约 6 GB。

### 3. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)——每个变体一个 service。

## API 用法

服务默认监听 `8000`。下面的所有业务端点都只在对应变体被加载时启用；当前变体不匹配的端点会返回 `501 Not Implemented`。`/healthz` 会告诉你当前加载的是哪个变体。

### GET `/v1/audio/voices`

列出可用音色（读取 `MOSS_VOICES_DIR`，所有变体可用）。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`  *（variant=tts）*

OpenAI TTS 兼容接口——**零样本音色克隆**。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moss-tts",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 `MAX_INPUT_CHARS` |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 为兼容 OpenAI SDK 保留，**实际忽略**（MOSS 无原生语速控制） |
| `tokens` | int | 可选，目标音频长度（MOSS 音频 token，约 40 ms / 个） |
| `max_new_tokens` | int | 可选，按请求覆盖 `MOSS_MAX_NEW_TOKENS` |
| `n_vq` | int | 可选 RVQ 深度（1-32）。仅当 `MOSS_MODEL` 为 `MossTTSLocal` checkpoint 时生效，Delay 会忽略。 |

### POST `/v1/audio/direct`  *（variant=tts）*

无参考 TTS——MOSS 随机采样音色。字段：`input`、`response_format`、`tokens`、`max_new_tokens`、`n_vq`（仅 Local 生效）。

### POST `/v1/audio/dialogue`  *（variant=ttsd）*

多说话人对话合成。在 `input` 文本中使用 `[S1]`、`[S2]`……标签区分说话人。可选 `voices` 列表按 speaker 顺序克隆音色（`voices[0]` → `[S1]`，以此类推）；省略 `voices` 时则全生成。

```bash
curl -s http://localhost:8001/v1/audio/dialogue \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "[S1] 准备好了吗？\n[S2] 完全没问题，我们开始吧。",
    "voices": ["alice", "bob"],
    "response_format": "mp3"
  }' \
  -o dialogue.mp3
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 含 `[S1]`/`[S2]`… 的对话文本 |
| `voices` | list[string] | 可选，按说话人顺序给出的音色 id |
| `response_format` | string | 同 `/speech` |
| `max_new_tokens` | int | 可选 |

### POST `/v1/audio/design`  *（variant=voicegen）*

基于自然语言描述的音色设计——无需参考音频。

```bash
curl -s http://localhost:8002/v1/audio/design \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "欢迎登机，请系好安全带。",
    "instruction": "温柔女声，语速平缓，略低沉",
    "response_format": "mp3"
  }' \
  -o design.mp3
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `instruction` | string | 音色描述，如 `"温柔女声，语速平缓"` |
| `response_format` | string | 同 `/speech` |
| `tokens` / `max_new_tokens` | int | 可选 |

### POST `/v1/audio/effects`  *（variant=sfx）*

音效生成。

```bash
curl -s http://localhost:8003/v1/audio/effects \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "雷声滚动，大雨打在铁皮屋顶上",
    "duration": 8,
    "response_format": "mp3"
  }' \
  -o sfx.mp3
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 音效描述 |
| `duration` | float | 可选，目标时长（秒）；以 12.5 tokens/s 换算为 `tokens` |
| `tokens` | int | 可选，显式 token 预算，设置后优先于 `duration` |
| `response_format` | string | 同 `/speech` |
| `max_new_tokens` | int | 可选 |

### 使用 OpenAI Python SDK（variant=tts）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="moss-tts",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`tokens`、`max_new_tokens` 等扩展字段通过 `extra_body={...}` 传入。

### GET `/healthz`

返回模型名、当前变体、设备、dtype、注意力实现、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MOSS_VARIANT` | `auto` | `auto` 按 `MOSS_MODEL` 字符串推断；或强制为 `tts` / `ttsd` / `voicegen` / `sfx`。 |
| `MOSS_MODEL` | *（变体默认值）* | HuggingFace 仓库 id 或本地路径。为空时使用当前变体的默认模型。 |
| `MOSS_DEVICE` | `auto` | `auto` → 有 CUDA 则用 CUDA，否则 CPU；或 `cuda` / `cpu`。 |
| `MOSS_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `MOSS_DTYPE` | `auto` | `auto` → CUDA = bfloat16，CPU = float32；或 `bfloat16` / `float16` / `float32`。 |
| `MOSS_CACHE_DIR` | — | 加载模型前写入 `HF_HOME`，同时作为 Transformers 的 `cache_dir` |
| `MOSS_ATTN_IMPLEMENTATION` | `auto` | `auto` → CUDA SM≥80 + fp16/bf16 + 已安装 `flash_attn` 时选 `flash_attention_2`，否则 CUDA 下 `sdpa`、CPU 下 `eager`。亦可强制指定。 |
| `MOSS_MAX_NEW_TOKENS` | `4096` | `model.generate(max_new_tokens=...)` 上限 |
| `MOSS_N_VQ_FOR_INFERENCE` | — | MossTTSLocal 专属 RVQ 深度（1-32，常用 4/8/16/32），Delay 模型会忽略。 |
| `MOSS_SFX_TOKENS_PER_SECOND` | `12.5` | `sfx` 变体在给 `duration`（不给 `tokens`）时的换算系数 |
| `MOSS_VOICES_DIR` | `/voices` | 音色目录 |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t moss-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t moss-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **一个容器一个变体**。8B 模型无法共存，需要多个变体并存时请起多个容器（compose 示例里就是这样），再用反代合并入口。
- **CPU 镜像用于功能验证 / 轻量使用**。8B 变体（`tts` 走默认 8B Delay、`ttsd`、`sfx`）在 CPU fp32 下约需 32 GB 内存、一句短话要几分钟。CPU 下实用的组合：`MOSS_VARIANT=tts` + `MOSS_MODEL=OpenMOSS-Team/MOSS-TTS-Local-Transformer`（1.7B Local），或 `MOSS_VARIANT=voicegen`（1.7B）。生产环境请用 CUDA 镜像。
- **模型体积大**——8B bf16 约 16 GB，VoiceGenerator 约 3.5 GB。首次启动下载耗时，**务必**挂载 `/root/.cache/huggingface` 复用权重。
- **不内置 FlashAttention 2**。FA2 需要源码编译，CI 成本高。CUDA 镜像自动回退到 PyTorch SDPA。要启用 FA2，请在运行中的容器内 `pip install flash-attn` 并设置 `MOSS_ATTN_IMPLEMENTATION=flash_attention_2`。
- **cuDNN SDPA 被强制关闭**。按 MOSS 上游 Quickstart 要求，在引擎初始化时执行；flash / mem-efficient / math SDPA 后端仍保留做 fallback。
- **`speed` 字段是 no-op**：MOSS 无原生语速控制，保留该字段只为让 OpenAI Python SDK 的默认请求体（`speed=1.0`）不被 422。需要控长度请用 `tokens`，或对返回音频做后处理。
- **TTSD 的说话人标签**：`POST /v1/audio/dialogue` 要求 `input` 含 `[S1]`/`[S2]`… 标签（最多 5 个说话人）；`voices` 数量少于文本里的 speaker 数没关系，没对应 voice 的说话人会走纯生成。所有被克隆的说话人 wav 会重采样到 24 kHz 后再编码。
- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）。MOSS 是零样本，没有内置音色；需要用这些名字请在 `voices/` 放同名 `.wav` + `.txt`。
- **并发**：MOSS 单实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。需要流式请使用 MOSS-TTS-Realtime（独立模型，本服务未封装）。
- **`tts` 变体的 `.txt` 仅作元数据**。不同于姊妹项目 `voxcpm-openai-tts-api`，MOSS-TTS 不接受参考文本；`.txt` 只通过 `/v1/audio/voices` 返给客户端预览，**不会**传给模型。TTSD 内部本身会用到 transcript，但此服务目前走纯音频 continuation 模式，`.txt` 同样不参与推理。
- **需要 `trust_remote_code=True`**。MOSS 每个模型都把代码放在 HuggingFace 仓库内（上游要求）。生产环境建议把 `MOSS_MODEL` 锁到具体 revision 控制运行的代码版本。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。

## 目录结构

```
.
├── MOSS-TTS/                   # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py               # 端点 + variant 路由
│   ├── engine.py               # 模型加载 + 各 variant 推理
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)。
