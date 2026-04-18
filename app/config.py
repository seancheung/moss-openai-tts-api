from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


AttnImpl = Literal["auto", "flash_attention_2", "sdpa", "eager"]
Variant = Literal["auto", "tts", "ttsd", "voicegen", "sfx"]
Quantization = Literal["none", "int8", "int4"]


VARIANT_DEFAULT_MODEL: dict[str, str] = {
    "tts": "OpenMOSS-Team/MOSS-TTS",
    "ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "voicegen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "sfx": "OpenMOSS-Team/MOSS-SoundEffect",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    moss_variant: Variant = Field(default="auto")
    moss_model: str = Field(default="")
    moss_device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    moss_cuda_index: int = Field(default=0)
    moss_dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(default="auto")
    moss_quantization: Quantization = Field(
        default="none",
        description=(
            "Load the backbone with bitsandbytes weight-only quantization. "
            "`int8` halves VRAM versus bf16; `int4` uses NF4 + double-quant "
            "and roughly quarters it. CUDA-only."
        ),
    )
    moss_audio_tokenizer_device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description=(
            "Where to place the MOSS-Audio-Tokenizer codec. `auto` follows the "
            "backbone; set to `cpu` to offload it (saves 1-2 GB VRAM at the "
            "cost of extra CPU work around each request)."
        ),
    )
    moss_audio_tokenizer_dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(
        default="auto",
        description=(
            "Codec precision. `auto` keeps the checkpoint's original dtype "
            "(upstream MOSS-Audio-Tokenizer has dtype-sensitive paths in its "
            "forward, so forcing bf16/fp16 may trigger dtype-mismatch errors "
            "at encode/decode time). If VRAM is tight, prefer "
            "`MOSS_AUDIO_TOKENIZER_DEVICE=cpu` instead."
        ),
    )
    moss_cache_dir: Optional[str] = Field(default=None)

    moss_attn_implementation: AttnImpl = Field(default="auto")

    moss_max_new_tokens: int = Field(default=4096, ge=64, le=16384)
    moss_sfx_tokens_per_second: float = Field(default=12.5, gt=0.0, le=100.0)
    moss_n_vq_for_inference: Optional[int] = Field(
        default=None,
        description=(
            "MossTTSLocal-only: RVQ depth at inference (e.g. 4/8/16/32). "
            "Ignored by MossTTSDelay models, which do not accept this kwarg."
        ),
    )

    moss_voices_dir: str = Field(default="/voices")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.moss_voices_dir)

    @property
    def resolved_variant(self) -> str:
        if self.moss_variant != "auto":
            return self.moss_variant
        stripped = (self.moss_model or "").lower().replace("_", "").replace("-", "")
        if "ttsd" in stripped:
            return "ttsd"
        if "voicegenerator" in stripped:
            return "voicegen"
        if "soundeffect" in stripped:
            return "sfx"
        return "tts"

    @property
    def resolved_model(self) -> str:
        if self.moss_model:
            return self.moss_model
        return VARIANT_DEFAULT_MODEL[self.resolved_variant]

    @property
    def resolved_device(self) -> str:
        import torch

        if self.moss_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.moss_cuda_index}"
            return "cpu"
        if self.moss_device == "cuda":
            return f"cuda:{self.moss_cuda_index}"
        return self.moss_device

    def resolved_dtype(self):
        import torch

        device = self.resolved_device
        if self.moss_dtype == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.moss_dtype]

    def resolved_audio_tokenizer_device(self, backbone_device: str) -> str:
        if self.moss_audio_tokenizer_device == "auto":
            return backbone_device
        if self.moss_audio_tokenizer_device == "cuda":
            return f"cuda:{self.moss_cuda_index}"
        return self.moss_audio_tokenizer_device

    def resolved_audio_tokenizer_dtype(self, backbone_dtype, device: str):
        """Returns a torch dtype to cast the codec to, or None to leave it alone.

        `auto` keeps the checkpoint's native dtype — MOSS-Audio-Tokenizer has
        dtype-sensitive forward paths that break on forced downcasts.
        """
        import torch

        if self.moss_audio_tokenizer_dtype == "auto":
            return None
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.moss_audio_tokenizer_dtype]

    def resolved_attn_impl(self, device: str, dtype) -> str:
        if self.moss_attn_implementation != "auto":
            return self.moss_attn_implementation

        import torch

        if device.startswith("cuda"):
            has_flash = importlib.util.find_spec("flash_attn") is not None
            supports_dtype = dtype in {torch.float16, torch.bfloat16}
            if has_flash and supports_dtype:
                try:
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        return "flash_attention_2"
                except Exception:
                    pass
            return "sdpa"
        return "eager"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
