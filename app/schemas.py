from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
VariantName = Literal["tts", "ttsd", "voicegen", "sfx"]


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request (variant=tts zero-shot cloning)."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Accepted for OpenAI compatibility; ignored (MOSS-TTS has no speed control).",
    )

    tokens: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Optional target audio length, in MOSS audio tokens (~40 ms each).",
    )
    max_new_tokens: Optional[int] = Field(
        default=None, ge=64, le=16384,
        description="Override MOSS_MAX_NEW_TOKENS for this request.",
    )
    n_vq: Optional[int] = Field(
        default=None, ge=1, le=32,
        description=(
            "MossTTSLocal-only: RVQ depth (typically 4/8/16/32). "
            "Higher = better fidelity, slower. Ignored by MossTTSDelay models."
        ),
    )


class DirectRequest(BaseModel):
    """`/v1/audio/direct` request — variant=tts, no reference audio."""

    input: str = Field(..., description="Text to synthesize.")
    response_format: ResponseFormat = Field(default="mp3")

    tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    max_new_tokens: Optional[int] = Field(default=None, ge=64, le=16384)
    n_vq: Optional[int] = Field(default=None, ge=1, le=32)


class DialogueRequest(BaseModel):
    """`/v1/audio/dialogue` request — variant=ttsd, multi-speaker dialogue."""

    input: str = Field(
        ...,
        description="Dialogue text with `[S1]`/`[S2]`/... speaker tags.",
    )
    voices: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional voice ids, in speaker order. `voices[0]` clones `[S1]`, "
            "`voices[1]` clones `[S2]`, and so on. Omit for a fully generated dialogue."
        ),
    )
    response_format: ResponseFormat = Field(default="mp3")

    max_new_tokens: Optional[int] = Field(default=None, ge=64, le=16384)


class DesignRequest(BaseModel):
    """`/v1/audio/design` request — variant=voicegen, text-prompted voice design."""

    input: str = Field(..., description="Text to synthesize.")
    instruction: str = Field(
        ..., description="Natural-language voice description, e.g. 'warm female, slow pace'.",
    )
    response_format: ResponseFormat = Field(default="mp3")

    tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    max_new_tokens: Optional[int] = Field(default=None, ge=64, le=16384)


class EffectRequest(BaseModel):
    """`/v1/audio/effects` request — variant=sfx, ambient sound effect generation."""

    input: str = Field(
        ...,
        description="Sound effect description (e.g. 'thunder rolls, heavy rain on metal roof').",
    )
    duration: Optional[float] = Field(
        default=None, ge=0.1, le=120.0,
        description="Target duration in seconds; converted to `tokens` at 12.5 tokens/second.",
    )
    response_format: ResponseFormat = Field(default="mp3")

    tokens: Optional[int] = Field(
        default=None, ge=1, le=16384,
        description="Explicit token budget; overrides `duration` when set.",
    )
    max_new_tokens: Optional[int] = Field(default=None, ge=64, le=16384)


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    variant: Optional[VariantName] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[Literal["none", "int8", "int4"]] = None
    attn_implementation: Optional[str] = None
    audio_tokenizer_device: Optional[str] = None
    audio_tokenizer_dtype: Optional[str] = None
    sample_rate: Optional[int] = None
