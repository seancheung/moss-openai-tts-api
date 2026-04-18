from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .audio import CONTENT_TYPES, encode
from .config import get_settings
from .engine import TTSEngine, VariantNotSupported
from .schemas import (
    DesignRequest,
    DialogueRequest,
    DirectRequest,
    EffectRequest,
    HealthResponse,
    SpeechRequest,
    VoiceInfo,
    VoiceList,
)
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.engine = None
    try:
        app.state.engine = TTSEngine(settings)
    except Exception:
        log.exception("failed to load MOSS model")
        raise
    yield


app = FastAPI(
    title="MOSS-TTS OpenAI-TTS API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    engine: TTSEngine | None = request.app.state.engine
    if engine is None:
        return HealthResponse(
            status="loading",
            model=settings.resolved_model,
            variant=settings.resolved_variant,
        )
    return HealthResponse(
        status="ok",
        model=engine.model_id,
        variant=engine.variant,
        device=engine.device,
        dtype=engine.dtype_str,
        quantization=engine.quantization,
        attn_implementation=engine.attn_impl,
        sample_rate=engine.sample_rate,
    )


@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(request: Request) -> VoiceList:
    catalog: VoiceCatalog = request.app.state.catalog
    base = str(request.base_url).rstrip("/")
    voices = catalog.scan()
    data = [
        VoiceInfo(
            id=v.id,
            preview_url=f"{base}/v1/audio/voices/preview?id={quote(v.id, safe='')}",
            prompt_text=v.prompt_text,
        )
        for v in voices.values()
    ]
    return VoiceList(data=data)


@app.get("/v1/audio/voices/preview")
async def preview_voice(id: str, request: Request):
    catalog: VoiceCatalog = request.app.state.catalog
    voice = catalog.get(id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
    return FileResponse(
        path=str(voice.wav_path), media_type="audio/wav", filename=f"{id}.wav"
    )


def _validate_text(raw: str, max_chars: int) -> str:
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="input is empty")
    if len(text) > max_chars:
        raise HTTPException(
            status_code=413, detail=f"input exceeds {max_chars} chars",
        )
    return text


def _validate_format(fmt: str) -> None:
    if fmt not in CONTENT_TYPES:
        raise HTTPException(
            status_code=422, detail=f"unsupported response_format: {fmt}",
        )


def _require_voice(catalog: VoiceCatalog, voice_id: str):
    voice = catalog.get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    return voice


def _encode_response(samples, sample_rate: int, fmt: str) -> Response:
    try:
        audio_bytes, content_type = encode(samples, sample_rate, fmt)
    except Exception as e:
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e
    return Response(content=audio_bytes, media_type=content_type)


async def _run(coro):
    try:
        return await coro
    except VariantNotSupported as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e


@app.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)
    voice = _require_voice(catalog, body.voice)

    samples = await _run(engine.synthesize_clone(
        text,
        prompt_wav=str(voice.wav_path),
        tokens=body.tokens,
        max_new_tokens=body.max_new_tokens,
        n_vq=body.n_vq,
    ))
    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/direct")
async def create_direct(body: DirectRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    samples = await _run(engine.synthesize_direct(
        text,
        tokens=body.tokens,
        max_new_tokens=body.max_new_tokens,
        n_vq=body.n_vq,
    ))
    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/dialogue")
async def create_dialogue(body: DialogueRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    speaker_wavs: list[str] = []
    if body.voices:
        for vid in body.voices:
            voice = _require_voice(catalog, vid)
            speaker_wavs.append(str(voice.wav_path))

    samples = await _run(engine.synthesize_dialogue(
        text,
        speaker_wavs=speaker_wavs,
        max_new_tokens=body.max_new_tokens,
    ))
    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/design")
async def create_design(body: DesignRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    instruction = (body.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="instruction is empty")

    samples = await _run(engine.synthesize_design(
        text,
        instruction=instruction,
        tokens=body.tokens,
        max_new_tokens=body.max_new_tokens,
    ))
    return _encode_response(samples, engine.sample_rate, body.response_format)


@app.post("/v1/audio/effects")
async def create_effect(body: EffectRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    description = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    samples = await _run(engine.synthesize_effect(
        description,
        duration=body.duration,
        tokens=body.tokens,
        max_new_tokens=body.max_new_tokens,
    ))
    return _encode_response(samples, engine.sample_rate, body.response_format)
