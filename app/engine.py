from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# Follow MOSS-TTS upstream guidance (README Quickstart): disable the broken
# cuDNN SDPA backend, keep the fallbacks enabled. This must happen before any
# model is loaded onto CUDA.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

log = logging.getLogger(__name__)


class VariantNotSupported(Exception):
    """Raised when a request targets an endpoint not enabled by the active variant."""


class TTSEngine:
    def __init__(self, settings):
        self.settings = settings

        self.variant = settings.resolved_variant
        self.model_id = settings.resolved_model
        self.device = settings.resolved_device
        self.dtype = settings.resolved_dtype()
        if self.device.startswith("cpu") and self.dtype != torch.float32:
            log.warning("cpu device detected; overriding dtype %s -> float32", self.dtype)
            self.dtype = torch.float32
        self.attn_impl = settings.resolved_attn_impl(self.device, self.dtype)

        log.info(
            "loading MOSS variant=%s model=%s device=%s dtype=%s attn=%s",
            self.variant, self.model_id, self.device, self.dtype, self.attn_impl,
        )

        # MOSS-TTS' remote-code processors reject cache_dir kwargs; the cache
        # location must be steered via HF_HOME before transformers is imported
        # (the entrypoint maps MOSS_CACHE_DIR -> HF_HOME for that reason).
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            attn_implementation=self.attn_impl,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        self.sample_rate = int(self.processor.model_config.sampling_rate)
        self.dtype_str = str(self.dtype).replace("torch.", "")
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float32(audio_tensor) -> np.ndarray:
        t = audio_tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        arr = t.numpy()
        if arr.ndim == 2:
            arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.reshape(-1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    def _require_variant(self, *allowed: str) -> None:
        if self.variant not in allowed:
            raise VariantNotSupported(
                f"active variant '{self.variant}' does not support this endpoint "
                f"(requires one of: {', '.join(allowed)})"
            )

    def _load_wav_mono(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(str(Path(wav_path).expanduser()))
        if wav.numel() == 0:
            raise ValueError(f"reference audio is empty: {wav_path}")
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
        return wav

    def _run_conversation(
        self, conversations, mode: str, max_new_tokens: int, **generate_kwargs,
    ):
        batch = self.processor(conversations, mode=mode)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        messages = self.processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("MOSS model returned no decoded audio")
        return messages[0].audio_codes_list[0]

    def _speech_generate_kwargs(self, n_vq: Optional[int]) -> dict:
        effective = n_vq if n_vq is not None else self.settings.moss_n_vq_for_inference
        return {"n_vq_for_inference": effective} if effective is not None else {}

    # ------------------------------------------------------------------
    # variant=tts
    # ------------------------------------------------------------------
    def _sync_speech_clone(self, text, prompt_wav, tokens, max_new_tokens, n_vq):
        kwargs = {"text": text, "reference": [prompt_wav]}
        if tokens is not None:
            kwargs["tokens"] = tokens
        msg = self.processor.build_user_message(**kwargs)
        return self._run_conversation(
            [[msg]], "generation", max_new_tokens, **self._speech_generate_kwargs(n_vq),
        )

    def _sync_speech_direct(self, text, tokens, max_new_tokens, n_vq):
        kwargs = {"text": text}
        if tokens is not None:
            kwargs["tokens"] = tokens
        msg = self.processor.build_user_message(**kwargs)
        return self._run_conversation(
            [[msg]], "generation", max_new_tokens, **self._speech_generate_kwargs(n_vq),
        )

    async def synthesize_clone(
        self, text: str, *, prompt_wav: str,
        tokens: Optional[int] = None, max_new_tokens: Optional[int] = None,
        n_vq: Optional[int] = None,
    ) -> np.ndarray:
        self._require_variant("tts")
        mnt = max_new_tokens or self.settings.moss_max_new_tokens
        async with self._lock:
            audio = await asyncio.to_thread(
                self._sync_speech_clone, text, prompt_wav, tokens, mnt, n_vq,
            )
        return self._to_float32(audio)

    async def synthesize_direct(
        self, text: str, *,
        tokens: Optional[int] = None, max_new_tokens: Optional[int] = None,
        n_vq: Optional[int] = None,
    ) -> np.ndarray:
        self._require_variant("tts")
        mnt = max_new_tokens or self.settings.moss_max_new_tokens
        async with self._lock:
            audio = await asyncio.to_thread(
                self._sync_speech_direct, text, tokens, mnt, n_vq,
            )
        return self._to_float32(audio)

    # ------------------------------------------------------------------
    # variant=ttsd (multi-speaker dialogue)
    # ------------------------------------------------------------------
    def _sync_dialogue(self, text, speaker_wavs, max_new_tokens):
        if not speaker_wavs:
            msg = self.processor.build_user_message(text=text)
            return self._run_conversation([[msg]], "generation", max_new_tokens)

        wavs = [self._load_wav_mono(p) for p in speaker_wavs]
        encoded = self.processor.encode_audios_from_wav(
            wavs, sampling_rate=self.sample_rate,
        )
        reference_codes = [codes for codes in encoded]
        concat_wav = torch.cat(wavs, dim=-1)
        prompt_audio = self.processor.encode_audios_from_wav(
            [concat_wav], sampling_rate=self.sample_rate,
        )[0]

        user_msg = self.processor.build_user_message(
            text=text, reference=reference_codes,
        )
        assistant_msg = self.processor.build_assistant_message(
            audio_codes_list=[prompt_audio],
        )
        return self._run_conversation(
            [[user_msg, assistant_msg]], "continuation", max_new_tokens,
        )

    async def synthesize_dialogue(
        self, text: str, *,
        speaker_wavs: Optional[list[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> np.ndarray:
        self._require_variant("ttsd")
        mnt = max_new_tokens or self.settings.moss_max_new_tokens
        async with self._lock:
            audio = await asyncio.to_thread(
                self._sync_dialogue, text, list(speaker_wavs or []), mnt,
            )
        return self._to_float32(audio)

    # ------------------------------------------------------------------
    # variant=voicegen
    # ------------------------------------------------------------------
    def _sync_design(self, text, instruction, tokens, max_new_tokens):
        kwargs = {"text": text, "instruction": instruction}
        if tokens is not None:
            kwargs["tokens"] = tokens
        msg = self.processor.build_user_message(**kwargs)
        return self._run_conversation([[msg]], "generation", max_new_tokens)

    async def synthesize_design(
        self, text: str, *, instruction: str,
        tokens: Optional[int] = None, max_new_tokens: Optional[int] = None,
    ) -> np.ndarray:
        self._require_variant("voicegen")
        mnt = max_new_tokens or self.settings.moss_max_new_tokens
        async with self._lock:
            audio = await asyncio.to_thread(
                self._sync_design, text, instruction, tokens, mnt,
            )
        return self._to_float32(audio)

    # ------------------------------------------------------------------
    # variant=sfx
    # ------------------------------------------------------------------
    def _sync_effect(self, description, tokens, max_new_tokens):
        kwargs = {"ambient_sound": description}
        if tokens is not None:
            kwargs["tokens"] = tokens
        msg = self.processor.build_user_message(**kwargs)
        return self._run_conversation([[msg]], "generation", max_new_tokens)

    async def synthesize_effect(
        self, description: str, *,
        duration: Optional[float] = None,
        tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> np.ndarray:
        self._require_variant("sfx")
        if tokens is None and duration is not None:
            tps = self.settings.moss_sfx_tokens_per_second
            tokens = max(1, int(round(float(duration) * tps)))
        mnt = max_new_tokens or self.settings.moss_max_new_tokens
        async with self._lock:
            audio = await asyncio.to_thread(
                self._sync_effect, description, tokens, mnt,
            )
        return self._to_float32(audio)
