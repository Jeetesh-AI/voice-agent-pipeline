"""Pluggable STT -> Streaming LLM -> Sentence-Chunked TTS voice pipeline."""

import asyncio
from dataclasses import dataclass
import os
import re
import time
from typing import AsyncIterator, Dict, List, Optional

import httpx


@dataclass
class VoiceTurnMetrics:
    stt_ms: float
    llm_ttft_ms: float
    tts_first_chunk_ms: float
    e2e_first_audio_ms: float
    total_turn_ms: float
    transcript: str
    reply_text: str
    audio_bytes: int


class STTEngine:
    """Speech-to-text stage (deterministic mock or Whisper API)."""

    async def transcribe(self, audio_bytes: bytes, hint_text: Optional[str] = None) -> str:
        await asyncio.sleep(0.085)  # ~85ms VAD + fast STT turn
        if hint_text:
            return hint_text.strip()
        return f"Transcribed {len(audio_bytes)} bytes of user speech."


class StreamingLLM:
    """Streams tokens from an OpenAI-compatible server or deterministic mock."""

    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        if self.base_url and self.api_key:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Reply concisely in 2 short sentences for voice playback."},
                            {"role": "user", "content": transcript},
                        ],
                        "max_tokens": 90,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    import json
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            chunk = json.loads(line[6:])
                            tok = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            if tok:
                                yield tok
                    return

        # Deterministic streaming fallback for local testing & CI
        await asyncio.sleep(0.11)  # ~110ms TTFT
        tokens = [
            "Got ", "it. ", "Your ", "order ", "is ", "confirmed ", "for ", "tomorrow ", "morning. ",
            "You ", "will ", "receive ", "a ", "tracking ", "link ", "shortly."
        ]
        for tok in tokens:
            await asyncio.sleep(0.015)
            yield tok


class TTSEngine:
    """Synthesizes sentence chunks into audio frames (PCM/WAV placeholder bytes)."""

    async def synthesize_sentence(self, sentence: str) -> bytes:
        await asyncio.sleep(0.045)  # ~45ms neural TTS first-chunk synthesis
        return f"AUDIO[{sentence}]".encode("utf-8")


class VoicePipeline:
    """Orchestrates STT -> LLM -> TTS with sentence-level streaming."""

    def __init__(self) -> None:
        self.stt = STTEngine()
        self.llm = StreamingLLM()
        self.tts = TTSEngine()

    async def run_turn(
        self,
        audio_bytes: bytes,
        hint_text: Optional[str] = None,
        stream_sentences: bool = True,
    ) -> VoiceTurnMetrics:
        t0 = time.perf_counter()

        transcript = await self.stt.transcribe(audio_bytes, hint_text=hint_text)
        stt_ms = (time.perf_counter() - t0) * 1000.0

        llm_start = time.perf_counter()
        llm_ttft_ms: Optional[float] = None
        tts_first_chunk_ms: Optional[float] = None
        e2e_first_audio_ms: Optional[float] = None

        buffer = ""
        full_reply: List[str] = []
        audio_chunks: List[bytes] = []

        async for token in self.llm.stream_reply(transcript):
            if llm_ttft_ms is None:
                llm_ttft_ms = (time.perf_counter() - llm_start) * 1000.0
            buffer += token
            full_reply.append(token)

            if stream_sentences and re.search(r"[.!?]\s*$", buffer) and len(buffer.strip()) >= 6:
                tts_t0 = time.perf_counter()
                chunk = await self.tts.synthesize_sentence(buffer.strip())
                if tts_first_chunk_ms is None:
                    tts_first_chunk_ms = (time.perf_counter() - tts_t0) * 1000.0
                    e2e_first_audio_ms = (time.perf_counter() - t0) * 1000.0
                audio_chunks.append(chunk)
                buffer = ""

        if buffer.strip():
            tts_t0 = time.perf_counter()
            chunk = await self.tts.synthesize_sentence(buffer.strip())
            if tts_first_chunk_ms is None:
                tts_first_chunk_ms = (time.perf_counter() - tts_t0) * 1000.0
                e2e_first_audio_ms = (time.perf_counter() - t0) * 1000.0
            audio_chunks.append(chunk)

        total_turn_ms = (time.perf_counter() - t0) * 1000.0
        return VoiceTurnMetrics(
            stt_ms=round(stt_ms, 1),
            llm_ttft_ms=round(llm_ttft_ms or 0.0, 1),
            tts_first_chunk_ms=round(tts_first_chunk_ms or 0.0, 1),
            e2e_first_audio_ms=round(e2e_first_audio_ms or total_turn_ms, 1),
            total_turn_ms=round(total_turn_ms, 1),
            transcript=transcript,
            reply_text="".join(full_reply).strip(),
            audio_bytes=sum(len(c) for c in audio_chunks),
        )
