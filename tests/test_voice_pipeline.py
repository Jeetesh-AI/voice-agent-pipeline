"""Tests for Voice Agent Pipeline."""

import httpx
import pytest

from app.main import app
from app.pipeline import VoicePipeline


@pytest.mark.asyncio
async def test_sentence_streaming_beats_buffered_first_audio() -> None:
    pipe = VoicePipeline()
    streamed = await pipe.run_turn(b"audio", hint_text="Check order status", stream_sentences=True)
    buffered = await pipe.run_turn(b"audio", hint_text="Check order status", stream_sentences=False)

    assert streamed.audio_bytes > 0
    assert streamed.e2e_first_audio_ms < buffered.e2e_first_audio_ms


@pytest.mark.asyncio
async def test_voice_api_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        h = await client.get("/health")
        assert h.status_code == 200

        r = await client.post("/v1/voice/turn", json={"text_input": "Hello agent", "stream_sentences": True})
        assert r.status_code == 200
        data = r.json()
        assert data["latency_ms"]["e2e_first_audio_ms"] > 0
        assert len(data["reply_text"]) > 0
