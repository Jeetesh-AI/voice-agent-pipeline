"""FastAPI service & interactive tester for the Voice Agent Pipeline."""

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.pipeline import VoicePipeline

app = FastAPI(title="Voice Agent Pipeline", version="1.0.0")
pipeline = VoicePipeline()


class VoiceTurnRequest(BaseModel):
    text_input: str = Field(..., min_length=1, max_length=500)
    stream_sentences: bool = True


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/voice/turn")
async def voice_turn(req: VoiceTurnRequest) -> Dict[str, Any]:
    res = await pipeline.run_turn(
        audio_bytes=req.text_input.encode("utf-8"),
        hint_text=req.text_input,
        stream_sentences=req.stream_sentences,
    )
    return {
        "transcript": res.transcript,
        "reply_text": res.reply_text,
        "audio_bytes": res.audio_bytes,
        "latency_ms": {
            "stt_ms": res.stt_ms,
            "llm_ttft_ms": res.llm_ttft_ms,
            "tts_first_chunk_ms": res.tts_first_chunk_ms,
            "e2e_first_audio_ms": res.e2e_first_audio_ms,
            "total_turn_ms": res.total_turn_ms,
        },
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Voice Agent Pipeline</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; background: #0d1117; color: #e6edf3; }
    h1 { color: #38bdf8; }
    .row { display: flex; gap: 8px; margin: 16px 0; }
    input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #30363d; background: #161b22; color: #e6edf3; }
    button { padding: 12px 18px; border-radius: 8px; border: 0; background: #0284c7; color: white; font-weight: 600; cursor: pointer; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 14px; }
  </style>
</head>
<body>
  <h1>Voice Agent Pipeline</h1>
  <p>Simulate a real-time STT → Streaming LLM → Sentence-Chunked TTS turn and inspect per-stage latency.</p>
  <div class="row">
    <input id="txt" value="Where is my order?" />
    <button onclick="runTurn()">Run Voice Turn</button>
  </div>
  <div id="out"></div>
  <script>
    async function runTurn() {
      const text_input = document.getElementById('txt').value;
      const r = await fetch('/v1/voice/turn', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text_input, stream_sentences: true })
      });
      const d = await r.json();
      const l = d.latency_ms;
      document.getElementById('out').innerHTML = `
        <div class="card">
          <div><strong>Reply:</strong> ${d.reply_text}</div>
          <p>STT: ${l.stt_ms} ms · LLM TTFT: ${l.llm_ttft_ms} ms · TTS: ${l.tts_first_chunk_ms} ms · <strong>First Audio: ${l.e2e_first_audio_ms} ms</strong></p>
        </div>`;
    }
  </script>
</body>
</html>"""
