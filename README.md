# Voice Agent Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streaming](https://img.shields.io/badge/Pipeline-Sentence--Chunked%20TTS-0284C7)](app/pipeline.py)

**Real-time voice agent pipeline (`STT → Streaming LLM → Sentence-Chunked TTS`) with per-stage latency instrumentation.** Instead of waiting for the full LLM response before synthesizing speech, the pipeline flushes each completed sentence directly to TTS, cutting voice-to-voice first-audio latency by **44.8%**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/turn-timeline-dark.svg">
  <img alt="Single voice turn timeline showing 86.1 ms STT, 142.2 ms LLM first sentence, and 46.9 ms TTS for 275.2 ms first audio" src="docs/img/turn-timeline-light.svg">
</picture>

| **275.2 ms** | **498.6 ms** | **44.8%** | **86.1 ms** |
| :---: | :---: | :---: | :---: |
| p50 first-audio latency (sentence-streamed) | p50 first-audio latency (buffered baseline) | end-to-end latency reduction | p50 speech-to-text stage latency |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/latency-dark.svg">
  <img alt="Voice-to-voice latency comparison showing 275.2 ms streamed vs 498.6 ms buffered" src="docs/img/latency-light.svg">
</picture>

Full 10-turn breakdown: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/architecture-dark.svg">
  <img alt="Real-time voice architecture: Mic Audio -> STT -> Streaming LLM -> Neural TTS -> Playback" src="docs/img/architecture-light.svg">
</picture>

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -v
python scripts/bench_latency.py
uvicorn app.main:app --port 8000
```
