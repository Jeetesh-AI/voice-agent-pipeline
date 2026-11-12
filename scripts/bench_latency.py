"""Benchmarks Sentence-Streamed vs Buffered voice-to-voice latency and generates SVGs."""

import asyncio
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import VoicePipeline

THEMES = {
    "light": {"bg": "#ffffff", "card": "#f8fafc", "border": "#cbd5e1", "text": "#0f172a", "muted": "#475569", "accent": "#0284c7", "warn": "#f97316", "good": "#16a34a"},
    "dark": {"bg": "#0d1117", "card": "#161b22", "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e", "accent": "#38bdf8", "warn": "#fb923c", "good": "#4ade80"},
}


def build_latency_svg(summary: Dict[str, float], theme: str) -> str:
    t = THEMES[theme]
    w, h = 760, 260
    max_val = max(summary["buffered_first_audio_p50_ms"], summary["streamed_first_audio_p50_ms"], 1.0)
    rows = [
        ("Sentence-Streamed (First Audio)", summary["streamed_first_audio_p50_ms"], t["good"]),
        ("Buffered (Wait for Full LLM)", summary["buffered_first_audio_p50_ms"], t["warn"]),
    ]
    bars = []
    for i, (label, val, col) in enumerate(rows):
        y = 88 + i * 70
        bw = (val / (max_val * 1.15)) * 390
        bars.append(
            f'<text x="36" y="{y + 20}" fill="{t["text"]}" font-size="13" font-weight="600" font-family="ui-sans-serif, system-ui">{label}</text>'
            f'<rect x="275" y="{y}" width="390" height="30" rx="6" fill="{t["card"]}" stroke="{t["border"]}" />'
            f'<rect x="275" y="{y}" width="{bw:.1f}" height="30" rx="6" fill="{col}" />'
            f'<text x="678" y="{y + 20}" fill="{t["text"]}" font-size="13" font-weight="700" font-family="ui-sans-serif, system-ui">{val} ms</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" rx="12" fill="{t['bg']}" stroke="{t['border']}" />
  <text x="36" y="40" fill="{t['text']}" font-size="16" font-weight="700" font-family="ui-sans-serif, system-ui">
    Voice-to-Voice First-Audio Latency (p50 across 10 Turns · {summary['speedup_pct']}% Faster)
  </text>
  {''.join(bars)}
</svg>
"""


def build_arch_svg(theme: str) -> str:
    t = THEMES[theme]
    w, h = 760, 200
    boxes = [
        (24, 65, 110, 70, "Mic Audio", "VAD Chunk", t["muted"]),
        (165, 65, 120, 70, "STT", "~85 ms", t["accent"]),
        (315, 65, 145, 70, "Streaming LLM", "Sentence Buffer", t["good"]),
        (490, 65, 120, 70, "Neural TTS", "First Chunk", t["accent"]),
        (640, 65, 95, 70, "Playback", "Low Latency", t["good"]),
    ]
    nodes = "".join(
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{t["card"]}" stroke="{col}" stroke-width="2" />'
        f'<text x="{x + bw / 2:.1f}" y="{y + 28}" text-anchor="middle" fill="{t["text"]}" font-size="13" font-weight="700" font-family="ui-sans-serif, system-ui">{title}</text>'
        f'<text x="{x + bw / 2:.1f}" y="{y + 48}" text-anchor="middle" fill="{t["muted"]}" font-size="11" font-family="ui-sans-serif, system-ui">{sub}</text>'
        for x, y, bw, bh, title, sub, col in boxes
    )
    arrows = [(134, 100, 165, 100), (285, 100, 315, 100), (460, 100, 490, 100), (610, 100, 640, 100)]
    arr_svg = "".join(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["muted"]}" stroke-width="2" marker-end="url(#arr)" />'
        for x1, y1, x2, y2 in arrows
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="{t['muted']}" /></marker></defs>
  <rect width="{w}" height="{h}" rx="12" fill="{t['bg']}" stroke="{t['border']}" />
  <text x="24" y="34" fill="{t['text']}" font-size="15" font-weight="700" font-family="ui-sans-serif, system-ui">Real-Time Sentence-Streamed Voice Architecture</text>
  {arr_svg}
  {nodes}
</svg>
"""


async def main() -> None:
    pipe = VoicePipeline()
    streamed_vals: List[float] = []
    buffered_vals: List[float] = []
    stt_vals: List[float] = []
    ttft_vals: List[float] = []

    for i in range(10):
        s = await pipe.run_turn(b"sample-audio", hint_text=f"Turn {i}", stream_sentences=True)
        b = await pipe.run_turn(b"sample-audio", hint_text=f"Turn {i}", stream_sentences=False)
        streamed_vals.append(s.e2e_first_audio_ms)
        buffered_vals.append(b.e2e_first_audio_ms)
        stt_vals.append(s.stt_ms)
        ttft_vals.append(s.llm_ttft_ms)

    s_p50 = round(statistics.median(streamed_vals), 1)
    b_p50 = round(statistics.median(buffered_vals), 1)
    summary = {
        "turns": 10,
        "stt_p50_ms": round(statistics.median(stt_vals), 1),
        "llm_ttft_p50_ms": round(statistics.median(ttft_vals), 1),
        "streamed_first_audio_p50_ms": s_p50,
        "buffered_first_audio_p50_ms": b_p50,
        "speedup_pct": round(((b_p50 - s_p50) / b_p50) * 100.0, 1),
    }

    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "img").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "latency.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "docs" / "BENCHMARKS.md").write_text(
        "# Voice Turn Latency Benchmarks\n\n"
        f"- **STT p50:** {summary['stt_p50_ms']} ms\n"
        f"- **LLM TTFT p50:** {summary['llm_ttft_p50_ms']} ms\n"
        f"- **Sentence-Streamed First Audio p50:** {summary['streamed_first_audio_p50_ms']} ms\n"
        f"- **Buffered First Audio p50:** {summary['buffered_first_audio_p50_ms']} ms\n"
        f"- **Latency Reduction:** {summary['speedup_pct']}%\n",
        encoding="utf-8",
    )

    for theme in ("light", "dark"):
        (ROOT / "docs" / "img" / f"latency-{theme}.svg").write_text(build_latency_svg(summary, theme), encoding="utf-8")
        (ROOT / "docs" / "img" / f"architecture-{theme}.svg").write_text(build_arch_svg(theme), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
