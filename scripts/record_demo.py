"""Drive a complete call headlessly and record it.

The browser demo needs a human with a microphone. This does the same thing
without one: it synthesizes the caller's side with macOS `say` (a different
voice from the agent's), feeds it into a real CallSession in real time, and
records both sides onto one timeline along with every event the orchestrator
emitted.

That makes the output reproducible and honest -- the latency numbers in the
rendered video are measured from this run, not drawn by hand. The script is
also the load-bearing piece of a regression check: if a change makes turn
taking worse, the recorded timeline says so.

    python scripts/record_demo.py --out demo/

Writes:
    demo/call.wav       the full call, both sides, mixed
    demo/timeline.json  events + per-turn latency traces
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceagent.audio import FRAME_BYTES, FRAME_MS, PCM_RATE, bytes_to_ms  # noqa: E402
from voiceagent.config import Settings  # noqa: E402
from voiceagent.orchestrator import CallSession  # noqa: E402

CALLER_VOICE = os.environ.get("DEMO_CALLER_VOICE", "Daniel")
AGENT_VOICE = os.environ.get("DEMO_AGENT_VOICE", "Samantha")

# The caller's side. `barge_in_after` starts this line while the agent is still
# speaking, which is what exercises cancellation.
SCRIPT: list[dict[str, Any]] = [
    {"text": "Hi, I want to book a cleaning for next week.", "gap": 0.7},
    {"text": "Tuesday morning works, around ten.", "gap": 0.6},
    {"text": "Actually, can you make it eleven instead?", "gap": 0.0,
     "barge_in_after": 0.9},
    {"text": "No, that's all. Thank you.", "gap": 0.6},
]


def say_to_pcm(text: str, voice: str, rate_wpm: int = 180) -> bytes:
    """Render text to 16 kHz mono PCM16 using the macOS `say` binary."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate_wpm), "-o", path,
             "--data-format=LEI16@16000", "--channels=1", "--file-format=WAVE",
             text],
            check=True, capture_output=True,
        )
        with contextlib.closing(wave.open(path, "rb")) as w:
            assert w.getframerate() == PCM_RATE
            return w.readframes(w.getnframes())
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


class Mixer:
    """A growable 16 kHz mono timeline you can write into at an offset."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def _ensure(self, end_byte: int) -> None:
        if end_byte > len(self.buf):
            self.buf.extend(b"\x00" * (end_byte - len(self.buf)))

    def write_at(self, t_seconds: float, pcm: bytes, gain: float = 1.0) -> None:
        start = int(t_seconds * PCM_RATE) * 2
        self._ensure(start + len(pcm))
        if gain == 1.0 and all(self.buf[start + i] == 0 for i in range(0, min(64, len(pcm)))):
            self.buf[start : start + len(pcm)] = pcm
            return
        # Overlap (caller and agent talking at once during a barge-in): add.
        import audioop
        existing = bytes(self.buf[start : start + len(pcm)])
        scaled = audioop.mul(pcm, 2, gain)
        self.buf[start : start + len(pcm)] = audioop.add(existing, scaled, 2)

    def silence_from(self, t_seconds: float) -> None:
        """Drop everything written after this point -- a barge-in clear."""
        start = int(t_seconds * PCM_RATE) * 2
        if start < len(self.buf):
            del self.buf[start:]

    def save(self, path: Path) -> float:
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(wave.open(str(path), "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(PCM_RATE)
            w.writeframes(bytes(self.buf))
        return len(self.buf) / 2 / PCM_RATE


@dataclass
class RecordingTransport:
    """Captures what the agent said, and when it said it."""

    mixer: Mixer
    t0: float
    events: list[dict[str, Any]] = field(default_factory=list)
    written_until: float = 0.0

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def send_audio(self, pcm16k: bytes) -> None:
        # Playout is paced to real time, so "sent at" is "heard at" to within
        # the pacing window. Append-at-the-write-head keeps chunks gapless even
        # when the event loop jitters.
        at = max(self.now(), self.written_until)
        self.mixer.write_at(at, pcm16k, gain=0.85)
        self.written_until = at + bytes_to_ms(len(pcm16k)) / 1000

    async def clear_audio(self) -> None:
        # Barge-in: the caller never hears what was queued but unplayed.
        cut = self.now()
        self.mixer.silence_from(cut)
        self.written_until = cut
        self.events.append({"t": round(cut, 3), "type": "transport_clear"})

    async def send_event(self, event: dict[str, Any]) -> None:
        self.events.append({"t": round(self.now(), 3), **event})


async def run_demo(out_dir: Path, settings: Settings) -> dict[str, Any]:
    mixer = Mixer()
    t0 = time.monotonic()
    transport = RecordingTransport(mixer=mixer, t0=t0)
    session = CallSession(transport, settings, call_id="demo")

    print(f"providers: {session.providers}")
    print("rendering caller audio with `say`...")
    caller_audio = [say_to_pcm(line["text"], CALLER_VOICE) for line in SCRIPT]

    await session.start()  # plays the greeting

    for line, pcm in zip(SCRIPT, caller_audio):
        barge = line.get("barge_in_after")
        if barge is not None:
            # Interrupt: wait only a moment into the agent's reply, then talk.
            deadline = time.monotonic() + 6.0
            while session.state.value != "speaking" and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            await asyncio.sleep(barge)
        else:
            # Normal turn: let the agent finish, then leave a natural gap.
            deadline = time.monotonic() + 15.0
            while session.state.value != "listening" and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            await asyncio.sleep(line.get("gap", 0.5))

        start_at = transport.now()
        print(f"  caller [{start_at:5.1f}s]: {line['text']}")
        mixer.write_at(start_at, pcm)

        # Feed it in at wall-clock speed, exactly as a live leg would arrive.
        for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
            await session.on_audio(pcm[i : i + FRAME_BYTES])
            await asyncio.sleep(FRAME_MS / 1000)

        # Trailing silence so the VAD can call end-of-turn.
        for _ in range(int(settings.end_of_turn_silence_ms / FRAME_MS) + 6):
            await session.on_audio(b"\x00" * FRAME_BYTES)
            await asyncio.sleep(FRAME_MS / 1000)

    # Let the final reply play out.
    deadline = time.monotonic() + 15.0
    while session.state.value != "listening" and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.6)

    await session.close()

    duration = mixer.save(out_dir / "call.wav")
    payload = {
        "call": session.metrics.to_dict(),
        "providers": session.providers,
        "events": transport.events,
        "duration_s": round(duration, 2),
        "voices": {"caller": CALLER_VOICE, "agent": AGENT_VOICE},
    }
    (out_dir / "timeline.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("demo"))
    ap.add_argument("--stt", default=os.environ.get("DEMO_STT", "mock"))
    ap.add_argument("--llm", default=os.environ.get("DEMO_LLM", "mock"))
    ap.add_argument("--tts", default=os.environ.get("DEMO_TTS", "espeak"))
    args = ap.parse_args()

    settings = Settings(
        stt_provider=args.stt, llm_provider=args.llm, tts_provider=args.tts,
        say_voice=AGENT_VOICE,
        end_of_turn_silence_ms=560,
        barge_in_speech_ms=200,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(run_demo(args.out, settings))

    s = payload["call"]["summary"]
    print("\n--- call summary -------------------------------------------")
    print(f"  turns        {s['turns']}   barge-ins {s['barge_ins']}")
    print(f"  latency p50  {s['response_latency_ms']['p50']} ms")
    print(f"  latency p95  {s['response_latency_ms']['p95']} ms")
    print(f"  duration     {payload['duration_s']} s")
    print(f"  wrote        {args.out}/call.wav, {args.out}/timeline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
