"""Per-turn latency tracing.

The number a caller feels is one thing: how long after they stop talking does
the agent start talking. Everything else is a component of that. So each turn
records absolute marks and the derived spans fall out of them:

    speech_end      caller stopped talking (VAD decided the turn was over)
    stt_final       final transcript in hand
    llm_first_token first text delta from the model
    tts_first_byte  first audio bytes from the synthesizer
    playout_start   first byte actually written to the caller's stream
    playout_end     last byte written

    response_latency_ms = playout_start - speech_end   <-- the one that matters

Keeping marks instead of pre-computed durations means a new question ("how much
of the wait was the model?") is answered from data already recorded rather than
by shipping new instrumentation.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

MARKS = (
    "speech_start",
    "speech_end",
    "stt_partial",
    "stt_final",
    "llm_first_token",
    "llm_done",
    "tts_first_byte",
    "playout_start",
    "playout_end",
    "barge_in",
)

# Budget for a call that feels like a conversation rather than a form. Humans
# start noticing a gap at roughly 500 ms and start talking over you past ~1 s.
TARGET_RESPONSE_MS = 800.0


@dataclass
class TurnTrace:
    turn: int
    # "caller" turns are responses to something the caller said and carry a
    # response latency. "greeting" turns are agent-initiated, so there is
    # nothing to measure a response against and they are excluded from the
    # call's latency stats.
    kind: str = "caller"
    marks: dict[str, float] = field(default_factory=dict)
    transcript: str = ""
    reply: str = ""
    barged_in: bool = False
    providers: dict[str, str] = field(default_factory=dict)
    audio_ms: float = 0.0

    def mark(self, name: str, at: float | None = None) -> float:
        """Record a mark. First write wins -- `tts_first_byte` means the first."""
        t = at if at is not None else time.monotonic() * 1000
        self.marks.setdefault(name, t)
        return t

    # Order the critical path must occur in. A violation means two turns got
    # mixed into one trace, or a mark was recorded on the wrong turn -- both
    # produce nonsense latencies, so they are surfaced rather than averaged in.
    CRITICAL_PATH = (
        "speech_end", "stt_final", "llm_first_token",
        "tts_first_byte", "playout_start",
    )

    def anomalies(self) -> list[str]:
        present = [m for m in self.CRITICAL_PATH if m in self.marks]
        out = []
        for a, b in zip(present, present[1:]):
            if self.marks[b] < self.marks[a]:
                out.append(f"{b} before {a}")
        return out

    def _span(self, a: str, b: str) -> float | None:
        if a not in self.marks or b not in self.marks:
            return None
        delta = self.marks[b] - self.marks[a]
        if delta < 0:
            # Never report a negative duration: it is always a bug, and
            # reporting it as a number lets it average away unnoticed.
            return None
        return round(delta, 1)

    @property
    def response_latency_ms(self) -> float | None:
        if self.kind != "caller":
            return None
        return self._span("speech_end", "playout_start")

    @property
    def open_for_speech(self) -> bool:
        """True while this trace is mid-caller-turn.

        A greeting has no speech_start, and a finished turn has speech_end, so
        neither will absorb the next utterance's marks.
        """
        return "speech_start" in self.marks and "speech_end" not in self.marks

    def breakdown(self) -> dict[str, float | None]:
        """Where the response latency actually went."""
        return {
            "stt_finalize": self._span("speech_end", "stt_final"),
            "llm_ttft": self._span("stt_final", "llm_first_token"),
            "tts_ttfb": self._span("llm_first_token", "tts_first_byte"),
            "playout_dispatch": self._span("tts_first_byte", "playout_start"),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Marks are monotonic-clock values; rebase on the turn's own start so a
        # trace is readable on its own and safe to serialise.
        if self.marks:
            t0 = min(self.marks.values())
            d["marks"] = {k: round(v - t0, 1) for k, v in self.marks.items()}
            d["t0"] = t0
        d["response_latency_ms"] = self.response_latency_ms
        d["breakdown"] = self.breakdown()
        d["anomalies"] = self.anomalies()
        d["within_budget"] = (
            None
            if self.response_latency_ms is None
            else self.response_latency_ms <= TARGET_RESPONSE_MS
        )
        return d


class CallMetrics:
    """All traces for one call, plus the roll-up."""

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.started_at = time.time()
        self.turns: list[TurnTrace] = []

    def new_turn(
        self, providers: dict[str, str] | None = None, kind: str = "caller"
    ) -> TurnTrace:
        t = TurnTrace(
            turn=len(self.turns) + 1, kind=kind, providers=providers or {}
        )
        self.turns.append(t)
        return t

    @property
    def current(self) -> TurnTrace | None:
        return self.turns[-1] if self.turns else None

    def summary(self) -> dict[str, Any]:
        lat = [t.response_latency_ms for t in self.turns if t.response_latency_ms]
        lat_sorted = sorted(lat)

        def pct(p: float) -> float | None:
            if not lat_sorted:
                return None
            # Nearest-rank percentile: with 4 turns in a demo, interpolation
            # would invent precision that is not there.
            k = max(0, min(len(lat_sorted) - 1, round(p / 100 * len(lat_sorted)) - 1))
            return round(lat_sorted[k], 1)

        anomalies = [a for t in self.turns for a in t.anomalies()]
        return {
            "call_id": self.call_id,
            "turns": len(self.turns),
            "caller_turns": sum(1 for t in self.turns if t.kind == "caller"),
            "anomalies": anomalies,
            "barge_ins": sum(1 for t in self.turns if t.barged_in),
            "response_latency_ms": {
                "p50": pct(50),
                "p95": pct(95),
                "max": round(max(lat), 1) if lat else None,
                "mean": round(sum(lat) / len(lat), 1) if lat else None,
            },
            "target_ms": TARGET_RESPONSE_MS,
            "within_budget": (
                None if not lat else sum(1 for v in lat if v <= TARGET_RESPONSE_MS)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "turns": [t.to_dict() for t in self.turns],
        }
