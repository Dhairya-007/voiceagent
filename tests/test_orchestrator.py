"""End-to-end turn behaviour against the mock providers.

These run the real orchestrator, real VAD, real chunker and real metrics -- only
the vendors and the transport are fakes. That is the seam worth testing: the
bugs in a voice agent live in turn-taking and cancellation, not in the vendor
SDKs.
"""
from __future__ import annotations

import asyncio

import pytest

from voiceagent.audio import FRAME_BYTES
from voiceagent.config import Settings
from voiceagent.orchestrator import CallSession, State
from conftest import silence, tone


class FakeTransport:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.events: list[dict] = []
        self.clears = 0

    async def send_audio(self, pcm16k: bytes) -> None:
        self.audio.extend(pcm16k)

    async def clear_audio(self) -> None:
        self.clears += 1
        self.audio.clear()

    async def send_event(self, event: dict) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e["type"] for e in self.events]


def settings(**kw) -> Settings:
    base = dict(
        stt_provider="mock", llm_provider="mock", tts_provider="mock",
        end_of_turn_silence_ms=200, barge_in_speech_ms=120,
    )
    base.update(kw)
    return Settings(**base)


async def feed(session: CallSession, pcm: bytes) -> None:
    """Push audio one frame at a time, yielding so provider tasks can run."""
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        await session.on_audio(pcm[i : i + FRAME_BYTES])
        await asyncio.sleep(0)


async def settle(timeout: float = 6.0, **_) -> None:
    await asyncio.sleep(0)


@pytest.fixture
async def session():
    t = FakeTransport()
    s = CallSession(t, settings(), greeting=None)
    await s.start()
    yield s, t
    await s.close()


async def wait_for(
    transport: FakeTransport, event: str, timeout: float = 6.0,
    turn: int | None = None,
):
    """Wait until `event` shows up, instead of sleeping a guessed duration.

    `turn` matters once a greeting is in play: the greeting emits its own
    turn_complete before the caller has said anything, so a test that just
    waits for "turn_complete" would race ahead of the turn it cares about.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for e in transport.events:
            if e["type"] == event and (turn is None or e.get("turn") == turn):
                return e
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"{event!r}{f' for turn {turn}' if turn else ''} never arrived; "
        f"saw {transport.types()}"
    )


async def test_starts_in_listening(session):
    s, t = session
    assert s.state is State.LISTENING
    assert "call_started" in t.types()


async def test_full_turn_produces_audio_and_a_latency_trace(session):
    s, t = session
    await feed(s, tone(500) + silence(400))
    await wait_for(t, "turn_complete")

    assert t.audio, "the agent produced no audio"
    trace = s.metrics.turns[0]
    assert trace.transcript, "no final transcript was recorded"
    assert trace.reply, "no reply text was recorded"

    # Every mark on the critical path must be present, in order.
    for mark in ("speech_start", "speech_end", "stt_final",
                 "llm_first_token", "tts_first_byte", "playout_start"):
        assert mark in trace.marks, f"missing mark {mark}"
    ordered = [trace.marks[m] for m in
               ("speech_end", "stt_final", "llm_first_token",
                "tts_first_byte", "playout_start")]
    assert ordered == sorted(ordered), f"marks out of order: {ordered}"

    assert trace.response_latency_ms > 0
    # The breakdown must account for the whole wait, with no unexplained gap.
    assert sum(v for v in trace.breakdown().values() if v) == pytest.approx(
        trace.response_latency_ms, abs=1.0
    )


async def test_returns_to_listening_after_speaking(session):
    s, t = session
    await feed(s, tone(500) + silence(400))
    await wait_for(t, "turn_complete")
    assert s.state is State.LISTENING


async def test_barge_in_cancels_playback_and_clears_the_transport(session):
    s, t = session
    await feed(s, tone(500) + silence(400))
    await wait_for(t, "playout_start")
    assert s.state is State.SPEAKING

    # Interrupt, loudly enough to clear the raised during-playout threshold.
    await feed(s, tone(400, amplitude=0.8))
    await wait_for(t, "barge_in")

    assert t.clears >= 1, "transport was never told to drop queued audio"
    assert s.metrics.turns[0].barged_in is True
    assert s.state in (State.LISTENING, State.THINKING)


async def test_quiet_audio_during_playout_does_not_barge_in(session):
    """Echo of the agent's own voice must not interrupt the agent.

    This is the bug that makes a bot talk over itself on a speakerphone: the
    caller's mic picks up the agent, the VAD calls it speech, and the agent
    cancels its own sentence.
    """
    s, t = session
    await feed(s, tone(500) + silence(400))
    await wait_for(t, "playout_start")

    # Above the listening threshold, below the during-playout threshold.
    await feed(s, tone(500, amplitude=0.03))
    assert "barge_in" not in t.types()
    assert s.metrics.turns[0].barged_in is False


async def test_silence_alone_never_starts_a_turn(session):
    s, t = session
    await feed(s, silence(1500))
    assert "speech_start" not in t.types()
    assert s.metrics.turns == []


async def test_history_is_trimmed_to_the_configured_window():
    t = FakeTransport()
    s = CallSession(t, settings(max_history_turns=2), greeting=None)
    await s.start()
    try:
        for _ in range(4):
            await feed(s, tone(400) + silence(300))
            await asyncio.sleep(0.6)
        assert len(s.history) <= 4, f"history grew to {len(s.history)}"
    finally:
        await s.close()


async def test_greeting_warms_the_pipeline_without_a_latency_number():
    t = FakeTransport()
    s = CallSession(t, settings(), greeting="Hello, clinic here.")
    await s.start()
    try:
        assert t.audio, "greeting produced no audio"
        greeting = s.metrics.turns[0]
        # Nothing was being responded to, so there is no response latency.
        assert greeting.response_latency_ms is None
        assert s.state is State.LISTENING
    finally:
        await s.close()


async def test_close_is_idempotent_and_reports_a_summary(session):
    s, t = session
    await feed(s, tone(400) + silence(300))
    await wait_for(t, "turn_complete")
    await s.close()
    await s.close()
    assert t.types().count("call_ended") == 1
    assert s.metrics.summary()["turns"] >= 1


async def test_greeting_does_not_absorb_the_first_caller_turn():
    """Regression: the first caller turn reported a negative latency.

    The greeting created a trace and never set speech_end on it, so the first
    caller utterance wrote speech_start/speech_end onto a trace that already
    had playout_start from the greeting. speech_end then landed *after*
    playout_start and the subtraction went negative.
    """
    t = FakeTransport()
    s = CallSession(t, settings(), greeting="Hello, clinic here.")
    await s.start()
    try:
        await feed(s, tone(500) + silence(400))
        await wait_for(t, "turn_complete", timeout=8.0, turn=2)

        greeting, first = s.metrics.turns[0], s.metrics.turns[1]
        assert greeting.kind == "greeting"
        assert "speech_end" not in greeting.marks
        assert first.kind == "caller"
        assert first.response_latency_ms > 0
        assert not first.anomalies()
    finally:
        await s.close()


async def test_no_turn_reports_a_negative_or_out_of_order_latency(session):
    s, t = session
    for _ in range(2):
        await feed(s, tone(400) + silence(400))
        await asyncio.sleep(0.8)
    for trace in s.metrics.turns:
        assert not trace.anomalies(), f"turn {trace.turn}: {trace.anomalies()}"
        for name, value in trace.breakdown().items():
            assert value is None or value >= 0, f"{name} was {value}"
    assert s.metrics.summary()["anomalies"] == []
