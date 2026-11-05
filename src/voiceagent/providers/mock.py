"""Offline stand-ins for all three providers.

These exist so the repo runs, and the tests pass, with no API keys and no
network. They imitate the *timing* of real vendors (first-token delay, chunked
delivery) so latency traces from a mock run look structurally like a real one.
"""
from __future__ import annotations

import asyncio
import math
import random
import re
import struct
from typing import AsyncIterator

from ..audio import PCM_RATE
from .base import Message, Transcript


class MockSTT:
    """Turns caller audio into plausible text without a model.

    It cannot actually transcribe, so it reports how long the caller spoke and
    emits a canned utterance from a rotating script. Interim results arrive
    while audio is still flowing, exactly as a streaming vendor would send them.
    """

    name = "mock-stt"

    SCRIPT = [
        "hi, i want to book a cleaning for next week",
        "tuesday morning works, around ten",
        "actually can you make it eleven instead",
        "no that's all, thank you",
    ]

    def __init__(self, first_partial_ms: int = 120, final_ms: int = 90) -> None:
        self._q: asyncio.Queue[Transcript | None] = asyncio.Queue()
        self._audio_ms = 0.0
        self._turn = 0
        self._emitted_partial = False
        self.first_partial_ms = first_partial_ms
        self.final_ms = final_ms

    async def start(self) -> None:
        self._audio_ms = 0.0

    async def send_audio(self, pcm16k: bytes) -> None:
        self._audio_ms += len(pcm16k) * 1000 / (PCM_RATE * 2)
        if not self._emitted_partial and self._audio_ms > 300:
            self._emitted_partial = True
            words = self._utterance().split()
            await asyncio.sleep(self.first_partial_ms / 1000)
            await self._q.put(
                Transcript(" ".join(words[: max(1, len(words) // 2)]), is_final=False)
            )

    async def finalize(self) -> None:
        """Called by the orchestrator on end-of-turn."""
        await asyncio.sleep(self.final_ms / 1000)
        await self._q.put(Transcript(self._utterance(), is_final=True, confidence=0.94))
        self._turn += 1
        self._emitted_partial = False
        self._audio_ms = 0.0

    def _utterance(self) -> str:
        return self.SCRIPT[self._turn % len(self.SCRIPT)]

    async def transcripts(self) -> AsyncIterator[Transcript]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        await self._q.put(None)


class MockLLM:
    """Keyword-routed canned replies, streamed word by word."""

    name = "mock-llm"

    RULES: list[tuple[str, str]] = [
        (r"\b(book|appointment|cleaning|checkup)\b",
         "Sure, I can book that. What day works for you?"),
        (r"\b(tuesday|monday|wednesday|thursday|friday|morning|ten|eleven)\b",
         "Got it. I have that slot open. Should I confirm it?"),
        (r"\b(change|instead|move|resched)\b",
         "No problem, I have moved it. Anything else?"),
        (r"\b(thanks|thank you|that's all|bye)\b",
         "Happy to help. See you then. Goodbye."),
        (r"\b(human|agent|person|manager)\b",
         "Of course, let me transfer you to a colleague."),
    ]
    FALLBACK = "Sorry, I missed that. Could you say it again?"

    def __init__(self, first_token_ms: int = 260, token_gap_ms: int = 28) -> None:
        self.first_token_ms = first_token_ms
        self.token_gap_ms = token_gap_ms

    def _reply(self, messages: list[Message]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        for pattern, reply in self.RULES:
            if re.search(pattern, last, re.I):
                return reply
        return self.FALLBACK

    async def complete(
        self, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        reply = self._reply(messages)
        await asyncio.sleep(self.first_token_ms / 1000)
        for i, word in enumerate(reply.split()):
            yield word if i == 0 else " " + word
            await asyncio.sleep(self.token_gap_ms / 1000)

    async def close(self) -> None:
        return None


class MockTTS:
    """Renders text as a hum whose pitch tracks the characters.

    Not speech -- but it is real PCM of a realistic duration, so playout
    pacing, barge-in cancellation and the latency waterfall are all exercised
    for real. Swap in Cartesia and nothing else changes.
    """

    name = "mock-tts"

    def __init__(self, first_byte_ms: int = 140, chars_per_second: float = 14.0) -> None:
        self.first_byte_ms = first_byte_ms
        self.chars_per_second = chars_per_second

    async def synthesize(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        first = True
        phase = 0.0
        async for chunk in text_chunks:
            if not chunk.strip():
                continue
            if first:
                await asyncio.sleep(self.first_byte_ms / 1000)
                first = False
            duration_s = len(chunk) / self.chars_per_second
            n_samples = max(1, int(PCM_RATE * duration_s))
            base = 150 + (sum(map(ord, chunk)) % 90)
            out = bytearray()
            for i in range(n_samples):
                # Light vibrato so it sounds voice-like rather than like a test tone.
                f = base + 12 * math.sin(2 * math.pi * 4.5 * i / PCM_RATE)
                phase += 2 * math.pi * f / PCM_RATE
                env = min(1.0, i / 400) * min(1.0, (n_samples - i) / 400)
                out += struct.pack("<h", int(7000 * env * math.sin(phase)))
                if len(out) >= 3200:  # ~100 ms per yielded chunk
                    yield bytes(out)
                    out.clear()
                    await asyncio.sleep(0)
            if out:
                yield bytes(out)

    async def close(self) -> None:
        return None
