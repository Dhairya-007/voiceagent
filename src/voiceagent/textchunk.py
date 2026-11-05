"""Split a streaming text feed into speakable chunks.

This is the single highest-leverage piece of latency work in the pipeline.

A naive agent waits for the LLM to finish, then synthesizes the whole reply:
time-to-first-audio = full LLM time + full TTS time. Chunking at clause
boundaries instead means TTS starts on "Sure, I can book that." while the model
is still writing "What day works for you?", so the two overlap and
time-to-first-audio collapses to first-clause LLM time + first-clause TTS time.

The tension: chunk too eagerly and prosody suffers (the synthesizer cannot
place sentence stress without seeing the clause), chunk too late and you give
the latency back. Breaking on terminal punctuation, and on commas only once a
minimum length has built up, is the compromise that sounds right.
"""
from __future__ import annotations

from typing import AsyncIterator

HARD_BREAKS = ".!?\n"
SOFT_BREAKS = ",;:-"


class ClauseChunker:
    def __init__(self, min_chars: int = 28, max_chars: int = 180) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        """Feed an LLM delta, get back zero or more ready-to-speak chunks."""
        self._buf += delta
        out: list[str] = []
        while True:
            idx = self._find_break()
            if idx is None:
                break
            chunk, self._buf = self._buf[: idx + 1], self._buf[idx + 1 :]
            chunk = chunk.strip()
            if chunk:
                out.append(chunk)
        return out

    def _find_break(self) -> int | None:
        # Terminal punctuation always breaks -- but not a decimal point or an
        # abbreviation, so require the next char to be space or end of buffer.
        for i, ch in enumerate(self._buf):
            if ch in HARD_BREAKS:
                nxt = self._buf[i + 1 : i + 2]
                if nxt in ("", " ", "\n"):
                    return i
        # Soft break only once there is enough text to be worth speaking.
        if len(self._buf) >= self.min_chars:
            for i in range(len(self._buf) - 1, -1, -1):
                if self._buf[i] in SOFT_BREAKS:
                    return i
        # Runaway sentence: break on the last space rather than grow forever.
        if len(self._buf) >= self.max_chars:
            sp = self._buf.rfind(" ", 0, self.max_chars)
            if sp > 0:
                return sp
        return None

    def flush(self) -> str:
        out, self._buf = self._buf.strip(), ""
        return out


async def clauses(deltas: AsyncIterator[str], **kw) -> AsyncIterator[str]:
    """Wrap a delta stream into a clause stream."""
    chunker = ClauseChunker(**kw)
    async for delta in deltas:
        for chunk in chunker.push(delta):
            yield chunk
    tail = chunker.flush()
    if tail:
        yield tail
