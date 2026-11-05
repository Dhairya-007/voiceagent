"""Free TTS using the macOS built-in `say` binary.

Zero install, zero cost, works offline, and the voices are decent. Good enough
for a demo and for local development; on Linux use the Piper adapter instead.

`say` has no streaming mode, so each clause is rendered to a temp WAV and read
back. Clause-level chunking (see voiceagent.textchunk) is what keeps
time-to-first-audio low despite that.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import wave
from typing import AsyncIterator

from ..audio import PCM_RATE


class SayTTS:
    name = "macos-say"

    def __init__(self, voice: str = "Samantha", rate_wpm: int = 185) -> None:
        self.voice = voice
        self.rate_wpm = rate_wpm
        self._proc: asyncio.subprocess.Process | None = None
        self._cancelled = False

    async def _render(self, text: str) -> bytes:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "say", "-v", self.voice, "-r", str(self.rate_wpm),
                "-o", path,
                "--data-format=LEI16@16000", "--channels=1", "--file-format=WAVE",
                text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await self._proc.communicate()
            if self._proc.returncode != 0:
                raise RuntimeError(f"say failed: {err.decode(errors='replace')}")
            with contextlib.closing(wave.open(path, "rb")) as w:
                assert w.getframerate() == PCM_RATE, w.getframerate()
                return w.readframes(w.getnframes())
        finally:
            self._proc = None
            with contextlib.suppress(OSError):
                os.unlink(path)

    async def synthesize(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        self._cancelled = False
        async for chunk in text_chunks:
            if self._cancelled:
                return
            if not chunk.strip():
                continue
            pcm = await self._render(chunk)
            # Hand the playout loop ~100 ms at a time so a barge-in can cut in
            # mid-clause instead of waiting for the whole clause to drain.
            step = int(PCM_RATE * 2 * 0.1)
            for i in range(0, len(pcm), step):
                if self._cancelled:
                    return
                yield pcm[i : i + step]
                await asyncio.sleep(0)

    async def cancel(self) -> None:
        """Kill an in-flight render on barge-in."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def close(self) -> None:
        await self.cancel()
