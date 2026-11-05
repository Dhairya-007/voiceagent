"""Free streaming TTS via espeak-ng (GPL, `brew install espeak-ng`).

Measured on an M-series laptop, time to first audio byte:

    espeak-ng   ~15 ms     formant synthesis, robotic
    macOS say  ~860 ms     natural voice, but ~860 ms of that is the process
                           loading the voice before it emits anything

That is the tradeoff this adapter exists to make visible. espeak-ng does not
sound good, but it starts speaking before a neural voice has finished loading,
and on a phone call the gap before the agent answers is far more noticeable
than the timbre once it does. A neural voice held warm in-process (see
tts_piper) is the option that gets both; a neural voice spawned per utterance
gets neither.

espeak-ng writes a WAV stream to stdout whose header declares a placeholder
length, so the header is skipped and the rest is read as raw PCM.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

from ..audio import PCM_RATE, Resampler

WAV_HEADER_BYTES = 44
ESPEAK_RATE = 22_050


class EspeakTTS:
    name = "espeak-ng"

    def __init__(
        self,
        binary: str = "espeak-ng",
        voice: str = "en-us",
        words_per_minute: int = 170,
        pitch: int = 50,
    ) -> None:
        self.binary = binary
        self.voice = voice
        self.words_per_minute = words_per_minute
        self.pitch = pitch
        self._proc: asyncio.subprocess.Process | None = None
        self._cancelled = False

    async def _render(self, text: str) -> AsyncIterator[bytes]:
        self._proc = await asyncio.create_subprocess_exec(
            self.binary,
            "-v", self.voice,
            "-s", str(self.words_per_minute),
            "-p", str(self.pitch),
            "--stdout", text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        proc = self._proc
        assert proc.stdout is not None

        # One resampler per utterance, stateful across reads so the chunk
        # seams do not click.
        down = Resampler(ESPEAK_RATE, PCM_RATE)
        try:
            with contextlib.suppress(asyncio.IncompleteReadError):
                await proc.stdout.readexactly(WAV_HEADER_BYTES)
            while not self._cancelled:
                raw = await proc.stdout.read(4096)
                if not raw:
                    break
                out = down(raw)
                if out:
                    yield out
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
            self._proc = None

    async def synthesize(
        self, text_chunks: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        self._cancelled = False
        async for chunk in text_chunks:
            if self._cancelled:
                return
            if not chunk.strip():
                continue
            async for pcm in self._render(chunk):
                if self._cancelled:
                    return
                yield pcm

    async def cancel(self) -> None:
        """Barge-in: stop the synthesizer now."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def close(self) -> None:
        await self.cancel()
