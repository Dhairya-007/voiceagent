"""Free neural TTS via Piper (MIT model weights, ONNX runtime).

The point of this adapter is that the model is loaded **once** and held in
process. Piper also ships a CLI, and using it would be the obvious thing to
do -- but spawning a process per utterance pays the model load every time,
which is exactly the ~860 ms trap the macOS `say` adapter documents. Loading
once turns that into a one-off startup cost.

Setup (free):
    uv pip install piper-tts
    python -m piper.download_voices en_US-amy-medium --data-dir models

Known issue: on some macOS wheels the bundled espeak-ng bridge has the build
machine's data path compiled in and aborts at import with

    Error processing file '/Users/runner/work/piper1-gpl/.../phontab'

If that happens, use the espeak-ng adapter (fast, robotic) or `say` (natural,
slow to start) instead; `build_tts` falls back automatically.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import AsyncIterator

from ..audio import PCM_RATE, Resampler

_SENTINEL = object()


class PiperTTS:
    name = "piper"

    def __init__(
        self, model_path: str, length_scale: float = 1.0, speaker_id: int = 0
    ) -> None:
        self.model_path = model_path
        self.length_scale = length_scale
        self.speaker_id = speaker_id
        self._voice = None
        self._rate = 22_050
        self._cancelled = False

    def _load(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice  # imported lazily: see the module docstring

        self._voice = PiperVoice.load(self.model_path)
        self._rate = self._voice.config.sample_rate

    async def start(self) -> None:
        await asyncio.to_thread(self._load)

    async def _render(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.to_thread(self._load)

        # Piper's synthesize() is a blocking generator. Run it on a worker
        # thread and hand chunks back through a queue so the event loop -- and
        # therefore the inbound audio pump and barge-in detection -- keeps
        # running while the model works.
        q: queue.Queue = queue.Queue(maxsize=8)

        def produce() -> None:
            try:
                for chunk in self._voice.synthesize(
                    text, length_scale=self.length_scale,
                    speaker_id=self.speaker_id or None,
                ):
                    if self._cancelled:
                        break
                    q.put(chunk.audio_int16_bytes)
            except Exception as exc:  # surfaced on the consumer side
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        threading.Thread(target=produce, daemon=True).start()

        down = Resampler(self._rate, PCM_RATE)
        while True:
            item = await asyncio.to_thread(q.get)
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            if self._cancelled:
                return
            out = down(item)
            if out:
                yield out

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
                yield pcm

    async def cancel(self) -> None:
        self._cancelled = True

    async def close(self) -> None:
        await self.cancel()
