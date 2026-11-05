"""Provider interfaces.

Every vendor sits behind one of these three protocols. The orchestrator never
imports a vendor SDK, so swapping Deepgram for AssemblyAI, or Cartesia for
ElevenLabs, touches exactly one file and no call logic.

All three are streaming interfaces. A request/response STT or TTS API cannot
hit conversational latency targets: you would be waiting for the caller to stop
talking before you start transcribing, and for a whole sentence of audio to
render before you start playing it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class Transcript:
    text: str
    is_final: bool
    confidence: float = 1.0


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@runtime_checkable
class STT(Protocol):
    name: str

    async def start(self) -> None:
        """Open the upstream socket."""

    async def send_audio(self, pcm16k: bytes) -> None:
        """Push caller audio. Never blocks on a network round trip."""

    def transcripts(self) -> AsyncIterator[Transcript]:
        """Interim and final transcripts as they arrive."""

    async def close(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message]) -> AsyncIterator[str]:
        """Yield text deltas. The first delta is the metric that matters."""

    async def close(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    name: str

    def synthesize(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Take a stream of text, yield a stream of PCM16 @ 16 kHz.

        Text in, audio out, both streaming: the synthesizer starts speaking the
        first clause while the LLM is still writing the last one.
        """

    async def close(self) -> None: ...
