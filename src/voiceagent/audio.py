"""Audio format plumbing.

Two sample rates matter here:

* 8 kHz mu-law  -- what a PSTN carrier hands Twilio, and what Twilio hands us.
* 16 kHz PCM16  -- what every STT vendor actually wants.

Everything inside the orchestrator is 16 kHz mono PCM16. Conversion lives here
and nowhere else, so a new transport only has to plug into these helpers.
"""
from __future__ import annotations

import base64
import warnings

# audioop is deprecated from 3.11 and removed in 3.13 (PEP 594). On 3.13+ the
# `audioop-lts` back-port provides the same C module under the same name, so
# the import below is all that changes between versions.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop
    except ImportError as exc:  # pragma: no cover - 3.13+ without the backport
        raise ImportError(
            "audioop is unavailable. On Python 3.13+ install the back-port:\n"
            "  uv pip install audioop-lts"
        ) from exc

PCM_RATE = 16_000
TELEPHONY_RATE = 8_000
SAMPLE_WIDTH = 2  # bytes, PCM16
FRAME_MS = 20
FRAME_BYTES = int(PCM_RATE * SAMPLE_WIDTH * FRAME_MS / 1000)  # 640


def ms_to_bytes(ms: int, rate: int = PCM_RATE) -> int:
    return int(rate * SAMPLE_WIDTH * ms / 1000)


def bytes_to_ms(n: int, rate: int = PCM_RATE) -> float:
    return n * 1000 / (rate * SAMPLE_WIDTH)


class Resampler:
    """Stateful rate conversion.

    `audioop.ratecv` carries interpolation state between calls. Passing None
    every time -- the obvious way to write this -- silently drops a sample at
    each chunk boundary and puts a click into the stream every 20 ms. Keep one
    Resampler per direction per call and feed it in order.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._state = None

    def __call__(self, pcm: bytes) -> bytes:
        if self.src_rate == self.dst_rate:
            return pcm
        out, self._state = audioop.ratecv(
            pcm, SAMPLE_WIDTH, 1, self.src_rate, self.dst_rate, self._state
        )
        return out

    def reset(self) -> None:
        self._state = None


class TelephonyCodec:
    """mu-law 8 kHz <-> PCM16 16 kHz for one call leg."""

    def __init__(self) -> None:
        self._up = Resampler(TELEPHONY_RATE, PCM_RATE)
        self._down = Resampler(PCM_RATE, TELEPHONY_RATE)

    def decode(self, mulaw: bytes) -> bytes:
        """Carrier inbound -> internal format."""
        return self._up(audioop.ulaw2lin(mulaw, SAMPLE_WIDTH))

    def encode(self, pcm16k: bytes) -> bytes:
        """Internal format -> carrier outbound."""
        return audioop.lin2ulaw(self._down(pcm16k), SAMPLE_WIDTH)

    def reset(self) -> None:
        self._up.reset()
        self._down.reset()


def mulaw8k_to_pcm16k(payload: bytes) -> bytes:
    """One-shot convenience for tests and offline files. Not for live streams:
    use TelephonyCodec so resampler state survives across chunks."""
    return TelephonyCodec().decode(payload)


def pcm16k_to_mulaw8k(payload: bytes) -> bytes:
    """One-shot convenience. See the caveat on mulaw8k_to_pcm16k."""
    return TelephonyCodec().encode(payload)


def b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def unb64(payload: str) -> bytes:
    return base64.b64decode(payload)


def rms(frame: bytes) -> float:
    """Root-mean-square amplitude, 0.0 - 1.0."""
    if not frame:
        return 0.0
    return audioop.rms(frame, SAMPLE_WIDTH) / 32768.0


class Framer:
    """Re-chunks a byte stream into fixed-size frames.

    Transports deliver whatever size they like; the VAD wants uniform frames.
    """

    def __init__(self, frame_bytes: int = FRAME_BYTES) -> None:
        self.frame_bytes = frame_bytes
        self._buf = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        self._buf.extend(chunk)
        out: list[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            out.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return out

    def flush(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out
