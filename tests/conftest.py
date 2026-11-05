import math
import struct

import pytest

from voiceagent.audio import FRAME_BYTES, PCM_RATE


def tone(ms: int, amplitude: float = 0.35, freq: float = 220.0) -> bytes:
    """Loud-enough PCM16 that the VAD will call speech."""
    n = int(PCM_RATE * ms / 1000)
    return b"".join(
        struct.pack("<h", int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / PCM_RATE)))
        for i in range(n)
    )


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(PCM_RATE * ms / 1000)


@pytest.fixture
def frame_bytes() -> int:
    return FRAME_BYTES
