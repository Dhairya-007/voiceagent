"""Energy-gate voice activity detection and turn-end detection.

An RMS gate is deliberately simple. It is enough to demo barge-in and
end-of-turn on clean audio, and it keeps the repo dependency-free. In
production you would swap `SpeechGate.is_speech` for webrtcvad or Silero and
keep the hysteresis logic below exactly as it is -- that logic, not the
classifier, is what makes turn-taking feel natural.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audio import FRAME_MS, rms


@dataclass
class SpeechGate:
    """Frame-level speech/silence decision with hysteresis.

    Hysteresis matters: a single-frame flip would chop words in half and make
    the agent interrupt itself on a breath.
    """

    open_threshold: float = 0.018  # RMS to call it speech
    close_threshold: float = 0.010  # lower bar to keep calling it speech
    _open: bool = field(default=False, init=False)

    def is_speech(self, frame: bytes) -> bool:
        level = rms(frame)
        if self._open:
            self._open = level > self.close_threshold
        else:
            self._open = level > self.open_threshold
        return self._open

    def reset(self) -> None:
        self._open = False


@dataclass
class TurnDetector:
    """Tracks when the caller started and stopped talking.

    `feed` returns one of: None, "speech_start", "speech_end".
    """

    end_of_turn_silence_ms: int = 640
    min_speech_ms: int = 160
    gate: SpeechGate = field(default_factory=SpeechGate)

    _speaking: bool = field(default=False, init=False)
    _speech_ms: float = field(default=0.0, init=False)
    _silence_ms: float = field(default=0.0, init=False)

    def feed(self, frame: bytes, frame_ms: float = FRAME_MS) -> str | None:
        speech = self.gate.is_speech(frame)

        if speech:
            self._silence_ms = 0.0
            self._speech_ms += frame_ms
            if not self._speaking and self._speech_ms >= self.min_speech_ms:
                self._speaking = True
                return "speech_start"
            return None

        if self._speaking:
            self._silence_ms += frame_ms
            if self._silence_ms >= self.end_of_turn_silence_ms:
                self.reset()
                return "speech_end"
        else:
            # Decay so scattered noise frames never add up to a turn.
            self._speech_ms = max(0.0, self._speech_ms - frame_ms)
        return None

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self.gate.reset()
