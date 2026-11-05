"""Provider factory.

The orchestrator asks for "an STT" and gets one. Which vendor that is comes
from config, and a provider that cannot start degrades to the mock rather than
dropping the call -- a caller hearing a slightly dumber bot beats a caller
hearing dead air.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache

from ..config import Settings
from .base import LLM, STT, TTS, Message, Transcript
from .mock import MockLLM, MockSTT, MockTTS

log = logging.getLogger(__name__)

__all__ = [
    "LLM", "STT", "TTS", "Message", "Transcript",
    "MockLLM", "MockSTT", "MockTTS",
    "build_stt", "build_llm", "build_tts", "describe",
]


def build_stt(s: Settings):
    if s.stt_provider == "whisper":
        try:
            from .stt_whisper import WhisperSTT

            return WhisperSTT(
                model_size=s.whisper_model,
                device=s.whisper_device,
                compute_type=s.whisper_compute_type,
            )
        except Exception as exc:
            if not s.fallback_to_mock:
                raise
            log.warning("STT 'whisper' unavailable (%s); using mock", exc)
    return MockSTT()


def build_llm(s: Settings):
    if s.llm_provider == "openai_compat":
        try:
            from .llm_openai_compat import OpenAICompatLLM

            return OpenAICompatLLM(
                base_url=s.llm_base_url,
                model=s.llm_model,
                api_key=s.llm_api_key,
                max_tokens=s.llm_max_tokens,
            )
        except Exception as exc:
            if not s.fallback_to_mock:
                raise
            log.warning("LLM 'openai_compat' unavailable (%s); using mock", exc)
    return MockLLM()


def build_tts(s: Settings):
    if s.tts_provider == "say":
        try:
            from .tts_say import SayTTS

            return SayTTS(voice=s.say_voice, rate_wpm=s.say_rate_wpm)
        except Exception as exc:
            if not s.fallback_to_mock:
                raise
            log.warning("TTS 'say' unavailable (%s); using mock", exc)
    elif s.tts_provider == "espeak":
        try:
            import shutil

            if not shutil.which(s.espeak_binary):
                raise RuntimeError(
                    f"{s.espeak_binary} not on PATH (brew install espeak-ng)"
                )
            from .tts_espeak import EspeakTTS

            return EspeakTTS(
                binary=s.espeak_binary, voice=s.espeak_voice,
                words_per_minute=s.espeak_wpm,
            )
        except Exception as exc:
            if not s.fallback_to_mock:
                raise
            log.warning("TTS 'espeak' unavailable (%s); using mock", exc)
    elif s.tts_provider == "piper":
        try:
            if not s.piper_model_path:
                raise RuntimeError("PIPER_MODEL_PATH is not set")
            if not os.path.isfile(s.piper_model_path):
                raise RuntimeError(f"model not found: {s.piper_model_path}")
            if not _piper_works(s.piper_model_path):
                raise RuntimeError(
                    "piper aborts during synthesis in this environment "
                    "(known macOS wheel bug: espeak-ng data path)"
                )
            from .tts_piper import PiperTTS

            return PiperTTS(
                model_path=s.piper_model_path,
                length_scale=s.piper_length_scale,
            )
        except Exception as exc:
            if not s.fallback_to_mock:
                raise
            log.warning("TTS 'piper' unavailable (%s); using mock", exc)
    return MockTTS()


@lru_cache(maxsize=4)
def _piper_works(model_path: str, timeout_s: float = 60.0) -> bool:
    """Synthesize one word in a throwaway process to prove Piper runs.

    Some macOS piper wheels have the build machine's espeak-ng data path
    compiled into the native bridge and abort the *process* -- not raise,
    abort -- so try/except around a lazy import cannot catch it, and the
    server would die on the first caller instead of degrading to another
    voice. Importing is not enough to detect it either: import and model load
    both succeed, and it only aborts once phonemization runs. So the probe has
    to actually synthesize, out of process, once per model path.
    """
    probe = (
        "from piper import PiperVoice;"
        f"v=PiperVoice.load({model_path!r});"
        "next(iter(v.synthesize('ok')))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, timeout=timeout_s
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def describe(stt, llm, tts) -> dict[str, str]:
    return {"stt": stt.name, "llm": llm.name, "tts": tts.name}
