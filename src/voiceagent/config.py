"""Runtime settings, read once from the environment / .env file.

Every provider option here is free. Three tiers:

  mock     no install, no network -- used by the tests and the rendered demo
  local    free and offline on your machine (whisper / ollama / say / piper)
  hosted   free tiers, no card required (Groq for the LLM)

There is no paid vendor in the default path, and no code change is needed to
move between tiers -- only these values.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- provider selection ------------------------------------------
    stt_provider: Literal["mock", "whisper"] = "mock"
    llm_provider: Literal["mock", "openai_compat"] = "mock"
    tts_provider: Literal["mock", "say", "piper", "espeak"] = "mock"

    # faster-whisper (free, local, offline)
    whisper_model: str = "tiny.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Any OpenAI-compatible endpoint. Defaults to a local Ollama install,
    # which is free and offline. Groq's free tier also speaks this protocol:
    #   LLM_BASE_URL=https://api.groq.com/openai/v1
    #   LLM_MODEL=llama-3.1-8b-instant
    #   LLM_API_KEY=gsk_...
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b-instruct"
    llm_api_key: str = ""
    llm_max_tokens: int = 120

    # macOS `say` (free, built in). Linux/Windows: use piper.
    say_voice: str = "Samantha"
    say_rate_wpm: int = 185

    # espeak-ng (free, cross platform, ~15 ms to first byte, robotic)
    espeak_binary: str = "espeak-ng"
    espeak_voice: str = "en-us"
    espeak_wpm: int = 170

    # Piper (free neural voice, loaded once and held in process)
    piper_model_path: str = ""
    piper_length_scale: float = 1.0

    public_ws_url: str = ""

    # ---- turn taking --------------------------------------------------
    end_of_turn_silence_ms: int = 640
    barge_in_speech_ms: int = 240
    max_history_turns: int = 8

    # Kept short on purpose: every token here is prompt latency on turn one,
    # and a small local model follows three tight rules better than ten loose
    # ones.
    system_prompt: str = (
        "You are Ravi, the phone assistant for a dental clinic in Bengaluru. "
        "You book, move and cancel appointments. "
        "Speak like a person on a phone call: one or two short sentences, no "
        "lists, no markdown, no emoji. Ask one question at a time. "
        "If the caller needs a human, say you will transfer them."
    )

    # ---- degradation -------------------------------------------------
    # A missing local dependency should downgrade to the mock and keep the call
    # up, not 500 the websocket. Set false to fail loudly instead.
    fallback_to_mock: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
