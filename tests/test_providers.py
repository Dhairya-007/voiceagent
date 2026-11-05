"""Provider selection and graceful degradation.

The rule under test: a provider that cannot start must downgrade to the mock
and keep the call up. A caller hearing a slightly dumber bot beats a caller
hearing dead air -- but only if the downgrade is detected at build time, not on
the first word the agent tries to say.
"""
from __future__ import annotations

import pytest

from voiceagent.config import Settings
from voiceagent.providers import build_llm, build_stt, build_tts, describe
from voiceagent.providers.mock import MockLLM, MockSTT, MockTTS


def test_defaults_are_mocks_so_the_repo_runs_with_no_setup():
    s = Settings()
    stt, llm, tts = build_stt(s), build_llm(s), build_tts(s)
    assert isinstance(stt, MockSTT)
    assert isinstance(llm, MockLLM)
    assert isinstance(tts, MockTTS)
    assert describe(stt, llm, tts) == {
        "stt": "mock-stt", "llm": "mock-llm", "tts": "mock-tts",
    }


def test_missing_piper_model_degrades_to_mock():
    tts = build_tts(Settings(tts_provider="piper", piper_model_path="nope.onnx"))
    assert tts.name == "mock-tts"


def test_piper_without_a_model_path_degrades_to_mock():
    assert build_tts(Settings(tts_provider="piper")).name == "mock-tts"


def test_missing_espeak_binary_degrades_to_mock():
    tts = build_tts(
        Settings(tts_provider="espeak", espeak_binary="espeak-ng-does-not-exist")
    )
    assert tts.name == "mock-tts"


def test_fallback_can_be_disabled_to_fail_loudly():
    """Degrading silently is right in production and wrong in CI."""
    with pytest.raises(Exception):
        build_tts(
            Settings(
                tts_provider="espeak",
                espeak_binary="espeak-ng-does-not-exist",
                fallback_to_mock=False,
            )
        )


def test_llm_base_url_selects_the_backend_without_a_code_change():
    """Ollama, llama.cpp, vLLM and Groq are all just a different base URL."""
    llm = build_llm(
        Settings(
            llm_provider="openai_compat",
            llm_base_url="https://api.groq.com/openai/v1",
            llm_model="llama-3.1-8b-instant",
            llm_api_key="test-key-not-real",
        )
    )
    assert "llama-3.1-8b-instant" in llm.name
