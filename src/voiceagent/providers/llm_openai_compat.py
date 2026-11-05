"""Streaming LLM over the OpenAI chat-completions wire format.

One adapter, several free backends -- all of these speak this protocol:

  Ollama      http://localhost:11434/v1   (local, free, offline)
  llama.cpp   http://localhost:8080/v1    (local, free, offline)
  vLLM        http://localhost:8000/v1    (local, free)
  Groq        https://api.groq.com/openai/v1  (free tier, very low latency)

Only the base URL and model name change, so the agent can run offline on a
laptop and on a hosted free tier without a code change.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import Message


class OpenAICompatLLM:
    name = "openai-compat"

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:3b-instruct",
        api_key: str = "",
        max_tokens: int = 120,
        temperature: float = 0.4,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout_s
        )
        self.name = f"openai-compat:{model}"

    async def complete(
        self, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        }
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("", "[DONE]"):
                    continue
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece

    async def close(self) -> None:
        await self._client.aclose()
