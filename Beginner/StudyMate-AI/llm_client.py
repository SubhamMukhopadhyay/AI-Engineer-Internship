"""
llm_client.py
--------------
A thin, provider-agnostic abstraction over an OpenAI-compatible Chat
Completions API. Reads configuration from environment variables so
that no API key or model name is ever hardcoded in source.

Required environment variables (see .env.example):
    LLM_API_KEY   - secret key for the provider
    LLM_BASE_URL  - base URL of the OpenAI-compatible endpoint
                    (e.g. https://api.openai.com/v1, https://api.groq.com/openai/v1,
                    https://openrouter.ai/api/v1)
    LLM_MODEL     - model identifier to use for completions

Why this design:
    - Keeping the HTTP call in one place means every feature (summarize,
      quiz, explain, improve) shares the same error handling and timeout
      behaviour instead of duplicating fragile request code.
    - Using the standard OpenAI-compatible schema means the same code
      works against many providers (OpenAI, Groq, OpenRouter, local
      vLLM/Ollama-compatible servers, etc.) just by changing env vars.
"""

from __future__ import annotations

import os
import re
import requests

# Invisible/control characters that sometimes survive a copy-paste from
# rich text sources (smart keyboards, notes apps, chat UIs) but are
# invalid inside a URL/API key and cause silent 404s or auth failures.
# We strip these everywhere in the string, not just the ends.
_INVISIBLE_CHARS_RE = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0\u2028\u2029\r\n\t]"
)


def _sanitize(value: str) -> str:
    value = _INVISIBLE_CHARS_RE.sub("", value)
    return value.strip()


class LLMConfigError(Exception):
    """Raised when required environment configuration is missing."""


class LLMRequestError(Exception):
    """Raised when the upstream LLM API call fails or returns an error."""


class LLMClient:
    def __init__(self) -> None:
        self.api_key = _sanitize(os.getenv("LLM_API_KEY", ""))
        self.base_url = _sanitize(os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = _sanitize(os.getenv("LLM_MODEL", ""))

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _require_config(self) -> None:
        missing = []
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        if not self.model:
            missing.append("LLM_MODEL")
        if missing:
            raise LLMConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 1200,
        timeout: int = 60,
    ) -> str:
        """
        Send a single chat completion request and return the assistant
        text. Raises LLMConfigError / LLMRequestError on failure so the
        UI layer can present a clear, honest error instead of a fake
        response.
        """
        self._require_config()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMRequestError(
                f"The request to the LLM API timed out after {timeout}s. "
                "Try again or check LLM_BASE_URL."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMRequestError(
                f"Could not connect to '{self.base_url}'. Check LLM_BASE_URL "
                "and your network connection."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMRequestError(f"Request to the LLM API failed: {exc}") from exc

        if resp.status_code == 401:
            raise LLMRequestError(
                "Authentication failed (401). Check that LLM_API_KEY is correct."
            )
        if resp.status_code == 404:
            body_preview = resp.text[:200] if resp.text else "(empty body)"
            raise LLMRequestError(
                f"Endpoint not found (404) at '{resp.url}'. "
                f"Check LLM_BASE_URL. Provider response: {body_preview}"
            )
        if resp.status_code == 429:
            raise LLMRequestError(
                "Rate limit exceeded (429). Wait a moment and try again."
            )
        if resp.status_code >= 500:
            raise LLMRequestError(
                f"The LLM provider returned a server error ({resp.status_code})."
            )
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text[:300]
            raise LLMRequestError(
                f"LLM API returned status {resp.status_code}: {detail}"
            )

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMRequestError(
                f"Unexpected response shape from LLM API: {exc}"
            ) from exc

        if not content or not content.strip():
            raise LLMRequestError("The LLM API returned an empty response.")

        return content.strip()
