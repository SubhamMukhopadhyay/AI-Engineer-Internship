"""
llm_client.py
--------------
Provider-agnostic OpenAI-compatible chat client, reused across projects.
No API key is hardcoded — configuration comes from environment variables.
"""

from __future__ import annotations

import os
import requests


class LLMConfigError(Exception):
    pass


class LLMRequestError(Exception):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        self.model = os.getenv("LLM_MODEL", "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _require_config(self) -> None:
        missing = [
            name
            for name, val in [
                ("LLM_API_KEY", self.api_key),
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL", self.model),
            ]
            if not val
        ]
        if missing:
            raise LLMConfigError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        timeout: int = 60,
    ) -> str:
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
            raise LLMRequestError(f"Request timed out after {timeout}s.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMRequestError(f"Could not connect to '{self.base_url}'.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMRequestError(f"Request failed: {exc}") from exc

        if resp.status_code == 401:
            raise LLMRequestError("Authentication failed (401). Check LLM_API_KEY.")
        if resp.status_code == 404:
            raise LLMRequestError(f"Endpoint not found (404) at '{url}'.")
        if resp.status_code == 429:
            raise LLMRequestError("Rate limit exceeded (429).")
        if resp.status_code >= 500:
            raise LLMRequestError(f"Provider server error ({resp.status_code}).")
        if resp.status_code != 200:
            raise LLMRequestError(f"LLM API returned status {resp.status_code}: {resp.text[:300]}")

        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMRequestError(f"Unexpected response shape: {exc}") from exc

        if not content or not content.strip():
            raise LLMRequestError("The LLM API returned an empty response.")
        return content.strip()
