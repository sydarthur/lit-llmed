"""Lightweight client for interacting with a local Ollama instance."""

from __future__ import annotations

import json
from typing import Dict, Optional

import requests

from src.core.log import get_logger

LOGGER = get_logger(__name__)


class OllamaLLM:
    """Convenience wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama2:latest",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"

    def summarize_text(self, text: str, max_words: int = 300) -> Optional[str]:
        prompt = (
            "Please provide a concise summary of the following text in approximately "
            f"{max_words} words. Focus on the main points, key findings, and important conclusions:\n\n{text}\n\nSummary:"
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        return self._make_request(self.chat_url, payload)

    def chat(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        return self._make_request(self.chat_url, payload)

    def test_connection(self) -> bool:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        }
        response = self._make_request(self.chat_url, payload)
        return response is not None

    def _make_request(self, endpoint: str, payload: Dict) -> Optional[str]:
        try:
            response = requests.post(endpoint, json=payload, stream=True, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network-dependent
            LOGGER.error("ollama_request_failed", error=str(exc), endpoint=endpoint)
            return None

        full_response = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover
                LOGGER.warning("ollama_response_decode_failed", error=str(exc))
                continue
            if "message" in data and "content" in data["message"]:
                full_response.append(data["message"]["content"])
            elif "response" in data:
                full_response.append(data["response"])
            if data.get("done"):
                break
        text = "".join(full_response).strip()
        return text or None
