"""
LLM Client - wraps local Ollama API calls
Authors: Navya G N
"""

import os
import requests


class LLMClient:
    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call the LLM with a given prompt.
        Returns the generated text response.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question directly and naturally."
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to reach Ollama. Make sure Ollama is installed, the server is running, "
                "and OLLAMA_BASE_URL is correct."
            ) from exc

        data = response.json()
        text = data.get("response", "").strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return text
