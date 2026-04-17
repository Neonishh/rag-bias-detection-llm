import os
import requests

class GroqClient:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = int(os.environ.get("GROQ_TIMEOUT", "60"))
        self.max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "1024"))
        self.max_continuations = int(os.environ.get("GROQ_MAX_CONTINUATIONS", "1"))
        self.fallback_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        if not self.api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_text = ""
        current_prompt = prompt

        for _ in range(self.max_continuations + 1):
            chunk, finish_reason = self._request_groq(url, headers, current_prompt, system_prompt)
            if chunk:
                full_text = self._merge_chunks(full_text, chunk)

            if finish_reason != "length":
                break

            current_prompt = (
                "Your previous answer was cut off due to output length. Continue exactly "
                "from where it stopped without repeating any previous text.\n\n"
                f"Original user request:\n{prompt}\n\n"
                f"Previous partial answer:\n{full_text}\n\n"
                "Continue now:"
            )

        if not full_text.strip():
            raise RuntimeError("Groq returned an empty response.")

        return full_text.strip()

    def _request_groq(self, url: str, headers: dict, prompt: str, system_prompt: str):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        # Retry once with a known-live fallback model when the configured model
        # has been decommissioned.
        if response.status_code != 200 and "model_decommissioned" in response.text:
            for model in self.fallback_models:
                if model == self.model:
                    continue
                payload["model"] = model
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    break

        if response.status_code != 200:
            raise RuntimeError(f"Groq error: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return "", ""

        text = choices[0].get("message", {}).get("content", "")
        finish_reason = str(choices[0].get("finish_reason", "")).lower()
        return text, finish_reason

    def _merge_chunks(self, existing: str, new_chunk: str) -> str:
        if not existing:
            return new_chunk
        if existing.endswith(("\n", " ")) or new_chunk.startswith((" ", ",", ".", ";", ":", "!", "?")):
            return existing + new_chunk
        return existing + "\n\n" + new_chunk