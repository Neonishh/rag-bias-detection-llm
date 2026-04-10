"""
LLM Client - wraps Google Gemini API calls
Authors: Navya G N
"""

import os
import requests


class LLMClient:
    def __init__(self):
        self.base_url = os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.timeout = int(os.environ.get("GEMINI_TIMEOUT", "60"))
        self.max_output_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
        self.max_continuations = int(os.environ.get("GEMINI_MAX_CONTINUATIONS", "2"))

        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Add it to your .env file before starting the backend."
            )

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
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        text, finish_reason = self._call_gemini(payload)

        # When Gemini stops due MAX_TOKENS, request continuation chunks.
        continuation_count = 0
        while finish_reason == "MAX_TOKENS" and continuation_count < self.max_continuations:
            continuation_prompt = (
                "Continue the answer from exactly where it stopped. "
                "Do not restart, do not repeat, and complete the final sentence.\n\n"
                f"Original user question:\n{prompt}\n\n"
                f"Current partial answer:\n{text}\n\n"
                "Write only the continuation."
            )

            continuation_payload = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": continuation_prompt}],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": self.max_output_tokens,
                },
            }

            next_text, finish_reason = self._call_gemini(continuation_payload)
            if not next_text:
                break

            if text and not text.endswith((" ", "\n")):
                text += " "
            text += next_text
            continuation_count += 1

        return text.strip()

    def _call_gemini(self, payload: dict) -> tuple[str, str]:
        """Single Gemini call. Returns (text, finish_reason)."""

        try:
            response = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(
                f"Gemini API error (HTTP {status}). Response: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to call Gemini API. Verify GEMINI_API_KEY, network access, and model name."
            ) from exc

        data = response.json()

        candidates = data.get("candidates", [])
        parts = []
        finish_reason = ""
        if candidates:
            first = candidates[0]
            finish_reason = first.get("finishReason", "")
            content = first.get("content", {})
            parts = content.get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()

        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text, finish_reason
