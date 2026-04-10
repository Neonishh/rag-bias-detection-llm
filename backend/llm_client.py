"""
LLM Client - wraps Google Gemini API calls
Authors: Navya G N
"""

import os
import requests
from groq_client import GroqClient



class LLMClient:
    def __init__(self):
        self.base_url = os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.timeout = int(os.environ.get("GEMINI_TIMEOUT", "60"))

        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Add it to your .env file before starting the backend."
            )

    # def generate(self, prompt: str, system_prompt: str = None) -> str:
    #     """
    #     Call the LLM with a given prompt.
    #     Returns the generated text response.
    #     """
    #     if system_prompt is None:
    #         system_prompt = (
    #             "You are a helpful assistant. Answer the user's question directly and naturally."
    #         )

    #     payload = {
    #         "system_instruction": {
    #             "parts": [{"text": system_prompt}],
    #         },
    #         "contents": [
    #             {
    #                 "role": "user",
    #                 "parts": [{"text": prompt}],
    #             }
    #         ],
    #         "generationConfig": {
    #             "maxOutputTokens": 512,
    #         },
    #     }

    #     try:
    #         response = requests.post(
    #             f"{self.base_url}/models/{self.model}:generateContent",
    #             params={"key": self.api_key},
    #             json=payload,
    #             timeout=self.timeout,
    #         )
    #         response.raise_for_status()
    #     except requests.HTTPError as exc:
    #         status = exc.response.status_code if exc.response is not None else "unknown"
    #         body = exc.response.text if exc.response is not None else ""
    #         raise RuntimeError(
    #             f"Gemini API error (HTTP {status}). Response: {body}"
    #         ) from exc
    #     except requests.RequestException as exc:
    #         raise RuntimeError(
    #             "Unable to call Gemini API. Verify GEMINI_API_KEY, network access, and model name."
    #         ) from exc

    #     data = response.json()

    #     candidates = data.get("candidates", [])
    #     parts = []
    #     if candidates:
    #         content = candidates[0].get("content", {})
    #         parts = content.get("parts", [])
    #     text = "".join(part.get("text", "") for part in parts).strip()

    #     if not text:
    #         raise RuntimeError("Gemini returned an empty response.")

    #     return text
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        try:
            return self._call_gemini(prompt, system_prompt)
        except Exception as e:
            print("⚠️ Gemini failed, switching to Groq...")
            return self._call_groq(prompt, system_prompt)


    def _call_gemini(self, prompt, system_prompt):
        # your existing Gemini code here (unchanged)
        return super().generate(prompt, system_prompt)


    def _call_groq(self, prompt, system_prompt):
        client = GroqClient()
        return client.generate(prompt, system_prompt)
