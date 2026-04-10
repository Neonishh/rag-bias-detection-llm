import os
import requests

class GroqClient:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")

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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"Groq error: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]