import requests
import json
import os

class OllamaLLM:
    def __init__(self, temperature=0.7):
        self.temperature = temperature
        self.ollama_host = os.getenv("OLLAMA_HOST", "ollama_app")
        self.ollama_port = os.getenv("OLLAMA_PORT", "11411")

    def generate(self, user_input: str) -> str:
        # Endpoint chat
        url = f"http://{self.ollama_host}:{self.ollama_port}/v1/chat/completions"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            "model": "llama3:latest",
            "temperature": self.temperature
        }


        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=500)
            response.raise_for_status()
            data = response.json()
            # structure probable: {"choices": [{"message": {"role": "assistant", "content": "..."} }]}
            choices = data.get("choices", [])
            if not choices:
                return "Aucune réponse renvoyée par Ollama."
            return choices[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"Erreur lors de l'appel à Ollama : {e}"
