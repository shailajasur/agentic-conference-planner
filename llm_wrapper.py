
import requests

class HuggingFaceLLM:
    def __init__(self, model_url=None):
        self.model_url = model_url or "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json"
        }

    def _get_token(self):
        import os
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("Please set your Hugging Face token in the HF_TOKEN environment variable.")
        return token

    def generate(self, prompt, max_tokens=300):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        response = requests.post(self.model_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")
        generated = response.json()
        if isinstance(generated, list) and "generated_text" in generated[0]:
            return generated[0]["generated_text"]
        elif isinstance(generated, dict) and "generated_text" in generated:
            return generated["generated_text"]
        else:
            return str(generated)
