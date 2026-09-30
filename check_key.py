from dotenv import load_dotenv
load_dotenv()
import os, httpx

k = os.getenv("MISTRAL_API_KEY")
r = httpx.get("https://api.mistral.ai/v1/models",
              headers={"Authorization": f"Bearer {k}"})
print(r.status_code)
print(r.text[:200])

r = httpx.post("https://api.mistral.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {k}"},
    json={"model": "open-mistral-nemo",
          "messages": [{"role": "user", "content": "say hi"}]})
print(r.status_code, r.text[:200])