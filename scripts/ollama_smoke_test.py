import httpx

response = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-coder:7b",
        "prompt": "Explain what RAG is in 2 sentences.",
        "stream": False
    },
    timeout=1200
)

data = response.json()

print(data["response"])