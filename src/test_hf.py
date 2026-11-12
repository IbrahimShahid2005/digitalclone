import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="groq",
    api_key=os.environ["HF_TOKEN"]
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Say Digital Clone is working!"}
    ],
    max_tokens=50
)

print(response.choices[0].message.content)
