
import os
import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from google import genai

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.json"

app = FastAPI(title="Ibrahim's Digital Clone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge base
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

corpus = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(corpus)

class ChatRequest(BaseModel):
    question: str
    mode: str = "Hybrid"

def retrieve(question, top_k=5):
    tokens = tokenize(question)

    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked:
        if scores[i] <= 0:
            continue

        item = chunks[i].copy()
        item["score"] = float(scores[i])
        results.append(item)

        if len(results) >= top_k:
            break

    return results

def fallback_answer(question, results):
    if not results:
        return "I don't have that in my notes."

    question_words = set(tokenize(question))
    candidates = []

    for item in results:
        text = item["text"]

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()

            if len(sentence) < 20:
                continue

            words = set(tokenize(sentence))
            overlap = len(question_words & words)

            if overlap:
                candidates.append((overlap, sentence))

    candidates.sort(key=lambda x: x[0], reverse=True)

    answers = []
    for _, sentence in candidates:
        if sentence not in answers:
            answers.append(sentence)
        if len(answers) == 3:
            break

    if not answers:
        return "I don't have that in my notes."

    return (
        "*Gemini is temporarily unavailable. "
        "This answer is extracted from my knowledge base.*\n\n"
        + "\n\n".join(answers)
    )

def generate_answer(question, results):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback_answer(question, results)

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in results
    )

    prompt = f"""
You are Ibrahim Shahid Butt's AI Digital Clone.

Speak naturally in first person as Ibrahim.
Be casual and friendly.
Use a professional tone for academic topics.

Only use the supplied context.
Never invent personal information.
If information is unavailable, say:
I don't have that in my notes.

CONTEXT:
{context}

QUESTION:
{question}

Give a concise, natural answer.
"""

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=prompt
        )

        if response.text:
            return response.text

        return fallback_answer(question, results)

    except Exception as error:
        print("Gemini error:", error, flush=True)
        return fallback_answer(question, results)

    finally:
        client.close()

@app.get("/")
def home():
    return {"message": "Ibrahim's Digital Clone API is running!"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "knowledge_chunks": len(chunks)
    }

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    results = retrieve(question)

    if not results:
        return {
            "answer": "I don't have that in my notes.",
            "sources": []
        }

    answer = generate_answer(question, results)

    sources = [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "score": round(item["score"], 6),
            "text": item["text"]
        }
        for item in results
    ]

    return {
        "answer": answer,
        "sources": sources
    }
