import os
import re
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.8-flash"

STOP_WORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "am", "is", "are", "was", "were", "do", "does", "did",
    "i", "my", "me", "the", "a", "an", "and", "or", "in",
    "on", "at", "to", "of", "for", "with", "about", "tell",
    "can", "have", "has", "been", "your", "you", "it"
}

RELATED_TERMS = {
    "studying": {"study", "degree", "education", "university"},
    "study": {"studying", "degree", "education"},
    "degree": {"studying", "study", "education"},
    "skills": {"technologies", "programming", "languages"},
    "project": {"projects", "developed", "built"},
    "projects": {"project", "developed", "built"},
    "internship": {"intern", "experience", "bigbrains"},
    "intern": {"internship", "experience"},
    "fyp": {"final", "year", "project"},
    "hobbies": {"hobby", "gaming", "games", "videos"},
}


def tokenize(text):
    words = set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
    return words - STOP_WORDS


def expand_keywords(words):
    expanded = set(words)

    for word in words:
        expanded.update(RELATED_TERMS.get(word, set()))

    return expanded


def extract_faq_entries(text):
    pattern = re.compile(
        r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Q:|$)",
        re.IGNORECASE | re.DOTALL
    )

    return [
        (match.group(1).strip(), match.group(2).strip())
        for match in pattern.finditer(text)
        if match.group(1).strip() and match.group(2).strip()
    ]


def fallback_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't have that in my notes."

    query_words = expand_keywords(tokenize(question))
    candidates = []

    for chunk in retrieved_chunks:
        text = chunk.get("text", "")

        # FAQ matching considers both the FAQ question and its answer.
        for faq_question, faq_answer in extract_faq_entries(text):
            question_words = expand_keywords(tokenize(faq_question))
            answer_words = expand_keywords(tokenize(faq_answer))

            overlap = query_words & question_words
            answer_overlap = query_words & answer_words

            score = (
                3 * len(overlap) +
                len(answer_overlap)
            )

            if score > 0:
                candidates.append(
                    (score, faq_answer, chunk.get("source", "Unknown"))
                )

        # Also consider ordinary document sentences.
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()

            if len(sentence.split()) < 5:
                continue

            sentence_words = expand_keywords(tokenize(sentence))
            overlap = query_words & sentence_words

            if overlap:
                candidates.append(
                    (
                        len(overlap),
                        sentence,
                        chunk.get("source", "Unknown")
                    )
                )

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected = []
    seen = set()

    for score, answer, source in candidates:
        normalized = answer.lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append(answer)

        if len(selected) == 2:
            break

    if selected:
        result = "\n\n".join(selected)
    else:
        # Never discard available context just because matching failed.
        result = (
            retrieved_chunks[0].get("text", "")[:700]
        )

    return (
        "*Gemini is temporarily unavailable. "
        "This answer is extracted from my knowledge base.*\n\n"
        + result
    )


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't have that in my notes."

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or len(api_key) < 20:
        print("Gemini unavailable. Using retrieval fallback.", flush=True)
        return fallback_answer(question, retrieved_chunks)

    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are Ibrahim Shahid Butt's AI Digital Clone.

Speak naturally in first person as Ibrahim.
Be casual and friendly.
Use a professional tone for academic and career topics.

Only use the supplied context.
Never invent personal information.
Answer directly and concisely.
If information is unavailable, say:
"I don't have that in my notes."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    client = None

    try:
        print("Sending request to Gemini...", flush=True)

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20000)
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )

        if response.text and response.text.strip():
            return response.text.strip()

    except Exception as error:
        print(
            f"Gemini unavailable: {type(error).__name__}: {error}",
            flush=True
        )

    finally:
        if client is not None:
            client.close()

    return fallback_answer(question, retrieved_chunks)
