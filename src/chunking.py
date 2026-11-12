from pathlib import Path
import re
import json

DATA_DIR = Path("data")
CHUNK_SIZE = 250
OVERLAP = 40


def load_documents():
    documents = []

    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() not in [".txt", ".md"]:
            continue

        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def build_chunks():
    documents = load_documents()
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "source": document["source"],
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()

    output_path = Path("chunks.json")
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=" * 45)
    print("DOCUMENT CHUNKING COMPLETED")
    print("=" * 45)
    print(f"Documents found: {len(load_documents())}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        word_count = len(chunk["text"].split())
        print(
            f"Chunk {chunk['chunk_id']} | "
            f"{chunk['source']} | "
            f"{word_count} words"
        )

    print("\nSaved to chunks.json")
