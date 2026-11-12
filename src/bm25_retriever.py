import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks_path="chunks.json"):
        path = Path(chunks_path)

        if not path.exists():
            raise FileNotFoundError(f"{chunks_path} not found")

        with path.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if not self.chunks:
            raise ValueError("No chunks found")

        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, top_k=5):
        tokens = self.tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    query = "What projects has Ibrahim developed?"
    results = retriever.search(query, top_k=3)

    print("\nBM25 SEARCH RESULTS")
    print("=" * 50)

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}...")
