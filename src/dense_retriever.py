import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        chunks_path="chunks.json",
        embeddings_path="embeddings.npy",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunks_path = Path(chunks_path)
        self.embeddings_path = Path(embeddings_path)

        if not self.chunks_path.exists():
            raise FileNotFoundError("chunks.json not found")

        with open(self.chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if not self.chunks:
            raise ValueError("No chunks found")

        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer(model_name)

        if self.embeddings_path.exists():
            print("Loading saved embeddings...")
            self.embeddings = np.load(self.embeddings_path)

            if self.embeddings.shape[0] != len(self.chunks):
                raise ValueError(
                    "Saved embeddings do not match current chunks. "
                    "Delete embeddings.npy and run again."
                )
        else:
            print("Generating document embeddings...")

            texts = [chunk["text"] for chunk in self.chunks]

            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            np.save(self.embeddings_path, self.embeddings)
            print("Embeddings saved successfully!")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Cosine similarity using normalized vectors
        scores = np.dot(
            self.embeddings,
            query_embedding[0]
        )

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            result = self.chunks[int(index)].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    query = "What kind of professional work have I done?"

    results = retriever.search(query, top_k=3)

    print("\nDENSE SEARCH RESULTS")
    print("=" * 50)

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:300]}...")

    print("\nEmbeddings file:", retriever.embeddings_path)
