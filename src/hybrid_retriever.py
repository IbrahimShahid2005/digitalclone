from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(self, k=60):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.k = k

    def search(self, query, top_k=5):
        bm25_results = self.bm25.search(query, top_k=10)
        dense_results = self.dense.search(query, top_k=10)

        fused_scores = {}
        chunk_lookup = {}

        for results in [bm25_results, dense_results]:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0)
                    + 1 / (self.k + rank)
                )

                chunk_lookup[chunk_id] = result

        ranked_ids = sorted(
            fused_scores,
            key=lambda cid: fused_scores[cid],
            reverse=True
        )

        final_results = []

        for chunk_id in ranked_ids[:top_k]:
            result = chunk_lookup[chunk_id].copy()
            result["rrf_score"] = fused_scores[chunk_id]
            final_results.append(result)

        return final_results


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "What are Ibrahim's projects and professional experiences?"

    results = retriever.search(query, top_k=5)

    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 50)

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Text: {result['text'][:300]}...")

    print("\nHybrid retrieval completed successfully!")
