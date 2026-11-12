import gradio as gr

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.generator import generate_answer

print("Loading Digital Clone...", flush=True)

bm25 = BM25Retriever()
dense = DenseRetriever()
hybrid = HybridRetriever()

print("All retrieval systems loaded!", flush=True)


def chat(question, mode):
    if not question or not question.strip():
        return "Please enter a question.", "No question submitted."

    try:
        if mode == "BM25 only":
            results = bm25.search(question, top_k=5)
        elif mode == "Dense only":
            results = dense.search(question, top_k=5)
        else:
            results = hybrid.search(question, top_k=5)

        print(f"Retrieved {len(results)} passages.", flush=True)

        if not results:
            return "I don't have that in my notes.", "No passages retrieved."

        answer = generate_answer(question, results)

        details = []

        for i, item in enumerate(results, 1):
            score = item.get("rrf_score", item.get("score", 0))

            details.append(
                f"RESULT {i}\n"
                f"Source: {item.get('source', 'Unknown')}\n"
                f"Chunk ID: {item.get('chunk_id', 'Unknown')}\n"
                f"Score: {score:.6f}\n"
                f"Text:\n{item.get('text', '')}\n"
                f"{'=' * 60}"
            )

        source_text = "\n\n".join(details)

        print("Answer and retrieved passages returned.", flush=True)

        return answer, source_text

    except Exception as error:
        import traceback
        traceback.print_exc()

        return (
            f"Application error: {type(error).__name__}: {error}",
            "An error occurred. Check the PowerShell terminal."
        )


with gr.Blocks(
    title="Ibrahim's Digital Clone",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # Ibrahim's Digital Clone

    ### Your personal AI-powered knowledge assistant

    **DISCLAIMER: This is an AI clone, not the real person.**

    Ask questions about Ibrahim's education, skills, projects,
    professional experience and interests.
    """)

    mode = gr.Dropdown(
        choices=["BM25 only", "Dense only", "Hybrid"],
        value="Hybrid",
        label="Retrieval Mode"
    )

    question = gr.Textbox(
        label="Ask me anything",
        placeholder="What am I studying?",
        lines=2
    )

    submit = gr.Button("Ask Digital Clone", variant="primary")

    gr.Markdown("## Answer")
    answer = gr.Markdown()

    gr.Markdown("## Retrieved Passages and Scores")
    sources = gr.Code(
    label="Retrieved Passages and Scores",
    language="markdown",
    lines=25,
    interactive=False
    )

    submit.click(
        fn=chat,
        inputs=[question, mode],
        outputs=[answer, sources]
    )

    question.submit(
        fn=chat,
        inputs=[question, mode],
        outputs=[answer, sources]
    )


if __name__ == "__main__":
    demo.launch()


