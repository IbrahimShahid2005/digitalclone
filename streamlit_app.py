
import os
import streamlit as st

from src.hybrid_retriever import HybridRetriever
from src.generator import generate_answer, fallback_answer

st.set_page_config(
    page_title="Ibrahim's Digital Clone",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Ibrahim's Digital Clone")
st.subheader("Your personal AI-powered knowledge assistant")

st.info("Disclaimer: This is an AI clone, not the real person.")

# Load API key from Streamlit secrets
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

@st.cache_resource
def load_retriever():
    return HybridRetriever()

try:
    retriever = load_retriever()
except Exception as e:
    st.error(f"Retrieval loading error: {e}")
    st.stop()

mode = st.selectbox(
    "Retrieval Mode",
    ["Hybrid", "BM25 only", "Dense only"]
)

question = st.text_input(
    "Ask me anything",
    placeholder="What am I studying?"
)

if st.button("Ask Digital Clone", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Thinking..."):

            if mode == "BM25 only":
                results = retriever.bm25.search(question, top_k=5)

            elif mode == "Dense only":
                results = retriever.dense.search(question, top_k=5)

            else:
                results = retriever.search(question, top_k=5)

            if not results:
                answer = "I don't have that in my notes."

            else:
                try:
                    answer = generate_answer(question, results)
                except Exception:
                    answer = fallback_answer(question, results)

        st.markdown("## Answer")
        st.markdown(answer)

        with st.expander("Retrieved Passages and Scores"):
            for i, item in enumerate(results, 1):
                score = item.get(
                    "rrf_score",
                    item.get("score", 0)
                )

                st.markdown(f"### Result {i}")
                st.write("Source:", item.get("source"))
                st.write("Score:", round(score, 6))
                st.write(item.get("text"))

                st.divider()
