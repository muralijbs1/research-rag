from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import concurrent.futures

import streamlit as st

from src.generation.generator import generate_answer
from src.graph.nodes import intent_check_node
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

st.set_page_config(page_title="Compare Models", page_icon="⚖️", layout="wide")
st.title("Claude Haiku vs GPT-4o-mini")

question = st.chat_input("Ask a question to compare both models...")

if question:
    # --- Intent check ---
    state = intent_check_node({"question": question})
    if state.get("error"):
        st.warning(state["error"])
        st.stop()

    with st.spinner("Retrieving and reranking chunks..."):
        chunks = retrieve(question, top_k=20)
        reranked = rerank(question, chunks, top_n=5)

    def run_claude(q, c):
        return generate_answer(q, c, model="anthropic")

    def run_openai(q, c):
        return generate_answer(q, c, model="openai")

    with st.spinner("Generating answers from both models simultaneously..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_claude = executor.submit(run_claude, question, reranked)
            future_openai = executor.submit(run_openai, question, reranked)
            result_claude = future_claude.result()
            result_openai = future_openai.result()

    st.markdown(f"**Question:** {question}")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Claude Haiku")
        st.caption(f"Model: `{result_claude['model']}` | Tokens: `{result_claude['token_count']}`")
        st.markdown(result_claude["answer"])
        with st.expander("Source chunks"):
            for i, chunk in enumerate(result_claude["source_chunks"], 1):
                st.markdown(f"**Chunk {i}** (score: `{chunk.get('score', 'n/a'):.4f}`)")
                st.caption(chunk["text"])

    with col_right:
        st.subheader("GPT-4o-mini")
        st.caption(f"Model: `{result_openai['model']}` | Tokens: `{result_openai['token_count']}`")
        st.markdown(result_openai["answer"])
        with st.expander("Source chunks"):
            for i, chunk in enumerate(result_openai["source_chunks"], 1):
                st.markdown(f"**Chunk {i}** (score: `{chunk.get('score', 'n/a'):.4f}`)")
                st.caption(chunk["text"])