from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st

from src.generation.generator import generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

st.set_page_config(page_title="Ask a Question", page_icon="💬")
st.title("Ask a Question")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        options=["openai", "anthropic"],
        format_func=lambda x: "GPT-4o" if x == "openai" else "Claude",
    )

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source_chunks"):
            with st.expander("Source chunks"):
                for i, chunk in enumerate(msg["source_chunks"], 1):
                    st.markdown(f"**Chunk {i}** (score: `{chunk.get('score', 'n/a'):.4f}`)")
                    st.caption(chunk["text"])

# --- Input ---
question = st.chat_input("Ask about your research papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                chunks = retrieve(question, top_k=20)
                reranked = rerank(question, chunks, top_n=5)
                result = generate_answer(question, reranked, model=model_choice)

                st.markdown(result["answer"])

                with st.expander("Source chunks"):
                    for i, chunk in enumerate(result["source_chunks"], 1):
                        st.markdown(f"**Chunk {i}** (score: `{chunk.get('score', 'n/a'):.4f}`)")
                        st.caption(chunk["text"])

                st.caption(f"Model: `{result['model']}` | Tokens: `{result['token_count']}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "source_chunks": result["source_chunks"],
                })

            except Exception as e:
                st.error(f"Error: {e}")
