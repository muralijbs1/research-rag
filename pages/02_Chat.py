from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st

from src.graph.rag_graph import rag_pipeline

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
                state = rag_pipeline.invoke({
                    "question": question,
                    "original_question": question,
                    "chunks": [],
                    "reranked": [],
                    "source_chunks": [],
                    "answer": "",
                    "retries": 0,
                    "quality_passed": False,
                })

                if state.get("error"):
                    st.markdown(state["error"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": state["error"],
                        "source_chunks": [],
                    })
                else:
                    answer = state["answer"]
                    source_chunks = state.get("source_chunks", [])

                    st.markdown(answer)

                    with st.expander("Source chunks"):
                        for i, chunk in enumerate(source_chunks, 1):
                            st.markdown(f"**Chunk {i}** (score: `{chunk.get('score', 'n/a'):.4f}`)")
                            st.caption(chunk["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_chunks": source_chunks,
                    })

            except Exception as e:
                st.error(f"Error: {e}")