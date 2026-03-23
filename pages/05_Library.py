from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st

from src.retrieval.vector_store import VectorStore

st.set_page_config(page_title="My Library", page_icon="📚")
st.title("My Library")
st.caption("Papers you uploaded in this session.")

uploaded = st.session_state.get("uploaded_papers", [])

if not uploaded:
    st.info("No papers uploaded in this session.", icon="📭")
    st.stop()

store = VectorStore()

for paper_name in uploaded:
    chunk_count = store.chunk_count_by_name(paper_name)
    col_name, col_chunks, col_btn = st.columns([5, 2, 1])

    with col_name:
        st.markdown(f"**{paper_name}**")
    with col_chunks:
        st.caption(f"{chunk_count} chunks")
    with col_btn:
        if st.button("Delete", key=f"del_{paper_name}", type="secondary"):
            info = next(
                (p for p in store.list_papers_with_info() if p["paper_name"] == paper_name),
                None,
            )
            if info and info.get("source") == "system":
                st.error("System papers cannot be deleted.", icon="🔒")
            else:
                store.delete(paper_name)
                st.session_state.uploaded_papers.remove(paper_name)
                st.rerun()

    st.divider()
