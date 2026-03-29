from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st
from src.retrieval.vector_store import VectorStore

st.set_page_config(page_title="My Library", page_icon="📚", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #EDE8DF !important;
    border-right: 3px solid #C8B89A !important;
    box-shadow: 3px 0 15px rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Georgia', serif !important;
    color: #2C2416 !important;
    font-size: 15px !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #FAF7F0 !important;
    box-shadow: inset 6px 0 20px rgba(180,160,120,0.12),
                inset -6px 0 20px rgba(180,160,120,0.12) !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 860px !important;
    padding-left: 60px !important;
    padding-right: 60px !important;
    padding-top: 2.5rem !important;
}
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    color: #2C2416 !important;
    line-height: 1.8 !important;
}
h1, h2, h3 {
    font-family: 'Georgia', serif !important;
    color: #1E3A5F !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border-bottom: 1px solid #C8B89A !important;
    padding-bottom: 8px !important;
}
p, span, label { font-size: 15px !important; }
.stButton > button[kind="primary"] {
    background: #1E3A5F !important;
    border: none !important;
    color: white !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.05em !important;
    font-weight: 400 !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #DC2626 !important;
    color: #2C2416 !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
    font-size: 12px !important;
    padding: 2px 10px !important;
    min-width: 80px !important;
    white-space: nowrap !important;
}
input, textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid #C8B89A !important;
    border-radius: 0 !important;
    font-family: 'Georgia', serif !important;
    color: #2C2416 !important;
}
[data-testid="stError"] {
    background: #FEF2F2 !important;
    border-left: 3px solid #DC2626 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#1E3A5F; background:#CFFAFE; display:inline-block; padding:5px 20px 5px 14px; border-radius:10px; margin-bottom:4px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.12), inset 0 -1px 3px rgba(255,255,255,0.5);'>📚 My Library</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color:#64748B; margin-bottom:24px;'>Papers you uploaded in this session.</div>",
    unsafe_allow_html=True
)

uploaded = st.session_state.get("uploaded_papers", [])

if not uploaded:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size:48px; margin-bottom:16px;'>📭</div>
        <div style='font-size:18px; color:#1E3A5F;'>No papers uploaded in this session.</div>
        <div style='font-size:14px; color:#64748B; margin-top:8px;'>Head to the Upload page to add your first paper.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

store = VectorStore()

for paper_name in uploaded:
    chunk_count = store.chunk_count_by_name(paper_name)

    st.markdown(f"""
    <div style='background: #FFFDF5;
                border: 1px solid #C8B89A;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 4px;
                display: flex;
                align-items: center;
                justify-content: space-between;'>
        <div>
            <div style='font-size:15px; font-weight:500; color:#1E3A5F;'>{paper_name}</div>
            <div style='font-size:12px; color:#64748B; margin-top:3px;'>{chunk_count} chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_gap, col_btn = st.columns([11, 1])
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