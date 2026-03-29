from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st
from src.retrieval.vector_store import VectorStore

st.set_page_config(page_title="My Library", page_icon="📚", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to right, #BEB6AA, #C8C0B4, #A89E92) !important;
    border-right: 18px solid #1A1020 !important;
    box-shadow: inset -20px 0 40px rgba(0,0,0,0.28) !important;
}
[data-testid="stSidebar"] * { color: #2E2820 !important; font-size: 15px !important; }
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(to right, #2A2040 0%, #17112E 45%, #09080F 100%) !important;
    box-shadow: inset -24px 0 60px rgba(0,0,0,0.55) !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 2rem !important;
}
html, body, [data-testid="stAppViewContainer"] { color: rgba(200,195,225,0.85) !important; }
h1, h2, h3 { color: rgba(220,215,240,0.92) !important; font-weight: 500 !important; }
p, span, label { font-size: 15px !important; }
.stButton > button[kind="primary"] {
    background: rgba(124,58,237,0.85) !important;
    border: none !important;
    color: rgba(255,255,255,0.92) !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 0.5px solid rgba(124,58,237,0.4) !important;
    color: rgba(167,139,250,0.7) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 2px 10px !important;
}
input, textarea {
    background: rgba(20,14,36,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    color: rgba(200,195,225,0.85) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.3) !important;
}
[data-testid="stError"] {
    background: rgba(180,40,40,0.12) !important;
    border-left: 3px solid rgba(200,60,60,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#F0C060; margin-bottom:4px;'>📚 My Library</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color:rgba(200,195,225,0.75); margin-bottom:24px;'>Papers you uploaded in this session.</div>",
    unsafe_allow_html=True
)

uploaded = st.session_state.get("uploaded_papers", [])

if not uploaded:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size:48px; margin-bottom:16px;'>📭</div>
        <div style='font-size:18px; color:rgba(200,195,225,0.75);'>No papers uploaded in this session.</div>
        <div style='font-size:14px; color:rgba(200,195,225,0.55); margin-top:8px;'>Head to the Upload page to add your first paper.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

store = VectorStore()

for paper_name in uploaded:
    chunk_count = store.chunk_count_by_name(paper_name)

    st.markdown(f"""
    <div style='background: rgba(42,32,64,0.45);
                border: 0.5px solid rgba(255,255,255,0.07);
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 4px;
                display: flex;
                align-items: center;
                justify-content: space-between;'>
        <div>
            <div style='font-size:15px; font-weight:500; color:rgba(220,215,240,0.9);'>{paper_name}</div>
            <div style='font-size:12px; color:rgba(200,195,225,0.55); margin-top:3px;'>{chunk_count} chunks</div>
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