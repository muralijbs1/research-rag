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

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #C9D8E8 !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * { color: #1E3A5F !important; font-size: 15px !important; }
[data-testid="stAppViewContainer"] > .main {
    background: #F8FAFC !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 2rem !important;
}
html, body, [data-testid="stAppViewContainer"] { color: #1E3A5F !important; }
h1, h2, h3 { color: #1E3A5F !important; font-weight: 500 !important; }
p, span, label { font-size: 15px !important; }
input, textarea {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E3A5F !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1E3A5F !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: #1E3A5F !important;
    border-radius: 8px !important;
    color: white !important;
    border: none !important;
}
[data-testid="stWarning"] {
    background: #FFF7ED !important;
    border-left: 3px solid #D97706 !important;
}
[data-testid="stError"] {
    background: #FEF2F2 !important;
    border-left: 3px solid #DC2626 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#1E3A5F; background:#FECDD3; display:inline-block; padding:5px 20px 5px 14px; border-radius:10px; margin-bottom:4px;'>⚖️ Compare Models</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color:#64748B; margin-bottom:20px;'>Same question, two models, side by side.</div>",
    unsafe_allow_html=True
)

question = st.chat_input("Ask a question to compare both models...")

if question:
    # Question banner
    st.markdown(f"""
    <div style='background: #EFF6FF;
                border-left: 3px solid #2563EB;
                border-radius: 6px;
                padding: 12px 16px;
                color: #1E3A5F;
                font-size:15px;
                margin-bottom: 20px;'>
        🔍 <strong>Question:</strong> {question}
    </div>
    """, unsafe_allow_html=True)

    # Intent check
    state = intent_check_node({"question": question})
    if state.get("error"):
        # Intent rejection
        st.markdown(f"""
        <div style='background: #FFF7ED;
                    border-left: 3px solid #D97706;
                    border-radius: 6px;
                    padding: 12px 16px;
                    color: #92400E;
                    font-size:15px;
                    margin-bottom: 20px;'>
            🤖 {state["error"]}
        </div>
        """, unsafe_allow_html=True)
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

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div style='background: #D97706;
                    border-radius: 10px 10px 0 0;
                    padding: 10px 16px;
                    display: flex;
                    align-items: center;
                    gap: 8px;'>
            <div style='width:8px; height:8px; border-radius:50%; background:rgba(255,255,255,0.7);'></div>
            <span style='font-size:14px; font-weight:500; color:white;'>Claude Haiku</span>
        </div>
        <div style='background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-top: none;
                    border-radius: 0 0 10px 10px;
                    padding: 16px;'>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result_claude["answer"])

        claude_papers = list(dict.fromkeys(
            c.get("paper_name", "") for c in result_claude["source_chunks"]
        ))
        claude_papers = [p for p in claude_papers if p]
        if claude_papers:
            chips = "".join(
                f"<span style='display:inline-block; background:#FFF7ED; color:#D97706; font-size:11px; padding:3px 10px; border-radius:20px; border:1px solid #FDE68A; margin-right:6px; margin-top:6px;'>📄 {p}</span>"
                for p in claude_papers
            )
            st.markdown(f"<div style='margin-top:8px;'>{chips}</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div style='background: #059669;
                    border-radius: 10px 10px 0 0;
                    padding: 10px 16px;
                    display: flex;
                    align-items: center;
                    gap: 8px;'>
            <div style='width:8px; height:8px; border-radius:50%; background:rgba(255,255,255,0.7);'></div>
            <span style='font-size:14px; font-weight:500; color:white;'>GPT-4o-mini</span>
        </div>
        <div style='background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-top: none;
                    border-radius: 0 0 10px 10px;
                    padding: 16px;'>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(result_openai["answer"])

        openai_papers = list(dict.fromkeys(
            c.get("paper_name", "") for c in result_openai["source_chunks"]
        ))
        openai_papers = [p for p in openai_papers if p]
        if openai_papers:
            chips = "".join(
                f"<span style='display:inline-block; background:#F0FDF4; color:#16A34A; font-size:11px; padding:3px 10px; border-radius:20px; border:1px solid #BBF7D0; margin-right:6px; margin-top:6px;'>📄 {p}</span>"
                for p in openai_papers
            )
            st.markdown(f"<div style='margin-top:8px;'>{chips}</div>", unsafe_allow_html=True)