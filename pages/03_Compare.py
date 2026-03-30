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
input, textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid #C8B89A !important;
    border-radius: 0 !important;
    font-family: 'Georgia', serif !important;
    color: #2C2416 !important;
}
[data-testid="stChatInput"] {
    background: #FFFDF5 !important;
    border: 1px solid #C8B89A !important;
    border-radius: 4px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #2C2416 !important;
    font-family: 'Georgia', serif !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: #1E3A5F !important;
    border-radius: 3px !important;
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
button[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#1E3A5F; background:#FECDD3; display:inline-block; padding:5px 20px 5px 14px; border-radius:10px; margin-bottom:4px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.12), inset 0 -1px 3px rgba(255,255,255,0.5);'>⚖️ Compare Models</div>",
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
    try:
        state = intent_check_node({"question": question})
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
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

    try:
        with st.spinner("Retrieving and reranking chunks..."):
            chunks = retrieve(question, top_k=20)
            reranked = rerank(question, chunks, top_n=5)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    def run_claude(q, c):
        return generate_answer(q, c, model="anthropic")

    def run_openai(q, c):
        return generate_answer(q, c, model="openai")

    try:
        with st.spinner("Generating answers from both models simultaneously..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_claude = executor.submit(run_claude, question, reranked)
                future_openai = executor.submit(run_openai, question, reranked)
                result_claude = future_claude.result()
                result_openai = future_openai.result()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

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
        <div style='background: #FFFDF5;
                    border: 1px solid #C8B89A;
                    border-top: none;
                    border-radius: 0 0 4px 4px;
                    padding: 16px;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.08);'>
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
        <div style='background: #FFFDF5;
                    border: 1px solid #C8B89A;
                    border-top: none;
                    border-radius: 0 0 4px 4px;
                    padding: 16px;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.08);'>
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