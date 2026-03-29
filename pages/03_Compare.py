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
input, textarea {
    background: rgba(20,14,36,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    color: rgba(200,195,225,0.85) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"] {
    background: rgba(42,32,64,0.6) !important;
    border: 0.5px solid rgba(124,58,237,0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: rgba(200,195,225,0.9) !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: rgba(124,58,237,0.85) !important;
    border-radius: 8px !important;
    color: white !important;
    border: none !important;
}
[data-testid="stWarning"] {
    background: rgba(180,120,0,0.12) !important;
    border-left: 3px solid rgba(180,140,0,0.6) !important;
}
[data-testid="stError"] {
    background: rgba(180,40,40,0.12) !important;
    border-left: 3px solid rgba(200,60,60,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#F0C060; margin-bottom:4px;'>⚖️ Compare Models</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color:rgba(200,195,225,0.6); margin-bottom:20px;'>Same question, two models, side by side.</div>",
    unsafe_allow_html=True
)

question = st.chat_input("Ask a question to compare both models...")

if question:
    # Question banner — teal
    st.markdown(f"""
    <div style='background: rgba(5,100,80,0.2);
                border-left: 3px solid rgba(20,180,140,0.7);
                border-radius: 6px;
                padding: 12px 16px;
                color: rgba(80,220,180,0.9);
                font-size:15px;
                margin-bottom: 20px;'>
        🔍 <strong>Question:</strong> {question}
    </div>
    """, unsafe_allow_html=True)

    # Intent check
    state = intent_check_node({"question": question})
    if state.get("error"):
        # Intent rejection — coral/red, clearly different
        st.markdown(f"""
        <div style='background: rgba(30,60,120,0.2);
                    border-left: 3px solid rgba(80,140,220,0.7);
                    border-radius: 6px;
                    padding: 12px 16px;
                    color: rgba(140,190,255,0.9);
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
        <div style='background: rgba(42,32,64,0.5);
                    border: 0.5px solid rgba(255,255,255,0.07);
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
                f"<span style='display:inline-block; background:rgba(30,21,48,0.8); color:rgba(217,119,6,0.9); font-size:11px; padding:3px 10px; border-radius:20px; border:0.5px solid rgba(217,119,6,0.4); margin-right:6px; margin-top:6px;'>📄 {p}</span>"
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
        <div style='background: rgba(42,32,64,0.5);
                    border: 0.5px solid rgba(255,255,255,0.07);
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
                f"<span style='display:inline-block; background:rgba(30,21,48,0.8); color:rgba(5,150,105,0.9); font-size:11px; padding:3px 10px; border-radius:20px; border:0.5px solid rgba(5,150,105,0.4); margin-right:6px; margin-top:6px;'>📄 {p}</span>"
                for p in openai_papers
            )
            st.markdown(f"<div style='margin-top:8px;'>{chips}</div>", unsafe_allow_html=True)