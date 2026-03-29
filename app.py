from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

st.set_page_config(page_title="Research RAG Assistant", page_icon="📄", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to right, #BEB6AA, #C8C0B4, #A89E92) !important;
    border-right: 18px solid #1A1020 !important;
    box-shadow: inset -20px 0 40px rgba(0,0,0,0.28) !important;
}
[data-testid="stSidebar"] * { color: #2E2820 !important; font-size: 15px !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #7C3AED !important;
    color: rgba(255,255,255,0.92) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
}
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
    border: 0.5px solid rgba(124,58,237,0.6) !important;
    color: rgba(167,139,250,0.8) !important;
    border-radius: 8px !important;
}
input, textarea {
    background: rgba(20,14,36,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    color: rgba(200,195,225,0.85) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.3) !important;
}
[data-testid="stInfo"] {
    background: rgba(124,58,237,0.12) !important;
    border-left: 3px solid rgba(124,58,237,0.6) !important;
    color: rgba(200,195,225,0.8) !important;
}
[data-testid="stWarning"] {
    background: rgba(180,120,0,0.12) !important;
    border-left: 3px solid rgba(180,140,0,0.6) !important;
}
[data-testid="stError"] {
    background: rgba(180,40,40,0.12) !important;
    border-left: 3px solid rgba(200,60,60,0.6) !important;
}
[data-testid="stSuccess"] {
    background: rgba(30,140,80,0.12) !important;
    border-left: 3px solid rgba(40,160,90,0.6) !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stFileUploader"] {
    background: rgba(42,32,64,0.4) !important;
    border: 1px dashed rgba(124,58,237,0.4) !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div {
    background: rgba(20,14,36,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: rgba(200,195,225,0.85) !important;
}
[data-testid="stPageLink"] a {
    font-size: 15px !important;
    color: rgba(167,139,250,0.85) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:52px; font-weight:600; color:#F0C060; margin-bottom:6px; line-height:1.2;'>📄 Research RAG Assistant</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:18px; color:#E8844A; margin-bottom:24px;'>"
    "Chat with AI/ML research papers — powered by RAG.</div>",
    unsafe_allow_html=True
)

pages = [
    ("pages/02_Chat.py",       "💬", "Chat",     "Ask questions across your paper corpus with streaming answers and citations."),
    ("pages/03_Compare.py",    "⚖️", "Compare",  "GPT-4o-mini vs Claude Haiku side by side."),
    ("pages/04_Evaluation.py", "📊", "Evaluate", "11 MLflow experiments — faithfulness, relevancy, recall, precision."),
    ("pages/01_Upload.py",     "📤", "Upload",   "Add your own PDFs to the vector store instantly."),
    ("pages/05_Library.py",    "📚", "Library",  "Manage uploaded papers — view chunks and delete."),
]

cols = st.columns(5)
for col, (page, icon, label, desc) in zip(cols, pages):
    with col:
        st.markdown(f"""
        <div style='background: linear-gradient(to right, #C8C0B4, #BEB6AA);
                    border: 0.5px solid rgba(0,0,0,0.1);
                    border-radius: 12px;
                    padding: 22px 16px;
                    height: 180px;
                    box-sizing: border-box;
                    box-shadow: inset 0 1px 4px rgba(0,0,0,0.15), inset -4px 0 10px rgba(0,0,0,0.1);
                    margin-bottom: 4px;
                    display: flex;
                    flex-direction: column;'>
            <div style='font-size:26px; margin-bottom:10px;'>{icon}</div>
            <div style='font-size:15px; font-weight:500; color:#2E2820;'>{label}</div>
            <div style='font-size:12px; color:#5A5248; margin-top:6px; line-height:1.5; flex:1;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Go to {label}")