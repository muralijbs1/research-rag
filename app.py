from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

st.set_page_config(page_title="Research RAG Assistant", page_icon="📄", layout="wide")

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
[data-testid="stSidebar"] .stButton > button {
    background: #1E3A5F !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.05em !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stButton > button * { color: white !important; }
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
    border: 1px solid #C8B89A !important;
    color: #2C2416 !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
}
input, textarea {
    background: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid #C8B89A !important;
    border-radius: 0 !important;
    font-family: 'Georgia', serif !important;
    color: #2C2416 !important;
}
[data-testid="stInfo"] {
    background: #F5F0E8 !important;
    border-left: 3px solid #C8B89A !important;
    color: #2C2416 !important;
}
[data-testid="stWarning"] {
    background: #FFF7ED !important;
    border-left: 3px solid #D97706 !important;
}
[data-testid="stError"] {
    background: #FEF2F2 !important;
    border-left: 3px solid #DC2626 !important;
}
[data-testid="stSuccess"] {
    background: #F0FDF4 !important;
    border-left: 3px solid #16A34A !important;
}
hr { border-color: #C8B89A !important; }
[data-testid="stFileUploader"] {
    background: #FFFDF5 !important;
    border: 1px dashed #C8B89A !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.08) !important;
}
[data-testid="stSelectbox"] > div {
    background: #FFFDF5 !important;
    border: 1px solid #C8B89A !important;
    border-radius: 4px !important;
    color: #2C2416 !important;
    font-family: 'Georgia', serif !important;
}
/* Equal-height nav card columns */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > div {
    display: flex !important;
    flex-direction: column !important;
}
/* Page link arrow buttons — per-column colors */
[data-testid="stPageLink"] a {
    font-size: 18px !important;
    font-weight: 700 !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.03em !important;
    display: inline-block !important;
    text-align: center !important;
    padding: 4px 14px !important;
    border-radius: 3px !important;
    text-decoration: none !important;
    line-height: 1.6 !important;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.12) !important;
}
.stColumns > div:nth-child(1) [data-testid="stPageLink"] a { background: #DDD6FE !important; color: #4C1D95 !important; }
.stColumns > div:nth-child(2) [data-testid="stPageLink"] a { background: #FECDD3 !important; color: #881337 !important; }
.stColumns > div:nth-child(3) [data-testid="stPageLink"] a { background: #D1FAE5 !important; color: #065F46 !important; }
.stColumns > div:nth-child(4) [data-testid="stPageLink"] a { background: #FEF3C7 !important; color: #78350F !important; }
.stColumns > div:nth-child(5) [data-testid="stPageLink"] a { background: #CFFAFE !important; color: #164E63 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:52px; font-weight:600; color:#1E3A5F; background:#D1FAE5; display:inline-block; padding:6px 20px 6px 14px; border-radius:10px; margin-bottom:6px; line-height:1.2; box-shadow: inset 0 2px 6px rgba(0,0,0,0.12), inset 0 -1px 3px rgba(255,255,255,0.5);'>📄 Research RAG Assistant</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:18px; color:#64748B; margin-bottom:24px;'>"
    "Chat with AI/ML research papers — powered by RAG.</div>",
    unsafe_allow_html=True
)

pages = [
    ("pages/02_Chat.py",       "💬", "Chat",     "Ask questions across your paper corpus with streaming answers and citations.", "#DDD6FE"),
    ("pages/03_Compare.py",    "⚖️", "Compare",  "GPT-4o-mini vs Claude Haiku side by side.",                                  "#FECDD3"),
    ("pages/04_Evaluation.py", "📊", "Evaluate", "11 MLflow experiments — faithfulness, relevancy, recall, precision.",         "#D1FAE5"),
    ("pages/01_Upload.py",     "📤", "Upload",   "Add your own PDFs to the vector store instantly.",                            "#FEF3C7"),
    ("pages/05_Library.py",    "📚", "Library",  "Manage uploaded papers — view chunks and delete.",                            "#CFFAFE"),
]

cols = st.columns(5)
for col, (page, icon, label, desc, color) in zip(cols, pages):
    with col:
        st.markdown(f"""
        <div style='background: {color};
                    border: 1px solid #C8B89A;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
                    border-radius: 12px;
                    padding: 22px 16px;
                    min-height: 220px;
                    box-sizing: border-box;
                    overflow: visible;
                    margin-bottom: 60px;
                    display: flex;
                    flex-direction: column;'>
            <div style='font-size:26px; margin-bottom:10px;'>{icon}</div>
            <div style='font-size:15px; font-weight:500; color:#1E3A5F;'>{label}</div>
            <div style='font-size:12px; color:#64748B; margin-top:6px; line-height:1.5; flex:1;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Go to {label} →")