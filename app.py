from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

st.set_page_config(page_title="Research RAG Assistant", page_icon="📄", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #C9D8E8 !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * { color: #1E3A5F !important; font-size: 15px !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1E3A5F !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #F8FAFC !important;
}
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 2rem !important;
}
html, body, [data-testid="stAppViewContainer"] { color: #1E3A5F !important; }
h1, h2, h3 { color: #1E3A5F !important; font-weight: 500 !important; }
p, span, label { font-size: 15px !important; }
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E3A5F !important;
    border-radius: 8px !important;
}
input, textarea {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E3A5F !important;
    border-radius: 8px !important;
}
[data-testid="stInfo"] {
    background: #EFF6FF !important;
    border-left: 3px solid #2563EB !important;
    color: #1E3A5F !important;
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
hr { border-color: #E2E8F0 !important; }
[data-testid="stFileUploader"] {
    background: #F1F5F9 !important;
    border: 1px dashed #2563EB !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E3A5F !important;
}
/* Page link arrow buttons — per-column colors */
[data-testid="stPageLink"] a {
    font-size: 18px !important;
    font-weight: 700 !important;
    display: inline-block !important;
    text-align: center !important;
    padding: 4px 14px !important;
    border-radius: 8px !important;
    text-decoration: none !important;
    line-height: 1.6 !important;
}
.stColumns > div:nth-child(1) [data-testid="stPageLink"] a { background: #DDD6FE !important; color: #4C1D95 !important; }
.stColumns > div:nth-child(2) [data-testid="stPageLink"] a { background: #FECDD3 !important; color: #881337 !important; }
.stColumns > div:nth-child(3) [data-testid="stPageLink"] a { background: #D1FAE5 !important; color: #065F46 !important; }
.stColumns > div:nth-child(4) [data-testid="stPageLink"] a { background: #FEF3C7 !important; color: #78350F !important; }
.stColumns > div:nth-child(5) [data-testid="stPageLink"] a { background: #CFFAFE !important; color: #164E63 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:52px; font-weight:600; color:#1E3A5F; background:#D1FAE5; display:inline-block; padding:6px 20px 6px 14px; border-radius:10px; margin-bottom:6px; line-height:1.2;'>📄 Research RAG Assistant</div>",
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
                    border: 1px solid #E2E8F0;
                    border-radius: 12px;
                    padding: 22px 16px;
                    height: 180px;
                    box-sizing: border-box;
                    margin-bottom: 4px;
                    display: flex;
                    flex-direction: column;'>
            <div style='font-size:26px; margin-bottom:10px;'>{icon}</div>
            <div style='font-size:15px; font-weight:500; color:#1E3A5F;'>{label}</div>
            <div style='font-size:12px; color:#64748B; margin-top:6px; line-height:1.5; flex:1;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Go to {label} →")