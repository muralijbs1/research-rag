from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import tempfile
import streamlit as st
from src.ingestion.ingest_pipeline import SystemPaperError, run_ingestion

st.set_page_config(page_title="Upload Paper", page_icon="📤", layout="wide")

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

/* Paper name input — green editable signal */
[data-testid="stTextInput"] input {
    background: rgba(20,140,60,0.15) !important;
    border: 1px solid rgba(40,180,80,0.6) !important;
    color: rgba(200,195,225,0.9) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.3) !important;
    transition: border 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border: 1px solid rgba(60,220,100,0.9) !important;
    box-shadow: 0 0 0 2px rgba(40,180,80,0.2), inset 0 1px 4px rgba(0,0,0,0.3) !important;
}
[data-testid="stTextInput"] label {
    color: rgba(80,200,120,0.9) !important;
    font-size: 13px !important;
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

/* File uploader — cream */
[data-testid="stFileUploader"] {
    background: linear-gradient(to right, #C8C0B4, #BEB6AA) !important;
    border: 1px dashed rgba(124,58,237,0.5) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] > div {
    background: linear-gradient(to right, #C8C0B4, #BEB6AA) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(to right, #C8C0B4, #BEB6AA) !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #2E2820 !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: rgba(124,58,237,0.85) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] button * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#F0C060; margin-bottom:6px;'>📤 Upload Research Paper</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div style='background: rgba(160,90,0,0.18);
            border-left: 3px solid rgba(220,140,0,0.8);
            border-radius: 6px;
            padding: 12px 16px;
            color: rgba(255,200,80,0.95);
            font-size: 15px;
            margin-bottom: 16px;'>
    ℹ️ Your paper will be saved to our database to answer your questions.
    You can delete it from the <strong>Library</strong> page.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    paper_name = st.text_input(
        "Paper name (optional)",
        value=Path(uploaded_file.name).stem,
        help="Used as the identifier in the vector store. Defaults to the file name.",
    )
    st.caption("✏️ You can edit the paper name before ingesting.")

    if st.button("Ingest PDF", type="primary"):
        with st.spinner("Ingesting — parsing, chunking, embedding..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                file_bytes = uploaded_file.read()
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                result = run_ingestion(tmp_path, paper_name=paper_name.strip() or None)

                if "uploaded_papers" not in st.session_state:
                    st.session_state.uploaded_papers = []
                name = paper_name.strip() or Path(uploaded_file.name).stem
                if name not in st.session_state.uploaded_papers:
                    st.session_state.uploaded_papers.append(name)

                st.success(
                    f"Done! **{result['num_chunks']} chunks** ingested for *{name}*."
                )

            except SystemPaperError as e:
                st.warning(str(e), icon="📚")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)