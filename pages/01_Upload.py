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

/* Paper name input — navy editable signal */
[data-testid="stTextInput"] input {
    background: #EFF6FF !important;
    border: 1px solid #2563EB !important;
    color: #1E3A5F !important;
    border-radius: 8px !important;
    transition: border 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border: 1px solid #1E3A5F !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
}
[data-testid="stTextInput"] label {
    color: #2563EB !important;
    font-size: 13px !important;
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

/* File uploader */
[data-testid="stFileUploader"] {
    background: #F1F5F9 !important;
    border: 1px dashed #2563EB !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] > div {
    background: #F1F5F9 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #F1F5F9 !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #1E3A5F !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] button * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:36px; font-weight:600; color:#1E3A5F; background:#FEF3C7; display:inline-block; padding:5px 20px 5px 14px; border-radius:10px; margin-bottom:6px;'>📤 Upload Research Paper</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div style='background: #EFF6FF;
            border-left: 3px solid #2563EB;
            border-radius: 6px;
            padding: 12px 16px;
            color: #1E3A5F;
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