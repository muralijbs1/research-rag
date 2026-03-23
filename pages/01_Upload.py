from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import tempfile

import streamlit as st

from src.ingestion.ingest_pipeline import SystemPaperError, run_ingestion

st.set_page_config(page_title="Upload Paper", page_icon="📤")
st.title("Upload Research Paper")

st.info(
    "Your paper will be saved to our database to answer your questions. "
    "You can delete it from the **Library** page.",
    icon="ℹ️",
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    paper_name = st.text_input(
        "Paper name (optional)",
        value=Path(uploaded_file.name).stem,
        help="Used as the identifier in the vector store. Defaults to the file name.",
    )

    if st.button("Ingest PDF", type="primary"):
        with st.spinner("Ingesting — parsing, chunking, embedding..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                file_bytes = uploaded_file.read()
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                result = run_ingestion(tmp_path, paper_name=paper_name.strip() or None)

                # Track this session's uploads for the Library page.
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
