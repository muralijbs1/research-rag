from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

st.set_page_config(page_title="Research RAG Assistant", page_icon="📄")

st.title("Research Paper RAG Assistant")
st.write("Upload research papers, ask questions, and evaluate your pipeline.")