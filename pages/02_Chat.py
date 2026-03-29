from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st

from src.graph.intent_response import generate_conversation_title, route_message
from src.generation.llm_router import generate_stream
from src.generation.prompt_builder import build_prompt
from src.retrieval.multi_query_retriever import multi_query_retrieve
from src.retrieval.reranker import rerank

st.set_page_config(page_title="Ask a Question", page_icon="💬", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to right, #BEB6AA, #C8C0B4, #A89E92) !important;
    border-right: 18px solid #1A1020 !important;
    box-shadow: inset -20px 0 40px rgba(0,0,0,0.28) !important;
}
[data-testid="stSidebar"] * { color: #2E2820 !important; font-size: 15px !important; }

/* New conversation button — gold */
[data-testid="stSidebar"] .stButton > button {
    background: #F0C060 !important;
    color: #1A1208 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.15) !important;
}

/* Model selector — visible on cream */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #C8C0B4 !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    border-radius: 8px !important;
    color: #1A1A2E !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] p {
    color: #1A1A2E !important;
    font-weight: 500 !important;
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

/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(124,58,237,0.2) !important;
    border: 0.5px solid rgba(124,58,237,0.35) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 16px !important;
    margin-left: 15% !important;
    margin-bottom: 12px !important;
}

/* Assistant chat bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(42,32,64,0.5) !important;
    border: 0.5px solid rgba(255,255,255,0.06) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 12px 16px !important;
    margin-right: 15% !important;
    margin-bottom: 12px !important;
}

/* Chat input */
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

/* Citation chips */
.citation-chip {
    display: inline-block;
    background: rgba(30,21,48,0.8);
    color: rgba(167,139,250,0.85);
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 0.5px solid rgba(124,58,237,0.4);
    margin-right: 6px;
    margin-top: 6px;
}

[data-testid="stError"] {
    background: rgba(180,40,40,0.12) !important;
    border-left: 3px solid rgba(200,60,60,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        "<div style='font-size:16px; font-weight:600; color:#2E2820; margin-bottom:12px;'>⚙️ Settings</div>",
        unsafe_allow_html=True
    )
    model_choice = st.selectbox(
        "Model",
        options=["openai", "anthropic"],
        format_func=lambda x: "GPT-4o-mini" if x == "openai" else "Claude Haiku",
    )
    st.divider()
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_title = None
        st.rerun()

# --- Conversation title ---
if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = None

title_placeholder = st.empty()

if st.session_state.conversation_title:
    title_placeholder.markdown(
        f"<div style='font-size:26px; font-weight:600; color:#F0C060; margin-bottom:4px;'>💬 {st.session_state.conversation_title}</div>",
        unsafe_allow_html=True
    )
else:
    title_placeholder.markdown(
        "<div style='font-size:26px; font-weight:600; color:#F0C060; margin-bottom:4px;'>💬 Ask a Question</div>",
        unsafe_allow_html=True
    )

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("papers"):
            chips = "".join(
                f"<span class='citation-chip'>📄 {p}</span>"
                for p in msg["papers"]
            )
            st.markdown(
                f"<div style='margin-top:8px;'>{chips}</div>",
                unsafe_allow_html=True
            )

# --- Input ---
question = st.chat_input("Ask about your research papers...")

if question:
    if not st.session_state.conversation_title:
        st.session_state.conversation_title = generate_conversation_title(question)
        title_placeholder.markdown(
            f"<div style='font-size:26px; font-weight:600; color:#F0C060; margin-bottom:4px;'>💬 {st.session_state.conversation_title}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    routing = route_message(question, st.session_state.messages[:-1])
    print(f"[ROUTING] original='{question}' → route='{routing['route']}' → rewritten='{routing.get('rewritten_question')}'")

    with st.chat_message("assistant"):
        if routing["route"] == "chat":
            message = routing["message"] or "I'm not sure how to help with that. Try asking about AI/ML research!"
            st.markdown(message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": message,
                "papers": [],
            })

        else:
            try:
                chunks = multi_query_retrieve(routing["rewritten_question"], top_k=20)
                reranked = rerank(routing["rewritten_question"], chunks, top_n=7)
                prompt = build_prompt(question=routing["rewritten_question"], chunks=reranked, top_n=len(reranked))

                papers = list(dict.fromkeys(
                    c.get("paper_name", "")
                    for c in reranked
                ))
                papers = [p for p in papers if p]

                history_context = ""
                relevant_history = [
                    m for m in st.session_state.messages[:-1]
                    if m.get("papers")
                ]
                if relevant_history:
                    last_2 = relevant_history[-2:]
                    history_context = "\n\nPrevious conversation context:\n"
                    for m in last_2:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history_context += f"{role}: {m['content'][:300]}\n"

                if history_context:
                    prompt = prompt + history_context

                full_response = ""
                placeholder = st.empty()
                for token in generate_stream(prompt, model=model_choice):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                if papers:
                    chips = "".join(
                        f"<span class='citation-chip'>📄 {p}</span>"
                        for p in papers
                    )
                    st.markdown(
                        f"<div style='margin-top:8px;'>{chips}</div>",
                        unsafe_allow_html=True
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "papers": papers,
                })

            except Exception as e:
                st.error(f"Error: {e}")