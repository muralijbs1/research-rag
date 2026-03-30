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
    background: #EDE8DF !important;
    border-right: 3px solid #C8B89A !important;
    box-shadow: 3px 0 15px rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Georgia', serif !important;
    color: #2C2416 !important;
    font-size: 15px !important;
}

/* New conversation button */
[data-testid="stSidebar"] .stButton > button {
    background: #1E3A5F !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.2), inset 0 -1px 3px rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] .stButton > button * { color: white !important; }
[data-testid="stSidebar"] .stButton > button p {
    color: white !important;
    font-size: 14px !important;
}

/* Model selector */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #FFFDF5 !important;
    border: 1px solid #C8B89A !important;
    border-radius: 3px !important;
    font-family: 'Georgia', serif !important;
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

/* Hide assistant avatar */
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #C8B89A !important;
    padding-left: 16px !important;
    background: transparent !important;
    border-bottom: 0.5px solid #E8E0D0 !important;
    margin-right: 10% !important;
    margin-bottom: 16px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) * {
    font-style: italic !important;
    color: #2C2416 !important;
    font-family: 'Georgia', serif !important;
}

/* Chat input */
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

/* Citation chips */
.citation-chip {
    display: inline-block;
    background: #F5F0E8;
    color: #1E3A5F;
    font-size: 11px;
    font-family: 'Georgia', serif;
    padding: 3px 10px;
    border-radius: 3px;
    border: 1px solid #C8B89A;
    margin-right: 6px;
    margin-top: 6px;
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

_USER_BUBBLE = (
    "background:#E8E4DC; border-radius:18px 18px 4px 18px; "
    "padding:12px 16px; color:#1E3A5F; font-family:'Georgia',serif; "
    "text-align:right; line-height:1.7;"
)

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        "<div style='font-size:16px; font-weight:600; color:#1E3A5F; margin-bottom:12px;'>⚙️ Settings</div>",
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
        f"<div style='font-size:26px; font-weight:700; color:#1E3A5F; font-family:Georgia,serif; background:transparent; border:none; padding:0; margin-bottom:16px;'>💬 {st.session_state.conversation_title}</div>",
        unsafe_allow_html=True
    )
else:
    title_placeholder.markdown(
        "<div style='font-size:26px; font-weight:700; color:#1E3A5F; font-family:Georgia,serif; background:transparent; border:none; padding:0; margin-bottom:16px;'>💬 Ask a Question</div>",
        unsafe_allow_html=True
    )

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        _, col = st.columns([1, 4])
        with col:
            st.markdown(
                f"<div style='{_USER_BUBBLE}'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if msg.get("papers"):
                with st.expander("📄 References"):
                    chips = "".join(
                        f"<span style='display:inline-block; background:#F5F0E8; color:#1E3A5F; font-size:12px; padding:4px 12px; border-radius:3px; border:1px solid #C8B89A; margin-right:6px; margin-top:6px; font-family:Georgia,serif;'>📄 {p}</span>"
                        for p in msg["papers"]
                    )
                    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

# --- Input ---
question = st.chat_input("Ask about your research papers...")

if question:
    if not st.session_state.conversation_title:
        try:
            st.session_state.conversation_title = generate_conversation_title(question)
        except Exception:
            st.session_state.conversation_title = question[:60]
        title_placeholder.markdown(
            f"<div style='font-size:26px; font-weight:700; color:#1E3A5F; font-family:Georgia,serif; background:transparent; border:none; padding:0; margin-bottom:16px;'>💬 {st.session_state.conversation_title}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "user", "content": question})
    _, col = st.columns([1, 4])
    with col:
        st.markdown(
            f"<div style='{_USER_BUBBLE}'>{question}</div>",
            unsafe_allow_html=True
        )

    try:
        routing = route_message(question, st.session_state.messages[:-1])
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    print(f"[ROUTING] original='{question}' → route='{routing['route']}' → rewritten='{routing.get('rewritten_question')}'")

    with st.chat_message("assistant", avatar="🤖"):
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
                    with st.expander("📄 References"):
                        chips = "".join(
                            f"<span style='display:inline-block; background:#F5F0E8; color:#1E3A5F; font-size:12px; padding:4px 12px; border-radius:3px; border:1px solid #C8B89A; margin-right:6px; margin-top:6px; font-family:Georgia,serif;'>📄 {p}</span>"
                            for p in papers
                        )
                        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "papers": papers,
                })

            except Exception as e:
                st.error(f"Error: {e}")
