from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import streamlit as st

from src.graph.intent_response import generate_conversation_title, route_message
from src.generation.llm_router import generate_stream
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank

st.set_page_config(page_title="Ask a Question", page_icon="💬")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        options=["openai", "anthropic"],
        format_func=lambda x: "GPT-4o-mini" if x == "openai" else "Claude Haiku",
    )
    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_title = None
        st.rerun()

# --- Conversation title placeholder ---
if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = None

title_placeholder = st.empty()
title_placeholder.title(st.session_state.conversation_title or "Ask a Question")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("papers"):
            st.caption("📄 Sources: " + " · ".join(msg["papers"]))

# --- Input ---
question = st.chat_input("Ask about your research papers...")

if question:
    # Generate title from first message
    if not st.session_state.conversation_title:
        st.session_state.conversation_title = generate_conversation_title(question)
        title_placeholder.title(st.session_state.conversation_title)

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Route through Groq
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
                chunks = retrieve(routing["rewritten_question"], top_k=20)
                reranked = rerank(routing["rewritten_question"], chunks, top_n=5)
                prompt = build_prompt(question=routing["rewritten_question"], chunks=reranked, top_n=len(reranked))

                # Get unique paper names for citation
                papers = list(dict.fromkeys(
                    c.get("paper_name", "")
                    for c in reranked
                ))
                papers = [p for p in papers if p]

                # Build prompt with relevant conversation history
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

                # Stream the response
                full_response = ""
                placeholder = st.empty()
                for token in generate_stream(prompt, model=model_choice):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                # Show citations
                if papers:
                    st.caption("📄 Sources: " + " · ".join(papers))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "papers": papers,
                })

            except Exception as e:
                st.error(f"Error: {e}")