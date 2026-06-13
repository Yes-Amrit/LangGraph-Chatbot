import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    get_all_chat_titles,
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ============================================================================
# SESSION STATE
# ============================================================================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    try:
        st.session_state["thread_titles"] = get_all_chat_titles()
    except Exception:
        st.session_state["thread_titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("SynapticOS")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# --- PDF Upload Section ---
st.sidebar.subheader("Document for this chat")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

# --- Chat History Section ---
st.sidebar.header("Chat History")

for thread_id in reversed(st.session_state["chat_threads"]):
    title = st.session_state["thread_titles"].get(thread_id, str(thread_id))

    if st.sidebar.button(title, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []

        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages
        st.session_state["ingested_docs"].setdefault(str(thread_id), {})
        st.rerun()

# ============================================================================
# MAIN CHAT WINDOW
# ============================================================================
st.title("Multi Utility RAG Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    current_thread = st.session_state["thread_id"]

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": current_thread},
        "metadata": {"thread_id": current_thread},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_box = None
        message_placeholder = st.empty()
        full_response = ""

        try:
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # --- Tool Tracking Block ---
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_box is None:
                        status_box = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_box.update(label=f"🔧 Using `{tool_name}` …", state="running")

                # --- Text Rendering Block ---
                elif isinstance(chunk, AIMessage):
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            full_response += chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and "text" in part:
                                    full_response += part["text"]
                                elif isinstance(part, str):
                                    full_response += part

                        message_placeholder.markdown(full_response + " ▌")

            if full_response.strip():
                message_placeholder.markdown(full_response)
            else:
                message_placeholder.markdown("✅ Task completed.")

            if status_box is not None:
                status_box.update(label="✅ Tool finished", state="complete", expanded=False)

        except Exception as e:
            message_placeholder.markdown(f"⚠️ An error occurred: {str(e)}")

    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response if full_response.strip() else "✅ Task completed."}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )