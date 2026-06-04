import streamlit as st
from langchain_backend import (
    chatbot,
    llm,
    retrieve_all_threads,
    save_chat_title,
    get_all_chat_titles
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(user_message):

    try:

        prompt = f"""
Generate a short conversation title.

Rules:
- Maximum 4 words
- Professional
- No quotes
- No punctuation at the end
- Return only the title

User message:
{user_message}
"""

        title = llm.invoke(prompt).content.strip()

        if title:
            return title

    except Exception:
        pass

    fallback = user_message[:35]

    if len(user_message) > 35:
        fallback += "..."

    return fallback


def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# ============================================================================
# SESSION STATE
# ============================================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:

    # Load all existing thread IDs from SQLite
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:

    # Load all saved titles from SQLite
    st.session_state["thread_titles"] = get_all_chat_titles()

add_thread(st.session_state["thread_id"])


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("SynapticOS")

if st.sidebar.button("➕ New Chat"):

    reset_chat()
    st.rerun()

st.sidebar.header("Chat History")

for thread_id in reversed(st.session_state["chat_threads"]):

    title = st.session_state["thread_titles"].get(
        thread_id,
        "Untitled Chat"
    )

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            role = (
                "user"
                if isinstance(msg, HumanMessage)
                else "assistant"
            )

            temp_messages.append(
                {
                    "role": role,
                    "content": msg.content
                }
            )

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ============================================================================
# MAIN CHAT WINDOW
# ============================================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Type your message..."
)

if user_input:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    current_thread = st.session_state["thread_id"]

    # ============================================================
    # GENERATE TITLE ONLY ON FIRST MESSAGE
    # ============================================================

    if current_thread not in st.session_state["thread_titles"]:

        generated_title = generate_chat_title(
            user_input
        )

        st.session_state["thread_titles"][
            current_thread
        ] = generated_title

        # Persist title in SQLite
        save_chat_title(
            current_thread,
            generated_title
        )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": current_thread
        }
    }

    with st.chat_message("assistant"):

        def ai_only_stream():

            try:

                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },
                    config=CONFIG,
                    stream_mode="messages"
                ):

                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):
                        yield message_chunk.content

            except Exception:

                yield (
                    "⚠️ Model temporarily unavailable. "
                    "Please try again."
                )

        ai_message = st.write_stream(
            ai_only_stream()
        )

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )