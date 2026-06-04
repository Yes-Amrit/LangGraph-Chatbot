from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
# ===== CHANGE START =====
# Added imports for retry handling and Gemini server errors
import time
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
    DeadlineExceeded
)
# ===== CHANGE END =====

import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =====================================================================================
# LLM CONFIGURATION
# =====================================================================================

# ===== CHANGE START =====
# Added temperature parameter.
# You can remove it if you want deterministic responses.
# ===== CHANGE END =====
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# =====================================================================================
# STATE DEFINITION
# =====================================================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =====================================================================================
# RETRY FUNCTION
# =====================================================================================

# ===== CHANGE START =====
# New function added.
#
# Purpose:
# Gemini sometimes throws:
# - 503 ServiceUnavailable
# - 429 ResourceExhausted
# - 500 InternalServerError
# - DeadlineExceeded
#
# Instead of crashing immediately,
# retry 5 times with exponential backoff.
# =====================================================================================

def invoke_with_retry(messages):

    max_retries = 5

    for attempt in range(max_retries):

        try:
            return llm.invoke(messages)

        except (
            ResourceExhausted,
            ServiceUnavailable,
            InternalServerError,
            DeadlineExceeded
        ) as e:

            # If all retries exhausted
            if attempt == max_retries - 1:
                raise e

            # Exponential backoff:
            # 1s → 2s → 4s → 8s → 16s
            wait_time = 2 ** attempt

            print(
                f"[Retry {attempt + 1}/{max_retries}] "
                f"Gemini overloaded. Waiting {wait_time}s..."
            )

            time.sleep(wait_time)

# ===== CHANGE END =====

# ============================================================
# CHAT TITLE STORAGE
# ============================================================

def create_title_table():

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    """)

    conn.commit()


def save_chat_title(
    thread_id: str,
    title: str
):

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO chat_titles
        (thread_id, title)
        VALUES (?, ?)
        """,
        (thread_id, title)
    )

    conn.commit()


def get_chat_title(
    thread_id: str
):

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT title
        FROM chat_titles
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    row = cursor.fetchone()

    if row:
        return row[0]

    return None


def get_all_chat_titles():

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT thread_id, title
        FROM chat_titles
        """
    )

    rows = cursor.fetchall()

    return {
        thread_id: title
        for thread_id, title in rows
    }
# =====================================================================================
# CHAT NODE
# =====================================================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    # ===== CHANGE START =====
    # Wrapped LLM call in try/except.
    #
    # Earlier:
    # response = llm.invoke(messages)
    #
    # Now:
    # response = invoke_with_retry(messages)
    #
    # Benefits:
    # - Handles temporary Gemini outages
    # - Prevents graph crash
    # - Better user experience
    # =================================================================================

    try:

        response = invoke_with_retry(messages)

        return {
            "messages": [response]
        }

    except Exception as e:

        print("LLM ERROR:", e)

        # Return graceful error response
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content":
                    "⚠️ Gemini is currently unavailable. Please try again in a few moments."
                }
            ]
        }

    # ===== CHANGE END =====


# =====================================================================================
# CHECKPOINTER
# =====================================================================================

# Current memory type:
# Stores chats only while application is running.
# Data disappears when Streamlit restarts.

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
create_title_table()
checkpointer = SqliteSaver(conn = conn)

# =====================================================================================
# GRAPH CREATION
# =====================================================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_edge(
    "chat_node",
    END
)

# =====================================================================================
# COMPILE GRAPH
# =====================================================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)