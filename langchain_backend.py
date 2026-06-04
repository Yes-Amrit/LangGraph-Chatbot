from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

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

checkpointer = InMemorySaver()

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