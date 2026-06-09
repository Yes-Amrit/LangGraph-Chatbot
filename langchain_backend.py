from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# --- Tool Imports ---
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

import sqlite3
import time
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
    DeadlineExceeded
)
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# =====================================================================================
# 1. LLM CONFIGURATION
# =====================================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)
# =====================================================================================
# 2. TOOLS DEFINITION
# =====================================================================================
search_tool = DuckDuckGoSearchRun(region="us-en")
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]

# Bind tools to Gemini
llm_with_tools = llm.bind_tools(tools)
# =====================================================================================
# 3. STATE DEFINITION
# =====================================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
# =====================================================================================
# 4. RETRY FUNCTION
# =====================================================================================

def invoke_with_retry(messages):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # IMPORTANT: Using llm_with_tools here so Gemini can use the tools
            return llm_with_tools.invoke(messages)
        except (
            ResourceExhausted,
            ServiceUnavailable,
            InternalServerError,
            DeadlineExceeded
        ) as e:
            # If all retries exhausted
            if attempt == max_retries - 1:
                raise e

            # Exponential backoff: 1s → 2s → 4s → 8s → 16s
            wait_time = 2 ** attempt

            print(
                f"[Retry {attempt + 1}/{max_retries}] "
                f"Gemini overloaded. Waiting {wait_time}s..."
            )

            time.sleep(wait_time)

# ============================================================
# 5. DB & CHAT TITLE STORAGE
# ============================================================
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
def create_title_table():
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    """)
    conn.commit()
def save_chat_title(thread_id: str, title: str):
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
def get_chat_title(thread_id: str):
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
    return {thread_id: title for thread_id, title in rows}
# Initialize tables & Checkpointer
create_title_table()
checkpointer = SqliteSaver(conn=conn)

# =====================================================================================
# 6. NODES
# =====================================================================================

def chat_node(state: ChatState):
    messages = state["messages"]
    try:
        response = invoke_with_retry(messages)
        return {"messages": [response]}
    except Exception as e:
        print("LLM ERROR:", e)
        # Return graceful error response
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "⚠️ Gemini is currently unavailable. Please try again in a few moments."
                }
            ]
        }
# Define the Tool Node
tool_node = ToolNode(tools)

# =====================================================================================
# 7. GRAPH CREATION
# =====================================================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")

# Routing: If the LLM returns a tool_call, go to tools. Otherwise, END.
graph.add_conditional_edges("chat_node", tools_condition)
# Routing: Once the tools finish running, return their output to the chat_node
graph.add_edge("tools", "chat_node")

# =====================================================================================
# 8. COMPILE GRAPH & HELPERS
# =====================================================================================
chatbot = graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)