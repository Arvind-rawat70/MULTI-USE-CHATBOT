from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langchain_groq import ChatGroq

from dotenv import load_dotenv

import os
import sqlite3
from datetime import date


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Sanity check: fail loudly instead of silently misbehaving
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment / .env")

if not os.getenv("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY not set — Tavily search will fail at call time.")


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-120b"
)


# =====================================================
# STATE
# =====================================================

class Chatbot(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    selected_connectors: list[str]


# =====================================================
# WIKIPEDIA
# =====================================================

api_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=4000
)


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual information about
    people, places, historical events, science,
    technology, organizations and other general topics.
    """

    result = api_wrapper.run(query)

    print("\n[wikipedia_search] query:", query)
    print("[wikipedia_search] result (first 300 chars):", result[:300])

    return result


# =====================================================
# TAVILY WEB SEARCH
# =====================================================

web_search = TavilySearch(
    max_results=5,
    time_range="day",       # bias toward pages indexed/updated in the last day
    search_depth="advanced"
)


# =====================================================
# ALL TOOLS
# =====================================================

tools = [
    wikipedia_search,
    web_search
]


# =====================================================
# QUERY NODE
# =====================================================

def query(state: Chatbot):

    messages = state["messages"]

    selected_connectors = state.get(
        "selected_connectors",
        []
    )

    print("\n================================")
    print("Selected connectors:")
    print(selected_connectors)
    print("================================")

    # ---------------------------------------------
    # Select tools according to frontend
    # ---------------------------------------------

    selected_tools = []

    if "wikipedia" in selected_connectors:

        selected_tools.append(
            wikipedia_search
        )

    if "tavily" in selected_connectors:

        selected_tools.append(
            web_search
        )

    # ---------------------------------------------
    # If connectors are selected
    # bind them to LLM
    # ---------------------------------------------

    if selected_tools:

        llm_to_use = llm.bind_tools(
            selected_tools
        )

        # -----------------------------------------
        # gpt-oss-120b will happily hedge in plain
        # text ("I can't access live data...")
        # instead of calling a bound tool unless
        # explicitly told the tool exists to be used.
        # Prepend a system instruction every turn.
        # -----------------------------------------

        tool_names = ", ".join(
            t.name for t in selected_tools
        )

        tool_capabilities = []

        if "wikipedia_search" in [t.name for t in selected_tools]:
            tool_capabilities.append(
                "- wikipedia_search: static encyclopedic "
                "facts (history, biographies, definitions, "
                "science, places). Cannot get live/current "
                "data."
            )

        if "web_search" in [t.name for t in selected_tools]:
            tool_capabilities.append(
                "- web_search: a LIVE web search engine. "
                "It CAN and SHOULD be used for current "
                "prices (crypto, stocks, currency exchange "
                "rates), weather, news, sports scores, and "
                "any other real-time or recent information. "
                "Include today's date in your search query "
                "(e.g. 'Bitcoin price today "
                f"{date.today().isoformat()}') "
                "to get the freshest result, and prefer the "
                "most recently published source among the "
                "results."
            )

        system_instruction = SystemMessage(
            content=(
                "You have access to ONLY these tools:\n"
                + "\n".join(tool_capabilities) +
                "\n\nDo not call any tool that is not in "
                "this list. If a question needs live/"
                "current information and a capable tool "
                "is listed above, you MUST call it - do "
                "not answer from memory and do not claim "
                "you lack real-time access when such a "
                "tool is available. Only decline to call "
                "a tool if none of the tools listed above "
                "are actually capable of answering the "
                "question (for example, only wikipedia_"
                "search is available and the question "
                "needs a live price) - in that case, say "
                "so plainly instead of guessing."
            )
        )

        invoke_messages = [system_instruction] + messages

    else:

        llm_to_use = llm
        invoke_messages = messages

    # ---------------------------------------------
    # LLM decides whether tool is needed
    # ---------------------------------------------

    response = llm_to_use.invoke(
        invoke_messages
    )

    print("\nLLM response:")
    print(response)
    print("tool_calls:", getattr(response, "tool_calls", None))

    return {
        "messages": [response]
    }


# =====================================================
# CONDITION
# =====================================================

def condition(
    state: Chatbot
) -> Literal["tools", "chatbot"]:

    last_message = state["messages"][-1]

    # LLM requested a tool
    if last_message.tool_calls:

        return "tools"

    # No tool required -> this label now routes straight to END
    return "chatbot"


# =====================================================
# CHATBOT NODE
# (only reached AFTER tools have run, to synthesize
# a final answer from the tool results)
# =====================================================

def chatbot(state: Chatbot):

    messages = state["messages"]

    response = llm.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# =====================================================
# TOOL NODE
# =====================================================

tool_node = ToolNode(
    tools
)


# =====================================================
# CREATE GRAPH
# =====================================================

graph = StateGraph(
    Chatbot
)


# =====================================================
# ADD NODES
# =====================================================

graph.add_node(
    "query",
    query
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "chatbot",
    chatbot
)


# =====================================================
# EDGES
# =====================================================

graph.add_edge(
    START,
    "query"
)


graph.add_conditional_edges(
    "query",
    condition,
    {
        "tools": "tools",
        "chatbot": END          # <-- FIX: no tool needed -> query's answer IS final,
                                  #     don't regenerate it via the chatbot node
    }
)


graph.add_edge(
    "tools",
    "chatbot"
)


graph.add_edge(
    "chatbot",
    END
)


# =====================================================
# SQLITE CHECKPOINTER
# =====================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn
)


# =====================================================
# COMPILE
# =====================================================

workflow = graph.compile(
    checkpointer=checkpointer
)


# =====================================================
# RETRIEVE ALL THREADS
# =====================================================

def retrive_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config[
            "configurable"
        ].get(
            "thread_id"
        )

        if thread_id:

            all_threads.add(
                thread_id
            )

    return list(all_threads)