from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    max_results=3,
    time_range="day",
    search_depth="basic"
)


# =====================================================
# RAG: EMBEDDINGS MODEL
# =====================================================
# Groq doesn't serve embedding models, so we use a small
# local sentence-transformers model - no extra API key,
# runs on CPU fine for this use case.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# RAG: PER-THREAD VECTOR STORE REGISTRY
# =====================================================
# Scoped per conversation: each thread_id gets its own
# in-memory FAISS index of whatever PDF(s) were uploaded
# to that conversation. Cleared when the thread is reset
# by the frontend (see reset_chat() in app.py).

RAG_RETRIEVERS: dict[str, object] = {}


def ingest_pdf_for_thread(
    thread_id: str,
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """
    Load a PDF, split it into overlapping chunks, embed
    them, and store/replace the retriever for this thread.
    Call this from the frontend right after a file upload.
    """

    # -----------------------------------------
    # 1) LOAD
    # -----------------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"\n[rag] loaded {len(documents)} page(s) from {pdf_path}")

    # -----------------------------------------
    # 2) CHUNK
    # -----------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    print(f"[rag] split into {len(chunks)} chunk(s)")

    if not chunks:
        raise ValueError(
            "No extractable text found in this PDF "
            "(it may be a scanned/image-only document)."
        )

    # -----------------------------------------
    # 3) EMBED + STORE
    # -----------------------------------------
    # If this thread already has a store, ADD to it so
    # multiple uploads in the same conversation accumulate
    # instead of overwriting each other.

    existing = RAG_RETRIEVERS.get(thread_id)

    if existing is not None:

        existing_vectorstore = existing["vectorstore"]
        existing_vectorstore.add_documents(chunks)
        vectorstore = existing_vectorstore

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

    RAG_RETRIEVERS[thread_id] = {
        "vectorstore": vectorstore,
        "retriever": vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
    }

    print(f"[rag] vector store ready for thread {thread_id}")


def clear_rag_for_thread(thread_id: str):

    RAG_RETRIEVERS.pop(thread_id, None)


@tool
def rag_search(query: str, config: RunnableConfig) -> str:
    """
    Search the PDF document(s) the user has uploaded to
    this conversation. Use this whenever the question is
    about the content, facts, or details of an uploaded
    document - not for general knowledge or live data.
    """

    thread_id = config.get(
        "configurable", {}
    ).get("thread_id")

    entry = RAG_RETRIEVERS.get(thread_id)

    if entry is None:

        return (
            "No documents have been uploaded to this "
            "conversation yet. Ask the user to upload a "
            "PDF first."
        )

    retriever = entry["retriever"]

    results = retriever.invoke(query)

    print(f"\n[rag_search] query: {query}")
    print(f"[rag_search] retrieved {len(results)} chunk(s)")

    if not results:
        return "No relevant passages found in the uploaded document(s) for this query."

    formatted = "\n\n---\n\n".join(
        f"(page {d.metadata.get('page', '?')})\n{d.page_content}"
        for d in results
    )

    return formatted


# =====================================================
# ALL TOOLS
# =====================================================

tools = [
    wikipedia_search,
    web_search,
    rag_search,
]


# =====================================================
# TOKEN BUDGET
# =====================================================
# Groq's on-demand tier caps gpt-oss-120b at 8000 TPM.
# Long threads accumulate raw ToolMessage search results
# forever via the checkpointer, so we trim history before
# every LLM call rather than sending the whole thread.

MAX_HISTORY_TOKENS = 4000


def _approx_token_counter(msgs: list[BaseMessage]) -> int:
    # Cheap approximation (~4 chars/token) - good enough
    # for a trimming budget, doesn't need to be exact.
    total_chars = sum(
        len(str(m.content)) for m in msgs
    )
    return total_chars // 4


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:

    return trim_messages(
        messages,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=_approx_token_counter,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )


# =====================================================
# QUERY NODE
# =====================================================

def query(state: Chatbot):

    messages = trim_history(
        state["messages"]
    )

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

    if "rag" in selected_connectors:

        selected_tools.append(
            rag_search
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

        if "rag_search" in [t.name for t in selected_tools]:
            tool_capabilities.append(
                "- rag_search: searches PDF document(s) the "
                "user has uploaded to THIS conversation. Use "
                "it for any question about the content of "
                "an uploaded document. Do not use "
                "web_search or wikipedia_search for that - "
                "and do not use rag_search for general "
                "knowledge or live/current-events questions "
                "unrelated to the uploaded document."
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

    messages = trim_history(
        state["messages"]
    )

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