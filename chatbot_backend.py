from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-120b"
)


# =====================================================
# State
# =====================================================

class Chatbot(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# =====================================================
# Node
# =====================================================

def query(state: Chatbot):

    messages = state["messages"]

    print("\nMessages received by query:")
    print(messages)

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# =====================================================
# Create Graph
# =====================================================

graph = StateGraph(Chatbot)


# =====================================================
# Checkpointer
# =====================================================

checkpointer = InMemorySaver()


# =====================================================
# Add Node
# =====================================================

graph.add_node(
    "query",
    query
)


# =====================================================
# Add Edges
# =====================================================

graph.add_edge(
    START,
    "query"
)

graph.add_edge(
    "query",
    END
)


# =====================================================
# Compile
# =====================================================

workflow = graph.compile(
    checkpointer=checkpointer
)