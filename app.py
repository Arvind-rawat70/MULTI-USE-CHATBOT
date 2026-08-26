import streamlit as st
from langchain_core.messages import HumanMessage
import uuid

from chatbot_backend import workflow


# =========================================================
# Utility Functions
# =========================================================

def generate_thread_id():
    """
    Generate a unique ID for every conversation.
    """
    return str(uuid.uuid4())


def add_thread(thread_id):
    """
    Add a thread to the conversation list
    if it doesn't already exist.
    """

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def get_config(thread_id=None):
    """
    Create LangGraph configuration.

    If thread_id is not provided,
    use the currently active thread.
    """

    if thread_id is None:
        thread_id = st.session_state["thread_id"]

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def load_conversation(thread_id):
    """
    Load conversation state from LangGraph
    using the thread_id.
    """

    config = get_config(thread_id)

    state = workflow.get_state(config)

    return state.values.get("messages", [])


def generate_chat_title(thread_id):
    """
    Generate a short title from the first
    HumanMessage of the conversation.

    Maximum = 7 words.
    """

    try:

        messages = load_conversation(thread_id)

        for message in messages:

            if isinstance(message, HumanMessage):

                words = message.content.strip().split()

                if not words:
                    return "New Chat"

                if len(words) > 7:
                    return " ".join(words[:7]) + "..."

                return " ".join(words)

    except Exception:
        pass

    return "New Chat"


def reset_chat():
    """
    Create a new conversation.
    """

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    st.session_state["messages"] = []

    add_thread(thread_id)


def switch_chat(thread_id):
    """
    Switch from current conversation
    to another conversation.
    """

    st.session_state["thread_id"] = thread_id

    messages = load_conversation(thread_id)

    temp_messages = []

    for message in messages:

        if isinstance(message, HumanMessage):
            role = "user"

        else:
            role = "assistant"

        temp_messages.append({
            "role": role,
            "content": message.content
        })

    st.session_state["messages"] = temp_messages


def initialize_session():
    """
    Initialize Streamlit session state.
    """

    if "thread_id" not in st.session_state:

        st.session_state["thread_id"] = generate_thread_id()


    if "messages" not in st.session_state:

        st.session_state["messages"] = []


    if "chat_threads" not in st.session_state:

        st.session_state["chat_threads"] = []


    # Add current thread
    add_thread(
        st.session_state["thread_id"]
    )


# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# Initialize Session
# =========================================================

initialize_session()


# =========================================================
# Main Title
# =========================================================

st.title("🤖 LangGraph Chatbot")


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("💬 LangGraph Chatbot")


# ---------------------------------------------------------
# New Chat Button
# ---------------------------------------------------------

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


st.sidebar.divider()

st.sidebar.subheader("Recent Chats")


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for thread_id in st.session_state["chat_threads"]:

    # Generate title from first user message
    chat_title = generate_chat_title(thread_id)


    # Highlight current chat
    is_current = (
        thread_id == st.session_state["thread_id"]
    )


    if is_current:

        button_label = f"🟢 {chat_title}"

    else:

        button_label = f"💬 {chat_title}"


    # Chat history button
    if st.sidebar.button(
        button_label,
        key=f"chat_{thread_id}",
        use_container_width=True
    ):

        switch_chat(thread_id)

        st.rerun()


# =========================================================
# Display Current Conversation
# =========================================================

for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "Type your message..."
)


# =========================================================
# Handle User Message
# =========================================================

if user_input:

    # -----------------------------------------------------
    # Display User Message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    # -----------------------------------------------------
    # Save User Message in Streamlit State
    # -----------------------------------------------------

    st.session_state["messages"].append({

        "role": "user",

        "content": user_input

    })


    # -----------------------------------------------------
    # LangGraph Configuration
    # -----------------------------------------------------

    config = get_config()


    # -----------------------------------------------------
    # Stream Assistant Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        response_container = st.empty()

        full_response = ""


        # ---------------------------------------------
        # Stream response from LangGraph
        # ---------------------------------------------

        for message_chunk, metadata in workflow.stream(

            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },

            config=config,

            stream_mode="messages"
        ):

            # -----------------------------------------
            # Check if chunk contains content
            # -----------------------------------------

            if message_chunk.content:

                full_response += (
                    message_chunk.content
                )


                # -------------------------------------
                # Update UI
                # -------------------------------------

                response_container.markdown(
                    full_response
                )


    # -----------------------------------------------------
    # Save Assistant Response
    # -----------------------------------------------------

    st.session_state["messages"].append({

        "role": "assistant",

        "content": full_response

    })


    # -----------------------------------------------------
    # Refresh UI
    # -----------------------------------------------------

    st.rerun()