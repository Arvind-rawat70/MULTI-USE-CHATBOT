import uuid
from chatbot_backend import workflow,retrive_all_threads
from langchain_core.messages import HumanMessage
import chatbot_backend
import streamlit as st
import os
import tempfile

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)


# =====================================================
# BACKEND
# =====================================================

workflow = chatbot_backend.workflow

retrive_all_threads = (
    chatbot_backend.retrive_all_threads
)


# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# CSS: pin the connector popover to the bottom bar,
# next to Streamlit's floating chat_input
# =====================================================

st.markdown(
    """
    <style>
    /* The popover trigger button block */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 18px;
        left: 22rem;        /* shift right of sidebar; tweak to taste */
        z-index: 999;
    }

    /* Push chat_input's left padding so it doesn't sit under the button */
    div[data-testid="stChatInput"] {
        padding-left: 3.5rem;
    }

    /* "Active: ..." caption under the popover — also pin it */
    #connector-active-caption {
        position: fixed;
        bottom: -6px;
        left: 22rem;
        z-index: 999;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def generate_thread_id():

    return str(
        uuid.uuid4()
    )


# =====================================================
# ADD THREAD
# =====================================================

def add_thread(thread_id):

    if thread_id not in st.session_state[
        "chat_threads"
    ]:

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


# =====================================================
# CONFIG
# =====================================================

def get_config(thread_id=None):

    if thread_id is None:

        thread_id = st.session_state[
            "thread_id"
        ]

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


# =====================================================
# LOAD CONVERSATION
# =====================================================

def load_conversation(thread_id):

    config = get_config(
        thread_id
    )

    state = workflow.get_state(
        config
    )

    return state.values.get(
        "messages",
        []
    )


# =====================================================
# GENERATE CHAT TITLE
# =====================================================

def generate_chat_title(thread_id):

    try:

        messages = load_conversation(
            thread_id
        )

        for message in messages:

            if isinstance(
                message,
                HumanMessage
            ):

                words = (
                    message.content
                    .strip()
                    .split()
                )

                if not words:

                    return "New Chat"

                if len(words) > 7:

                    return (
                        " ".join(words[:7])
                        + "..."
                    )

                return " ".join(
                    words
                )

    except Exception:

        pass

    return "New Chat"


# =====================================================
# RESET CHAT
# =====================================================

def reset_chat():

    thread_id = generate_thread_id()

    st.session_state[
        "thread_id"
    ] = thread_id

    st.session_state[
        "messages"
    ] = []

    # Reset connectors for new chat
    st.session_state[
        "selected_connectors"
    ] = []

    # Drop any RAG vector store tied to the OLD thread
    # (new chat = new thread_id = no uploaded doc context)
    st.session_state[
        "rag_uploaded_filenames"
    ] = []

    add_thread(
        thread_id
    )


# =====================================================
# SWITCH CHAT
# =====================================================

def switch_chat(thread_id):

    st.session_state[
        "thread_id"
    ] = thread_id

    messages = load_conversation(
        thread_id
    )

    temp_messages = []

    for message in messages:

        # -----------------------------------------
        # Human message
        # -----------------------------------------

        if isinstance(
            message,
            HumanMessage
        ):

            temp_messages.append({
                "role": "user",
                "content": message.content
            })

        # -----------------------------------------
        # AI message
        # -----------------------------------------

        elif isinstance(
            message,
            AIMessage
        ):

            # Don't display tool-call-only AI messages
            if message.content:

                temp_messages.append({
                    "role": "assistant",
                    "content": message.content
                })

        # -----------------------------------------
        # Don't display ToolMessage
        # -----------------------------------------

        elif isinstance(
            message,
            ToolMessage
        ):

            continue

    st.session_state[
        "messages"
    ] = temp_messages


# =====================================================
# INITIALIZE SESSION
# =====================================================

def initialize_session():

    if "thread_id" not in st.session_state:

        st.session_state[
            "thread_id"
        ] = generate_thread_id()


    if "messages" not in st.session_state:

        st.session_state[
            "messages"
        ] = []


    if "chat_threads" not in st.session_state:

        st.session_state[
            "chat_threads"
        ] = retrive_all_threads()


    if "selected_connectors" not in st.session_state:

        st.session_state[
            "selected_connectors"
        ] = []


    if "rag_uploaded_filenames" not in st.session_state:

        st.session_state[
            "rag_uploaded_filenames"
        ] = []


    add_thread(
        st.session_state[
            "thread_id"
        ]
    )


# =====================================================
# INITIALIZE
# =====================================================

initialize_session()


# =====================================================
# TITLE
# =====================================================

st.title(
    "🤖 LangGraph Chatbot"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "💬 LangGraph Chatbot"
)


# =====================================================
# NEW CHAT
# =====================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


st.sidebar.divider()

st.sidebar.subheader(
    "Recent Chats"
)


# =====================================================
# CHAT HISTORY
# =====================================================

for thread_id in st.session_state[
    "chat_threads"
]:

    chat_title = generate_chat_title(
        thread_id
    )

    is_current = (
        thread_id
        ==
        st.session_state["thread_id"]
    )

    if is_current:

        button_label = (
            f"🟢 {chat_title}"
        )

    else:

        button_label = (
            f"💬 {chat_title}"
        )


    if st.sidebar.button(
        button_label,
        key=f"chat_{thread_id}",
        use_container_width=True
    ):

        switch_chat(
            thread_id
        )

        st.rerun()


# =====================================================
# DISPLAY CURRENT CONVERSATION
# =====================================================

for message in st.session_state[
    "messages"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =====================================================
# CONNECTOR BUTTON  (now pinned to the bottom bar via CSS above)
# =====================================================
# Wikipedia and Tavily web search are always available to
# the model now - it decides on its own, per question,
# whether it needs to look something up and which of the
# two fits. There's nothing to toggle for them anymore.
#
# RAG (uploaded documents) is the one remaining connector,
# since it only makes sense to offer it once a PDF has
# actually been uploaded to this thread.

connector_button = st.popover(
    "➕"
)


with connector_button:

    st.markdown(
        "### 🔌 Documents"
    )

    st.caption(
        "📚 Wikipedia and 🌐 Web Search are always on — "
        "the assistant decides when it needs to look "
        "something up. Upload a PDF below to let it search "
        "your document too."
    )


    # ---------------------------------------------
    # RAG (uploaded PDFs)
    # ---------------------------------------------

    rag_enabled = st.checkbox(
        "📄 My Documents (RAG)",
        value=(
            "rag"
            in st.session_state[
                "selected_connectors"
            ]
        ),
        key="rag_checkbox"
    )

    uploaded_pdf = st.file_uploader(
        "Upload a PDF to search",
        type=["pdf"],
        key="rag_pdf_uploader"
    )

    if uploaded_pdf is not None:

        already_ingested = (
            uploaded_pdf.name
            in st.session_state["rag_uploaded_filenames"]
        )

        if not already_ingested:

            with st.spinner(
                f"Reading {uploaded_pdf.name}..."
            ):

                # Save to a temp file - PyPDFLoader needs
                # a real path, not an in-memory buffer.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_pdf.getvalue()
                    )

                    tmp_path = tmp_file.name

                try:

                    chatbot_backend.ingest_pdf_for_thread(
                        thread_id=st.session_state[
                            "thread_id"
                        ],
                        pdf_path=tmp_path,
                    )

                    st.session_state[
                        "rag_uploaded_filenames"
                    ].append(uploaded_pdf.name)

                    st.success(
                        f"✅ {uploaded_pdf.name} indexed"
                    )

                except Exception as e:

                    st.error(
                        f"Failed to read {uploaded_pdf.name}: {e}"
                    )

                finally:

                    os.unlink(tmp_path)

    if st.session_state["rag_uploaded_filenames"]:

        st.caption(
            "Indexed: "
            + ", ".join(
                st.session_state["rag_uploaded_filenames"]
            )
        )


# =====================================================
# UPDATE CONNECTORS
# =====================================================
# Only RAG is a real "connector" now - wikipedia/tavily
# are unconditionally offered by the backend, so there's
# nothing to append for them here.

selected_connectors = []


if rag_enabled:

    selected_connectors.append(
        "rag"
    )


st.session_state[
    "selected_connectors"
] = selected_connectors


# =====================================================
# SHOW ACTIVE CONNECTORS
# =====================================================

names = [
    "📚 Wikipedia",
    "🌐 Web Search",
]

if "rag" in selected_connectors:

    names.append(
        "📄 My Documents"
    )

st.markdown(
    f'<div id="connector-active-caption">Active: {" • ".join(names)}</div>',
    unsafe_allow_html=True
)


# =====================================================
# CHAT INPUT
# =====================================================

user_input = st.chat_input(
    "Type your message..."
)


# =====================================================
# HANDLE USER MESSAGE
# =====================================================

if user_input:

    # ---------------------------------------------
    # Display user message
    # ---------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ---------------------------------------------
    # Save in Streamlit state
    # ---------------------------------------------

    st.session_state[
        "messages"
    ].append({

        "role": "user",

        "content": user_input

    })


    # ---------------------------------------------
    # LangGraph config
    # ---------------------------------------------

    config = get_config()


    # ---------------------------------------------
    # Assistant response
    # ---------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        response_container = st.empty()

        full_response = ""


        # -----------------------------------------
        # STREAM
        # -----------------------------------------

        try:

            for message_chunk, metadata in workflow.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ],

                    "selected_connectors":
                        st.session_state[
                            "selected_connectors"
                        ]
                },

                config=config,

                stream_mode="messages"
            ):

                # -------------------------------------
                # Only collect text from AI messages.
                # ToolMessage chunks carry the raw tool
                # output (e.g. Tavily's JSON) and must
                # NOT be shown to the user directly.
                # -------------------------------------

                if (
                    isinstance(message_chunk, AIMessage)
                    and message_chunk.content
                ):

                    full_response += (
                        message_chunk.content
                    )

                    response_container.markdown(
                        full_response
                    )

        except Exception as e:

            full_response = (
                "⚠️ Something went wrong while "
                "generating a response (the model may "
                "have tried to use a tool that isn't "
                "enabled). Try uploading a relevant "
                "document above, or rephrase your "
                "question.\n\n"
                f"`{type(e).__name__}: {e}`"
            )

            response_container.markdown(
                full_response
            )


    # ---------------------------------------------
    # Save assistant response
    # ---------------------------------------------

    st.session_state[
        "messages"
    ].append({

        "role": "assistant",

        "content": full_response

    })


    # ---------------------------------------------
    # Refresh
    # ---------------------------------------------

    st.rerun()