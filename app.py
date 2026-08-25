import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import workflow


st.title("🤖 Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []


# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input
user_input = st.chat_input("Type here...")


if user_input:

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # Thread configuration
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }


    # Run LangGraph
    response = workflow.invoke(
        {
            "message": [
                HumanMessage(content=user_input)
            ]
        },
        config=config
    )


    # Get AI response
    ai_response = response["message"][-1].content


    # Show AI response
    with st.chat_message("assistant"):
        st.write(ai_response)


    # Save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })