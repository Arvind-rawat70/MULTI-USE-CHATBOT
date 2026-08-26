# 🤖 LangGraph Chatbot

A conversational AI chatbot built using **LangGraph, LangChain, Groq, and Streamlit**.

This project demonstrates how to build a **stateful conversational AI application** with LangGraph, including conversation threads, checkpoint-based state management, streaming responses, and a ChatGPT-style conversation history interface.

---

## 📸 Screenshots

### Chat Interface

<img width="1077" height="871" alt="Chatbot Interface" src="https://github.com/user-attachments/assets/c21fb816-be46-4bca-8421-bf2a04cf9953" />

### LangGraph Workflow

<img width="135" height="256" alt="LangGraph Workflow" src="https://github.com/user-attachments/assets/1c06a605-25ed-4d26-8000-c761c83dc4fd" />

---

## 🚀 Features

- 🤖 Conversational AI chatbot
- ⚡ Streaming AI responses
- 💬 ChatGPT-style chat interface
- 🗂️ Multiple conversation threads
- ➕ Create new conversations
- 🔀 Switch between previous conversations
- 🧵 Thread-based conversation state
- 💾 LangGraph checkpointing
- 🧠 Stateful conversations
- 📋 Chat history in the sidebar
- 🔐 Secure API key management using `.env`
- 🎨 Streamlit frontend
- ⚙️ LangGraph backend workflow
- 🔗 LangChain message integration
- 🚀 Groq LLM inference

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Workflow and state management |
| LangChain | LLM and message integration |
| Groq | LLM inference |
| Streamlit | Web interface |
| TypedDict | State definition |
| `Annotated` | State update configuration |
| `add_messages` | Conversation message management |
| `InMemorySaver` | In-memory checkpointing |
| python-dotenv | Environment variable management |

---

## 📁 Project Structure

```text
MULTI-USE-CHATBOT/
│
├── app.py                  # Streamlit frontend
├── chatbot_backend.py      # LangGraph workflow and LLM logic
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignored files
│
└── .env                    # Local API key - not committed



🧠 Application Architecture

The application is divided into two main parts:


                    USER
                      │
                      ↓
            ┌──────────────────┐
            │   Streamlit UI   │
            │      app.py      │
            └────────┬─────────┘
                     │
                     ↓
              HumanMessage
                     │
                     ↓
            ┌──────────────────┐
            │    LangGraph     │
            │    Workflow      │
            └────────┬─────────┘
                     │
                     ↓
                Query Node
                     │
                     ↓
                 Groq LLM
                     │
                     ↓
                 AIMessage
                     │
                     ↓
              Stream Response
                     │
                     ↓
            ┌──────────────────┐
            │   Streamlit UI   │
            └──────────────────┘


🔄 LangGraph Workflow

The chatbot uses a simple LangGraph workflow

START
  │
  ↓
Query Node
  │
  ↓
Groq LLM
  │
  ↓
END


🧵 Conversation Threads

Every conversation is assigned a unique thread_id.

For example:


Thread A
│
├── User message
├── AI response
├── User message
└── AI response


Thread B
│
├── User message
└── AI response


💾 Checkpointing

The current project uses:
InMemorySaver()


                 Checkpointer
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Thread A    Thread B    Thread C
          │           │           │
          ↓           ↓           ↓
       Messages    Messages    Messages



💬 Chat History

The Streamlit sidebar provides a ChatGPT-style conversation history.

Instead of showing raw thread IDs such as:

8f31c9a2
72bd912a

the UI displays a short title generated from the conversation:

How can I implement streaming...
Explain LangGraph state management
Build a RAG chatbot...

Users can click a previous conversation and continue that specific thread.


🔗 Frontend and Backend Flow

When a user sends a message:

User enters message
        ↓
Streamlit receives input
        ↓
HumanMessage is created
        ↓
thread_id is attached
        ↓
LangGraph workflow starts
        ↓
Query node receives state
        ↓
Messages are sent to Groq
        ↓
Groq generates response
        ↓
Response is streamed
        ↓
Streamlit displays response
        ↓
Conversation state is updated



🔮 Future Improvements

Planned improvements include:

 Persistent database checkpointing
 PostgreSQL-based conversation storage
 Better automatic conversation titles
 Delete conversations
 Rename conversations
 Search conversations
 User authentication
 Tool calling
 Web search
 RAG pipeline
 PDF document Q&A
 Vector database integration
 Multiple LLM providers
 Model selection
 Token usage tracking
 Better error handling
 Retry mechanisms
 FastAPI backend
 Dockerization
 Production deployment
