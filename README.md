# 🤖 LangGraph Chatbot

A conversational AI chatbot built using **LangGraph, LangChain, Groq, and Streamlit**.

This project demonstrates how to build a **stateful, tool-using conversational AI application** with LangGraph — including conversation threads, persistent checkpoint-based state management, streaming responses, pluggable search connectors (Wikipedia + Tavily), and a ChatGPT-style conversation history interface.

---

## 📸 Screenshots

### Chat Interface

<img width="1865" height="923" alt="Screenshot 2026-08-30 192546" src="https://github.com/user-attachments/assets/d4dff44c-ee93-418e-b225-ab789886c09d" />

### LangGraph Workflow

<img width="135" height="256" alt="LangGraph Workflow" src="https://github.com/user-attachments/assets/1c06a605-25ed-4d26-8000-c761c83dc4fd" />

---

## 🚀 Features

- 🤖 Conversational AI chatbot
- ⚡ Streaming AI responses (filtered to only stream the model's actual reply, not raw tool output)
- 💬 ChatGPT-style chat interface
- 🗂️ Multiple conversation threads
- ➕ Create new conversations
- 🔀 Switch between previous conversations
- 🧵 Thread-based conversation state
- 💾 LangGraph checkpointing (SQLite-backed, persists across restarts)
- 🧠 Stateful conversations
- 📋 Chat history in the sidebar, with auto-generated titles
- 🔌 **Toggleable connectors** — Wikipedia search and Tavily live web search, selectable per-conversation from a popover next to the chat input
- 🛠️ **Tool calling** — the LLM decides when to call a connector, executes it, and synthesizes the result into a natural-language answer
- 🗓️ **Freshness-aware web search** — Tavily calls are biased toward same-day results and the model is prompted to include the current date in its search query
- ✂️ **Conversation trimming** — history is capped by an approximate token budget before every LLM call, so long threads don't blow past Groq's per-minute token limits
- 🛡️ **Graceful error handling** — provider-side errors (rate limits, tool-choice mismatches) are caught and shown as a readable message instead of crashing the app
- 🔐 Secure API key management using `.env`
- 🎨 Streamlit frontend
- ⚙️ LangGraph backend workflow
- 🔗 LangChain message integration
- 🚀 Groq LLM inference (`openai/gpt-oss-120b`)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Workflow, state management, conditional tool routing |
| LangChain | LLM, message, and tool integration |
| Groq | LLM inference |
| Tavily (`langchain_tavily`) | Live web search connector |
| Wikipedia (`langchain_community`) | Encyclopedic search connector |
| Streamlit | Web interface |
| TypedDict | State definition (`messages`, `selected_connectors`) |
| `Annotated` / `add_messages` | Conversation message accumulation |
| `SqliteSaver` | Persistent, on-disk checkpointing (`chatbot.db`) |
| `trim_messages` | Token-budget-aware history trimming |
| python-dotenv | Environment variable management |

---

## 📁 Project Structure

```text
MULTI-USE-CHATBOT/
│
├── app.py                  # Streamlit frontend
├── chatbot_backend.py      # LangGraph workflow, tools, and LLM logic
├── chatbot.db               # SQLite checkpoint store (created at runtime)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignored files
│
└── .env                    # Local API keys (GROQ_API_KEY, TAVILY_API_KEY) - not committed
```

---

## 🧠 Application Architecture

```
                    USER
                      │
                      ↓
            ┌──────────────────┐
            │   Streamlit UI   │
            │      app.py      │
            └────────┬─────────┘
                      │
                      ↓
      HumanMessage + selected_connectors
                      │
                      ↓
            ┌──────────────────┐
            │    LangGraph     │
            │    Workflow      │
            └────────┬─────────┘
                      │
                      ↓
                 Query Node
              (binds selected
               connector tools)
                      │
                      ↓
                  Groq LLM
                      │
              tool call needed? ──── no ──→ END (query's answer is final)
                      │
                     yes
                      ↓
                Tools Node
          (executes wikipedia_search
             and/or web_search)
                      │
                      ↓
               Chatbot Node
        (synthesizes tool results
          into a final answer)
                      │
                      ↓
                    END
                      │
                      ↓
              Stream Response
           (AIMessage chunks only)
                      │
                      ↓
            ┌──────────────────┐
            │   Streamlit UI   │
            └──────────────────┘
```

---

## 🔄 LangGraph Workflow

The chatbot now uses a **conditional, tool-using** workflow rather than a single straight-line path:

```
START
  │
  ↓
Query Node ──(no tool needed)──→ END
  │
(tool call requested)
  │
  ↓
Tools Node
  │
  ↓
Chatbot Node
  │
  ↓
END
```

- **Query Node** — binds whichever connectors are selected in the UI (`wikipedia_search`, `web_search`, both, or neither) to the LLM, and adds a system instruction that spells out exactly what each bound tool can and can't do, so the model doesn't hallucinate a call to an unavailable tool or refuse to use one that's actually available.
- **Tools Node** — a standard LangGraph `ToolNode` that executes whichever tool the model called and appends the result as a `ToolMessage`.
- **Chatbot Node** — only reached *after* a tool has actually run; turns the raw tool output into a clean natural-language answer. (Earlier versions of this graph routed every turn through this node, which caused duplicate/garbled responses — fixed by sending the no-tool-needed path straight to `END` instead.)

---

## 🔌 Connectors

Selectable per-conversation from a "+" popover pinned next to the chat input:

| Connector | Backs onto | Good for |
|---|---|---|
| 📚 Wikipedia | `WikipediaAPIWrapper` | Static, encyclopedic facts (history, biography, science, places) |
| 🌐 Tavily | `TavilySearch` | Live/current information (prices, weather, news, exchange rates, scores) |

The system prompt sent to the model changes based on which connectors are active, so:
- With **only Wikipedia** enabled, a live-price question gets an honest "I don't have a connector for that" instead of a hallucinated number or an unhandled provider error.
- With **only Tavily** enabled, the model is told explicitly that it *can and should* use it for real-time questions, rather than defaulting to "I don't have real-time access."
- With **both** enabled, the model is nudged to pick the right tool for the question rather than always reaching for the same one.

---

## ✂️ Token Budget & Trimming

Groq's on-demand tier enforces a tokens-per-minute cap on `openai/gpt-oss-120b`. Because the SQLite checkpointer persists full thread history (including raw search-result `ToolMessage`s) forever, long-running threads can eventually exceed that cap in a single request.

To handle this, `trim_history()` approximates the token size of the conversation and keeps only the most recent slice of it (starting cleanly on a `HumanMessage` boundary) before every LLM call, in both the `query` and `chatbot` nodes. Tavily's own result size is also capped (`max_results=3`, `search_depth="basic"`) to keep each individual tool call lighter.

---

## 🛡️ Error Handling

Provider-side failures (rate limits, tool-choice mismatches, etc.) are caught around the streaming loop in `app.py` and shown as a readable in-chat warning instead of crashing the Streamlit process with a raw traceback.

---

## 🧵 Conversation Threads

Every conversation is assigned a unique `thread_id`.

```
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
```

---

## 💾 Checkpointing

The project now uses **`SqliteSaver`** (persistent, on-disk at `chatbot.db`) rather than `InMemorySaver`, so conversation threads and their `selected_connectors` state survive an app restart.

```
                 Checkpointer (SQLite)
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Thread A    Thread B    Thread C
          │           │           │
          ↓           ↓           ↓
       Messages    Messages    Messages
```

---

## 💬 Chat History

The Streamlit sidebar provides a ChatGPT-style conversation history.

Instead of showing raw thread IDs such as:

```
8f31c9a2
72bd912a
```

the UI displays a short title generated from the first user message:

```
How can I implement streaming...
Explain LangGraph state management
Build a RAG chatbot...
```

Users can click a previous conversation and continue that specific thread — including its `selected_connectors` state.

---

## 🔗 Frontend and Backend Flow

When a user sends a message:

```
User enters message
        ↓
Streamlit receives input + selected_connectors
        ↓
HumanMessage is created
        ↓
thread_id is attached
        ↓
LangGraph workflow starts
        ↓
Query node binds selected connector tools, receives state
        ↓
Messages are sent to Groq
        ↓
Model decides: answer directly, or call a tool?
        ↓
  (if tool called) → Tools node executes it → Chatbot node synthesizes final answer
        ↓
Response is streamed (AIMessage content only)
        ↓
Streamlit displays response
        ↓
Conversation state is updated & trimmed for next turn
```

---

## 🔮 Future Improvements

Planned improvements include:

- [x] ~~Persistent database checkpointing~~ *(SQLite — done)*
- [ ] PostgreSQL-based conversation storage
- [ ] Better automatic conversation titles
- [ ] Delete conversations
- [ ] Rename conversations
- [ ] Search conversations
- [ ] User authentication
- [x] ~~Tool calling~~ *(done)*
- [x] ~~Web search~~ *(Tavily connector — done)*
- [ ] RAG pipeline
- [ ] PDF document Q&A
- [ ] Vector database integration
- [ ] Multiple LLM providers
- [ ] Model selection
- [ ] Token usage tracking / display in UI
- [x] ~~Better error handling~~ *(provider errors caught gracefully — done)*
- [ ] Retry mechanisms
- [ ] FastAPI backend
- [ ] Dockerization
- [ ] Production deployment
