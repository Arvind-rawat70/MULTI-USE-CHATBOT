# 🤖 LangGraph Chatbot

A simple AI chatbot built using **LangGraph**, **LangChain**, **Groq**, and **Streamlit**.

This project demonstrates how to create a basic conversational workflow using LangGraph with an in-memory checkpointer to maintain conversation state.

## 🚀 Tech Stack

* **Python**
* **LangGraph** – Workflow and state management
* **LangChain** – Message handling and LLM integration
* **Groq** – Large Language Model
* **Streamlit** – Chatbot web interface
* **Pydantic / TypedDict** – State definition
* **dotenv** – Environment variable management

## 📁 Project Structure

```text
project/
│
├── chatbot_backend.py     # LangGraph workflow and LLM logic
├── app.py                 # Streamlit frontend
├── .env                   # API key
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🔄 How It Works

The chatbot follows a simple LangGraph workflow:

```text
User Input
    ↓
Streamlit Chat Interface
    ↓
HumanMessage
    ↓
LangGraph Workflow
    ↓
Query Node
    ↓
Groq LLM
    ↓
AI Response
    ↓
Streamlit
```

The graph contains:

* **START** → Starts the workflow
* **Query Node** → Sends the conversation to the Groq LLM
* **END** → Finishes the workflow

## 🧠 State Management

The chatbot uses a TypedDict state:

```python
class Chatbot(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
```

`add_messages` allows new messages to be added to the existing conversation state.

The project also uses:

```python
InMemorySaver()
```

to store the conversation state while the application is running.

## 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not upload your `.env` file or API key to GitHub.

Add this to `.gitignore`:

```text
.env
__pycache__/
.venv/
```

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <project-folder>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The chatbot will open in your browser.

## 💬 Example

```text
User:
What is LangGraph?

Assistant:
LangGraph is a framework for building stateful,
multi-step applications with LLMs.
```

## 🎯 Learning Objectives

This project was created to understand:

* LangGraph StateGraph
* Nodes and edges
* START and END
* TypedDict state
* `Annotated`
* `add_messages`
* LangChain messages
* LLM invocation
* LangGraph checkpointers
* Thread IDs
* Streamlit chat interface
* Connecting frontend and backend

## 🔮 Future Improvements

Possible improvements include:

* Add conversation history UI
* Add multiple conversation threads
* Use persistent database-based checkpointing
* Add streaming responses
* Add tool calling
* Add web search
* Add RAG capabilities
* Deploy the application

## 👨‍💻 Author

**Arvind Rawat**

B.Tech CSE (AI/ML)

---

⭐ This project is part of my learning journey with **LangGraph and Generative AI**.
