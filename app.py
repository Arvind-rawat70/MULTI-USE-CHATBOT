import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "PDF_QA"

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Multi-Tool Chatbot",
    layout="wide"
)

st.title("📄 PDF & Web Search Chatbot")

with st.sidebar:
    st.title("⚙ Settings")

    model = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]
    )
    
    tavily_api_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))
    
    st.subheader("Active Tools")
    use_tavily = st.checkbox("Tavily Search", value=True)
    use_duckduckgo = st.checkbox("DuckDuckGo Search", value=True)
    use_wikipedia = st.checkbox("Wikipedia Search", value=True)
    use_pdf = st.checkbox("PDF Document Search", value=True)

# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = None
if use_pdf:
    uploaded_file = st.file_uploader(
        "Upload PDF for Document Search",
        type="pdf"
    )

# ----------------------------
# Process Tools
# ----------------------------
if "tools" not in st.session_state:
    st.session_state.tools = []

current_tools = []

# Configure Web Search Tools
if use_tavily and tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    tavily_tool = TavilySearchResults(max_results=3)
    current_tools.append(tavily_tool)
elif use_tavily and not tavily_api_key:
    st.sidebar.warning("Please provide a Tavily API Key to use Tavily Search.")

if use_duckduckgo:
    ddg_tool = DuckDuckGoSearchRun()
    current_tools.append(ddg_tool)

if use_wikipedia:
    wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    current_tools.append(wiki_tool)

# Process PDF Tool
if use_pdf and uploaded_file:
    pdf_path = uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process PDF"):
        with st.spinner("Processing Document..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(documents)
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            vectorstore = FAISS.from_documents(
                docs,
                embeddings
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k":4}
            )
            
            pdf_tool = create_retriever_tool(
                retriever,
                "pdf_search",
                "Search and return information from the uploaded PDF document. Always use this tool when answering questions about the uploaded document."
            )
            st.session_state.pdf_tool = pdf_tool
            
        st.success("PDF Processed Successfully!")

if "pdf_tool" in st.session_state and use_pdf:
    current_tools.append(st.session_state.pdf_tool)

st.session_state.tools = current_tools

# ----------------------------
# Ask Questions
# ----------------------------
st.divider()

if not st.session_state.tools:
    st.info("Please activate at least one tool from the sidebar to start asking questions.")
else:
    question = st.text_input("Ask a question")

    if question:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent assistant capable of searching the web and reading uploaded PDFs. Use the tools available to you to answer the user's question accurately. If you don't know the answer, just say that you don't know."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, st.session_state.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=st.session_state.tools, verbose=True)

        with st.spinner("Generating Answer..."):
            try:
                response = agent_executor.invoke({"input": question})
                st.subheader("Answer")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")