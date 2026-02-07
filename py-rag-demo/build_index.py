# file: build_index.py
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.textsplitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

if not OPENAI_API_BASE:
    raise ValueError("OPENAI_API_BASE environment variable is not set")

# 1. Your corpus (replace with file loaders in your training later)
raw_docs = [
    "Agentic AI systems can autonomously decide which tools to call.",
    "RAG combines retrieval from a vector store with generation from an LLM.",
    "Chroma is an open‑source vector database optimized for AI applications.",
]

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)
docs = text_splitter.split_documents([Document(page_content=t) for t in raw_docs])

# 3. Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# 4. Create / persist Chroma store
PERSIST_DIR = "./chroma-store"

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)

print("Chroma index built at", PERSIST_DIR)
