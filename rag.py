from uuid import uuid4
from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ===============================
# Configuration
# ===============================

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "web_assistant"

llm = None
vector_store = None


# ===============================
# Initialize LLM + Vector Store
# ===============================

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=1000,
            api_key=st.secrets["GROQ_API_KEY"]
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ===============================
# Process URLs
# ===============================

def process_urls(urls):
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading data from URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    if not documents:
        yield "No documents loaded from URLs."
        return

    yield "Splitting data into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )

    docs = splitter.split_documents(documents)

    if not docs:
        yield "No document chunks were created."
        return

    # Attach source metadata
    for i, doc in enumerate(docs):
        doc.metadata["source"] = urls[i % len(urls)]

    yield "Adding document chunks to ChromaDB..."

    ids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=ids)

    yield "Vector store successfully updated."


# ===============================
# Generate Answer
# ===============================

def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector database is empty. Please process URLs first.")

    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "I don't know.", "No sources found."

    content = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the content given below.
If the answer is not present, say "I don't know". Do not hallucinate.

Content:
{content}

Question:
{query}
"""

    response = llm.invoke(prompt)

    sources = "\n".join(
        list(set([doc.metadata.get("source", "") for doc in docs]))
    )

    return response.content.strip(), sources
