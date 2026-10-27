from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings

# Resolve relative to this file so it works regardless of CWD
KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"


def load_and_split():
    """Load all .txt and .pdf files and split into chunks."""
    docs = []
    for file in KB_DIR.iterdir():
        if file.suffix == ".txt":
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def ensure_kb():
    """Rebuild the collection only if it's empty. Safe to call on every startup."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )
    vs = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
    )
    count = vs._collection.count()
    if count == 0:
        chunks = load_and_split()
        vs.add_documents(chunks)
        print(f"[startup] ingested {len(chunks)} chunks into Chroma", flush=True)
    else:
        print(f"[startup] Chroma already has {count} chunks", flush=True)