import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    MAX_RETRIEVAL_ITERATIONS = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "3"))
    COLLECTION_NAME = "rag_docs"

settings = Settings()