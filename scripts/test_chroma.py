import time, sys
print("STEP 1: importing", flush=True)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings

print("STEP 2: building embeddings", flush=True)
t = time.time()
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=settings.GEMINI_API_KEY,
)
print(f"STEP 2 done in {time.time()-t:.1f}s", flush=True)

print("STEP 3: opening Chroma", flush=True)
t = time.time()
vs = Chroma(
    persist_directory=settings.CHROMA_PERSIST_DIR,
    collection_name=settings.COLLECTION_NAME,
    embedding_function=embeddings,
)
print(f"STEP 3 done in {time.time()-t:.1f}s", flush=True)
print(f"Collection count: {vs._collection.count()}", flush=True)

print("STEP 4: embedding a single test query", flush=True)
t = time.time()
try:
    vec = embeddings.embed_query("Who is the CEO of Acme Corp?")
    print(f"STEP 4 done in {time.time()-t:.1f}s, dim={len(vec)}", flush=True)
except Exception as e:
    print(f"STEP 4 ERROR: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)

print("STEP 5: similarity search", flush=True)
t = time.time()
docs = vs.similarity_search("Who is the CEO of Acme Corp?", k=4)
print(f"STEP 5 done in {time.time()-t:.1f}s, got {len(docs)} docs", flush=True)
for d in docs:
    print("---", d.page_content, flush=True)