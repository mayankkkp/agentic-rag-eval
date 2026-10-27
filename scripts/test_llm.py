import time
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

print("Testing plain LLM...", flush=True)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
)
t = time.time()
try:
    r = llm.invoke("Say the word: hello")
    print(f"PLAIN OK in {time.time()-t:.1f}s: {r.content!r}", flush=True)
except Exception as e:
    print(f"PLAIN FAIL: {type(e).__name__}: {e}", flush=True)

print("\nTesting JSON-mode LLM...", flush=True)
grader = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    response_mime_type="application/json",
)
t = time.time()
try:
    r = grader.invoke('Return JSON: {"verdict": "YES"}')
    print(f"JSON OK in {time.time()-t:.1f}s: {r.content!r}", flush=True)
except Exception as e:
    print(f"JSON FAIL: {type(e).__name__}: {e}", flush=True)
