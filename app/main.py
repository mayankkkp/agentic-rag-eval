import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.graph import rag_graph
from app.state import RAGState
from app.kb import ensure_kb
from google.genai.errors import ServerError, ClientError

app = FastAPI(title="Self-Correcting RAG Agent")


@app.on_event("startup")
def _startup():
    ensure_kb()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    iterations_used: int
    verdict_history: list[str]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    initial_state = RAGState(
        question=request.question,
        current_query=request.question,
        documents=[],
        relevance_verdict="",
        refinement_reason="",
        refined_query="",
        generation="",
        iterations=0,
        sources=[],
    )

    try:
        result = await asyncio.to_thread(rag_graph.invoke, initial_state)
    except ServerError as e:
        if "DEADLINE_EXCEEDED" in str(e):
            raise HTTPException(status_code=504, detail="AI service timed out. Please retry in 30 seconds.")
        raise HTTPException(status_code=502, detail="AI service unavailable.")
    except ClientError as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=429, detail="Rate limit reached. Try again in a few minutes.")
        raise HTTPException(status_code=502, detail="AI service error.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return QueryResponse(
        answer=result["generation"],
        sources=list(set(result["sources"])),
        iterations_used=result["iterations"],
        verdict_history=[result["relevance_verdict"]],
    )


@app.get("/health")
def health():
    return {"status": "ok"}