import os
from langgraph.graph import StateGraph, END
from app.state import RAGState
from app.nodes import retrieve, grade_retrieval, rewrite_query, generate
from app.config import settings


def should_rewrite(state: RAGState):
    if os.getenv("BASELINE_MODE") == "true":
        return "generate"
    if state["relevance_verdict"] == "YES":
        return "generate"
    if state["iterations"] >= settings.MAX_RETRIEVAL_ITERATIONS:
        return "generate"
    return "rewrite"      

builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_retrieval)
builder.add_node("rewrite", rewrite_query)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", should_rewrite, {
    "generate": "generate",
    "rewrite": "rewrite"
})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

rag_graph = builder.compile()