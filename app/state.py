
from typing import List, TypedDict

class RAGState(TypedDict):
    question: str
    current_query: str
    documents: List[str]
    relevance_verdict: str          # "YES" or "NO"
    refinement_reason: str
    refined_query: str
    generation: str
    iterations: int
    sources: List[str]