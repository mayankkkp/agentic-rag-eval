import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.config import settings
from app.state import RAGState



os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

def _content_to_text(response):
    """Normalize Gemini response content to plain string."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    timeout=30,
    max_retries=2,
)

grader_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    response_mime_type="application/json",
    timeout=30,
    max_retries=2,
)

# ---------- Retrieval ----------
def retrieve(state: RAGState):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",  # Changed as well
    google_api_key=settings.GEMINI_API_KEY,
    )
    vectorstore = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
    )
    docs = vectorstore.similarity_search(state["current_query"], k=4)
    return {
        "documents": [d.page_content for d in docs],
        "sources": [d.metadata.get("source", "unknown") for d in docs],
        "iterations": state.get("iterations", 0),
    }

# ---------- Grading ----------
GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict retrieval grader. Given a question and retrieved documents, decide if the documents CONTAIN THE ACTUAL ANSWER to the question.

Respond with JSON only. Format: {{"verdict": "YES" or "NO", "reason": "...", "refined_query": "..."}}

STRICT RULES:
- YES only if the documents explicitly state the fact needed to answer.
- NO if the documents mention the topic but do not contain the specific answer.
- NO if the documents are unrelated.

Examples:
Q: "Who is the CEO?" Docs: "Acme builds software" → {{"verdict": "NO", "reason": "CEO not mentioned", "refined_query": "Acme Corp chief executive officer name"}}
Q: "Who is the CEO?" Docs: "The CEO is Priya Sharma" → {{"verdict": "YES", "reason": "CEO named", "refined_query": ""}}
Q: "Who is the CTO?" Docs: "The CEO is Priya Sharma" → {{"verdict": "NO", "reason": "CTO not mentioned", "refined_query": "Acme Corp chief technology officer leadership team"}}
Q: "What is the recipe for cake?" Docs about software → {{"verdict": "NO", "reason": "unrelated", "refined_query": "chocolate cake ingredients baking instructions"}}

CRITICAL: refined_query MUST differ from the original question. Add synonyms or specific terms. Never echo the question."""),
    ("human", "Question: {question}\n\nDocuments:\n{documents}"),
])

def grade_retrieval(state: RAGState):
    chain = GRADE_PROMPT | grader_llm
    raw = chain.invoke({
        "question": state["question"],
        "documents": "\n---\n".join(state["documents"]) if state["documents"] else "(no documents retrieved)",
    })
    text = _content_to_text(raw)
    try:
        result = JsonOutputParser().parse(text)
    except Exception:
        result = {"verdict": "NO", "reason": "parse failure", "refined_query": state["current_query"]}
    return {
        "relevance_verdict": result.get("verdict", "NO"),
        "refinement_reason": result.get("reason", ""),
        "refined_query": result.get("refined_query", state["current_query"]) or state["current_query"],
    }

# ---------- Query Rewriting ----------
def rewrite_query(state: RAGState):
    return {
        "current_query": state["refined_query"],
        "iterations": state["iterations"] + 1,
    }

# ---------- Generation ----------
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using ONLY the provided context.
If the context does not contain the answer, say exactly: "I don't have enough information to answer that."
Do not hallucinate. Cite sources by their number in brackets like [1], [2]."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def generate(state: RAGState):
    if state["documents"]:
        context = "\n---\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(state["documents"]))
    else:
        context = "(no context available)"
    chain = GENERATE_PROMPT | llm
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": _content_to_text(response)}
