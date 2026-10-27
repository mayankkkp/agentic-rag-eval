from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.config import settings

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    response_mime_type="application/json",
    timeout=60,
    max_retries=3,
)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge for a RAG system. Score the answer on two dimensions (1-5).

You are given:
- The question
- The context retrieved by the system
- The system's answer
- The EXPECTED answer (ground truth)

1. faithfulness: Is every claim in the answer supported by the context? (1=hallucinated, 5=fully grounded)
2. relevance: Does the answer correctly match the expected answer? (1=wrong, 5=exactly right)

For refusal cases where expected_answer is "I don't have enough information":
- An answer that refuses → relevance 5
- An answer that hallucinates a specific fact → relevance 1

Respond with JSON only. Format: {{"faithfulness": <int>, "relevance": <int>, "reason": "..."}}"""),
    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nExpected answer: {expected_answer}"),
])


def score(question: str, context: str, answer: str, expected_answer: str = "") -> dict:
    chain = JUDGE_PROMPT | judge_llm | JsonOutputParser()
    try:
        return chain.invoke({
            "question": question,
            "context": context[:3000],
            "answer": answer,
            "expected_answer": expected_answer or "(no ground truth provided)",
        })
    except Exception as e:
        print(f"  [judge error] {type(e).__name__}: {e}", flush=True)
        return {"faithfulness": 0, "relevance": 0, "reason": f"judge error: {e}"}