import json
import time
from app.graph import rag_graph
from app.state import RAGState
from eval.judge import score

# ============ GOLDEN CASES ============
# Each case has expected_answer — the judge scores against this, not just against context.
GOLDEN_CASES = [
    # ---- Easy answerable (single fact) ----
    {"question": "Who is the CEO of Acme Corp?",
     "expected_behavior": "answerable",
     "expected_answer": "Priya Sharma"},
    {"question": "When was Acme Corp founded?",
     "expected_behavior": "answerable",
     "expected_answer": "2019"},
    {"question": "What is Acme's flagship product?",
     "expected_behavior": "answerable",
     "expected_answer": "RouteMate"},
    {"question": "How much funding did Acme raise?",
     "expected_behavior": "answerable",
     "expected_answer": "2 million dollars (seed round, 2022)"},

    # ---- Retrieval-sensitive answerable ----
    {"question": "Who is the Head of Engineering at Acme?",
     "expected_behavior": "answerable",
     "expected_answer": "Arjun Mehta"},
    {"question": "What pricing does RouteMate use?",
     "expected_behavior": "answerable",
     "expected_answer": "2,999 rupees per month"},
    {"question": "When did Acme reach profitability?",
     "expected_behavior": "answerable",
     "expected_answer": "Q4 2024"},
    {"question": "Which competitor did Acme's CEO previously work at?",
     "expected_behavior": "answerable",
     "expected_answer": "Delhivery"},

    # ---- Adversarial (topic-adjacent, must refuse) ----
    {"question": "Who is Acme's CTO?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},
    {"question": "What was Acme's revenue growth rate?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},
    {"question": "Who is Acme's Head of Marketing?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},
    {"question": "What is Acme's CTO's educational background?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},

    # ---- Off-topic (must refuse) ----
    {"question": "What is the recipe for chocolate cake?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},
    {"question": "Who won the 2022 FIFA World Cup?",
     "expected_behavior": "should_refuse",
     "expected_answer": "I don't have enough information"},
]


def _invoke_with_retry(graph, state, max_retries=4):
    """Run the graph, retrying on rate limits / deadlines."""
    for attempt in range(max_retries):
        try:
            return graph.invoke(state)
        except Exception as e:
            msg = str(e)
            is_retryable = (
                "DEADLINE_EXCEEDED" in msg
                or "429" in msg
                or "ResourceExhausted" in msg
                or "DeadlineExceeded" in msg
                or "rate" in msg.lower()
            )
            if is_retryable and attempt < max_retries - 1:
                wait = (2 ** attempt) * 10
                print(f"  [retry {attempt+1}/{max_retries-1}] {type(e).__name__}, waiting {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _write_report(results):
    """Write the report to disk after every case so partial runs are preserved."""
    n = len(results)
    if n == 0:
        return
    summary = {
        "total_cases": n,
        "avg_faithfulness": sum(r["faithfulness"] for r in results) / n,
        "avg_relevance": sum(r["relevance"] for r in results) / n,
        "refusal_accuracy": sum(1 for r in results if r["refusal_correct"]) / n,
        "avg_iterations": sum(r["iterations"] for r in results) / n,
        "results": results,
    }
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_evaluation():
    results = []
    total = len(GOLDEN_CASES)
    print(f"Running {total} cases...", flush=True)

    for i, case in enumerate(GOLDEN_CASES):
        print(f"\n[{i+1}/{total}] {case['question']}", flush=True)
        try:
            state = RAGState(
                question=case["question"],
                current_query=case["question"],
                documents=[],
                relevance_verdict="",
                refinement_reason="",
                refined_query="",
                generation="",
                iterations=0,
                sources=[],
            )
            output = _invoke_with_retry(rag_graph, state)
            context = "\n".join(output["documents"])
            print(f"  -> iter={output['iterations']}, answer={output['generation'][:80]!r}", flush=True)

            # Score with judge — passes expected_answer for comparison
            scores = score(
                question=case["question"],
                context=context,
                answer=output["generation"],
                expected_answer=case.get("expected_answer", ""),
            )

            refused = "don't have enough information" in output["generation"].lower()
            refusal_correct = (
                (case["expected_behavior"] == "should_refuse" and refused)
                or (case["expected_behavior"] == "answerable" and not refused)
            )

            results.append({
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "expected_answer": case.get("expected_answer", ""),
                "answer": output["generation"],
                "faithfulness": scores.get("faithfulness", 0),
                "relevance": scores.get("relevance", 0),
                "refusal_correct": refusal_correct,
                "iterations": output["iterations"],
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "expected_answer": case.get("expected_answer", ""),
                "answer": "ERROR",
                "faithfulness": 0,
                "relevance": 0,
                "refusal_correct": False,
                "iterations": 0,
                "error": str(e),
            })

        _write_report(results)
        time.sleep(1)  # avoid hammering the API between cases

    return json.load(open("eval_report.json", encoding="utf-8"))


if __name__ == "__main__":
    report = run_evaluation()
    print("\n=== SUMMARY ===", flush=True)
    print(f"total_cases:      {report['total_cases']}", flush=True)
    print(f"avg_faithfulness: {report['avg_faithfulness']:.2f}", flush=True)
    print(f"avg_relevance:    {report['avg_relevance']:.2f}", flush=True)
    print(f"refusal_accuracy: {report['refusal_accuracy']:.2f}", flush=True)
    print(f"avg_iterations:   {report['avg_iterations']:.2f}", flush=True)