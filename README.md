# Agentic RAG with Self-Correction

A retrieval-augmented generation (RAG) agent that **grades its own retrieval**, **rewrites failed queries**, and **refuses to answer** when the knowledge base doesn't contain the answer — instead of hallucinating. Built with LangGraph, FastAPI, ChromaDB, and Gemini, with a Streamlit dashboard for live querying and evaluation results.

**Live demo:** https://agentic-rag-dashboard.onrender.com 
**Evaluation dashboard:** https://agentic-rag-dashboard-xxxx.onrender.com  
**API:** https://agentic-rag-api-snmw.onrender.com

> Replace the dashboard URL with your actual Render URL once the second service is live.

---

## What it does

Most RAG pipelines do one thing: retrieve documents, stuff them into a prompt, generate an answer. If retrieval returns the wrong chunk, the model either hallucinates or gives a useless answer. This project adds a **self-correction loop**:

1. **Retrieve** — semantic search over the knowledge base via ChromaDB
2. **Grade** — an LLM judges whether the retrieved documents actually contain the answer
3. **Rewrite** — if the grade is `NO`, the LLM writes a better search query and retries (up to 2 iterations)
4. **Refuse** — if the answer still isn't found after max iterations, return "I don't have enough information to answer that" instead of making something up
5. **Cite** — every answer includes the source files it drew from

The refusal behavior is the point. A model that says "I don't know" is more trustworthy than one that guesses.

---

## Architecture
User question
│
▼
┌─────────────┐ ┌──────────────┐ ┌─────────────┐
│ Retrieve │────▶│ Grade │────▶│ Generate │
│ (Chroma) │ │ (LLM judge) │ │ (LLM) │
└─────────────┘ └──────┬───────┘ └─────────────┘
▲ │ NO
│ ▼
│ ┌──────────────┐
└────────────│ Rewrite │
│ (LLM) │
└──────────────┘

text

State is managed by LangGraph. Each node is a pure function that reads and writes to a typed `RAGState` dict. The conditional edge after `grade` decides whether to generate or loop back through `rewrite`.

**Stack:**
- **Agent:** LangGraph (state machine), LangChain (prompt chains)
- **LLM:** Gemini 3.5 Flash-Lite (generation, grading, query rewriting, evaluation judging)
- **Vector store:** ChromaDB (persistent, rebuilt on startup)
- **API:** FastAPI + Uvicorn (async endpoint, `asyncio.to_thread` for the blocking graph)
- **Dashboard:** Streamlit (chat + evaluation results in two tabs)
- **Deployment:** Render (Docker for the API, Python runtime for the dashboard)

---

## Evaluation

The project includes a 13-case evaluation harness (`eval/harness.py`) that compares **baseline** (single-pass retrieval, no correction loop) against **with-loop** (self-correction enabled). Each case has an expected answer and is scored by an LLM-as-judge on faithfulness and relevance.

**Results (13 cases):**

| Metric | Baseline | With Loop |
|---|---|---|
| Avg faithfulness | 5.00 / 5 | 5.00 / 5 |
| Avg relevance | 5.00 / 5 | 5.00 / 5 |
| Refusal accuracy | 92% | 92% |
| **Avg iterations** | **0.00** | **1.38** |

**Interpretation:** The loop fires on hard queries (avg 1.38 rewrite iterations) but doesn't move faithfulness or refusal accuracy because the generator already handles this knowledge base correctly on the first pass. The loop's value would appear on larger, noisier document sets where retrieval sometimes returns the wrong chunk — not on a small, clean 9-chunk demo corpus. This is a measured finding, not a marketing claim: the eval showed that retrieval quality, not generation quality, is the bottleneck at this scale.

**Run it yourself:**

```bash
# Baseline (loop disabled)
$env:BASELINE_MODE="true"
python -m eval.harness
cp eval_report.json eval_report_baseline.json

# With loop
$env:BASELINE_MODE="false"
python -m eval.harness
cp eval_report.json eval_report_with_loop.json
Reports are loaded by the dashboard's Evaluation results tab.

Demo limitations
This runs on Render's free tier and the Gemini free API. Both impose hard constraints:

Cold starts: Render sleeps the service after 15 minutes of inactivity. The first request after wake-up takes 30–60 seconds. The browser may time out before the server responds — retry once.

Query latency: 10–20 seconds per query due to Gemini free-tier rate limits. Each query makes 2–4 LLM calls (grade, optional rewrite, generate).

Daily quota: ~500 requests/day shared across all users of the demo API. Heavy use may return 429 Rate limit reached. Wait a few minutes.

Ephemeral storage: The vector DB rebuilds from source .txt files on every container restart ([startup] ingested 9 chunks in the logs). This adds ~5 seconds to cold start but keeps the demo functional without a paid disk.

These are constraints of the free tier, not the code. For a smoother experience, run locally with your own API key.

Run locally
Prerequisites: Python 3.10+, a Gemini API key.

bash
# Clone and enter
git clone https://github.com/YOUR_USERNAME/agentic-rag-eval
cd agentic-rag-eval

# Create venv
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# Ingest the knowledge base
python -m scripts.ingest
# Expected: "[startup] ingested 9 chunks into Chroma"

# Start the API
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs

# In a second terminal, start the dashboard
streamlit run eval/dashboard.py
# Open http://localhost:8501
Test query:

bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the CEO of Acme Corp?"}'
Expected:

json
{
  "answer": "The CEO of Acme Corp is Priya Sharma [1], [2], [3].",
  "sources": [".../company_faq.txt", ".../acme_team.txt", "..."],
  "iterations_used": 0,
  "verdict_history": ["YES"]
}
Test refusal:

bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recipe for chocolate cake?"}'
Expected: iterations_used: 2 (or 3), answer saying "I don't have enough information."

Project structure
text
agentic-rag-eval/
├── app/
│   ├── config.py         # env vars, defaults
│   ├── state.py          # RAGState TypedDict
│   ├── nodes.py          # retrieve, grade, rewrite, generate
│   ├── graph.py          # LangGraph wiring + should_rewrite router
│   ├── kb.py             # load + chunk + ensure_kb (auto-ingest on startup)
│   └── main.py           # FastAPI app (async /query, /health, startup hook)
├── eval/
│   ├── judge.py          # LLM-as-judge scoring (faithfulness, relevance)
│   ├── harness.py        # 13-case golden set + run_evaluation
│   └── dashboard.py      # Streamlit: chat tab + eval tab
├── scripts/
│   └── ingest.py         # thin wrapper around ensure_kb()
├── data/
│   └── knowledge_base/   # 7 .txt files: Acme Corp (fictional)
├── Dockerfile            # for Render API service
├── render.yaml           # two services: api + dashboard
└── requirements.txt
Using your own documents
The knowledge base is a fictional Acme Corp dataset so the demo is safe and self-contained. To use it for your own documents:

Delete the contents of data/knowledge_base/

Add your own .txt or .pdf files

Run python -m scripts.ingest (or redeploy — the startup hook rebuilds automatically)

Update GOLDEN_CASES in eval/harness.py to match your domain

Ask questions about your documents

Chunk size is 500 characters with 50 characters overlap (app/kb.py). Adjust for long-form documents (increase to 1000/100 for PDFs, decrease for short FAQs).

What this project demonstrates
Agentic loops with LangGraph — state machine with a conditional router and bounded iteration budget

LLM-as-judge evaluation — automated scoring of faithfulness and relevance with ground-truth comparison

Retrieval quality diagnosis — identified via evaluation that generation, not retrieval, was the limiter at this scale

RAG refusal behavior — the agent prefers "I don't know" to a hallucinated answer, and the eval measures whether this actually happens

Production-ish deployment — async FastAPI endpoint, Docker, auto-rebuild of ephemeral state on startup, health check, honest documentation of free-tier constraints

