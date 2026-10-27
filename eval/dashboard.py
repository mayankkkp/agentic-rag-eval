import streamlit as st
import json
import requests

API_URL = "https://agentic-rag-api-snmw.onrender.com"

st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("Self-Correcting RAG Agent")

tab_chat, tab_eval = st.tabs(["Ask a question", "Evaluation results"])

with tab_chat:
    st.caption(
        "The agent grades retrieval relevance, rewrites failed queries, and refuses "
        "when the knowledge base doesn't contain the answer. First request may take "
        "30-60s due to free-tier cold start."
    )
    q = st.text_input("Your question", placeholder="Who is the CEO of Acme Corp?")
    if st.button("Ask") and q:
        with st.spinner("Thinking…"):
            try:
                r = requests.post(f"{API_URL}/query", json={"question": q}, timeout=120)
                r.raise_for_status()
                data = r.json()
                st.markdown(f"**Answer:** {data['answer']}")
                st.markdown(f"**Iterations used:** {data['iterations_used']}")
                st.markdown("**Sources:**")
                for s in data["sources"]:
                    st.code(s)
            except requests.exceptions.Timeout:
                st.error("The server is cold-starting. Wait ~60s and try again.")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_eval:
    st.caption("Baseline = single-pass retrieval. With loop = self-correction enabled.")
    try:
        b = json.load(open("eval_report_baseline.json"))
        w = json.load(open("eval_report_with_loop.json"))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faithfulness", f"{w['avg_faithfulness']:.2f}/5",
                    delta=f"{w['avg_faithfulness']-b['avg_faithfulness']:+.2f}")
        col2.metric("Relevance", f"{w['avg_relevance']:.2f}/5",
                    delta=f"{w['avg_relevance']-b['avg_relevance']:+.2f}")
        col3.metric("Refusal accuracy", f"{w['refusal_accuracy']:.0%}",
                    delta=f"{(w['refusal_accuracy']-b['refusal_accuracy'])*100:+.0f}pp")
        col4.metric("Avg iterations", f"{w['avg_iterations']:.2f}",
                    delta=f"{w['avg_iterations']-b['avg_iterations']:+.2f}")

        st.subheader("Per-case results")
        st.dataframe(w["results"], use_container_width=True)
    except FileNotFoundError:
        st.info("No evaluation report found. Run `python -m eval.harness` first.")