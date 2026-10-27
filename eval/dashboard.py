import streamlit as st
import json
import importlib
import eval.harness
importlib.reload(eval.harness)

if st.button("Refresh"):
    st.rerun()

st.title("RAG Agent Evaluation Dashboard")

# Baseline vs. current comparison
st.header("Run Evaluation")
if st.button("Run Fresh Evaluation"):
    from eval.harness import run_evaluation
    report = run_evaluation()
    st.json(report)

st.header("Load Previous Report")
try:
    with open("eval_report.json") as f:
        report = json.load(f)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faithfulness", f"{report['avg_faithfulness']:.2f}/5")
    col2.metric("Relevance", f"{report['avg_relevance']:.2f}/5")
    col3.metric("Refusal Accuracy", f"{report['refusal_accuracy']:.0%}")
    col4.metric("Avg Iterations", f"{report['avg_iterations']:.1f}")

    st.subheader("Per-Case Results")
    st.dataframe(report["results"])
except FileNotFoundError:
    st.info("No report found. Run an evaluation first.")