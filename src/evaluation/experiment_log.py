"""
experiment_log.py

Wraps evaluate_rag() with MLflow logging.

Usage (in a notebook)
---------------------
from src.evaluation.experiment_log import run_experiment

scores = run_experiment(
    run_name="run_01_baseline",
    model="openai/gpt-4o-mini",
    chunk_size=500,
    chunk_overlap=50,
    reranker="sbert",
    rerank_n=5,
    use_langgraph=False,
    notes="Baseline",
)
"""

from __future__ import annotations

import os
import mlflow
from pathlib import Path

from src.evaluation.evaluator import evaluate_rag


def _run_langgraph_pipeline(dataset, model: str | None = None):
    """Run every question through LangGraph and return a HuggingFace Dataset."""
    from datasets import Dataset
    from src.graph.rag_graph import rag_pipeline

    questions, answers, contexts, ground_truths = [], [], [], []

    for row in dataset:
        q = row["question"]
        state = {"question": q, "model": model}
        result = rag_pipeline.invoke(state)

        questions.append(q)
        answers.append(result.get("answer", ""))
        contexts.append([c["text"] for c in result.get("source_chunks", [])])
        ground_truths.append(row["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_experiment(
    run_name: str,
    model: str = "openai/gpt-4o-mini",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    reranker: str = "sbert",
    rerank_n: int = 5,
    use_langgraph: bool = False,
    notes: str = "",
    mlflow_experiment: str = "RAG_Experiments",
) -> dict[str, float]:

    root = Path(os.getenv("PYTHONPATH"))
    json_path = str(root / "eval_data" / "test_questions.json")
    output_dir = str(root / "eval_results")

    # --- 1. Override env vars so config.py picks up experiment settings ----
    os.environ["RERANKER"] = reranker
    os.environ["RERANK_TOP_N"] = str(rerank_n)

    # --- 2. Run evaluation -------------------------------------------------
    if use_langgraph:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from src.evaluation.eval_dataset import load_eval_dataset
        from src.evaluation.evaluator import save_results_csv
        from datetime import datetime

        METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
        dataset = load_eval_dataset(json_path)
        filled = _run_langgraph_pipeline(dataset, model=model)
        scores = evaluate(filled, metrics=METRICS, raise_exceptions=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = str(Path(output_dir) / f"{run_name}_{timestamp}.csv")
        save_results_csv(scores, filled, csv_path)
        result = {"scores": scores, "csv_path": csv_path}
    else:
        result = evaluate_rag(json_path=json_path, model=model, output_dir=output_dir)

    # --- 3. Extract mean scores --------------------------------------------
    scores_df = result["scores"].to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    mean_scores = {
        m: float(scores_df[m].mean())
        for m in metric_names
        if m in scores_df.columns
    }

    # --- 4. Log to MLflow --------------------------------------------------
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model",         model)
        mlflow.log_param("chunk_size",    chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("reranker",      reranker)
        mlflow.log_param("rerank_n",      rerank_n)
        mlflow.log_param("use_langgraph", use_langgraph)

        for name, value in mean_scores.items():
            mlflow.log_metric(name, value)

        mlflow.log_artifact(result["csv_path"])

        if notes:
            mlflow.set_tag("notes", notes)

    # --- 5. Print summary --------------------------------------------------
    print(f"\n=== {run_name} ===")
    for name, value in mean_scores.items():
        print(f"  {name:<22} {value:.4f}")

    return mean_scores