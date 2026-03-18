from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.evaluation.eval_dataset import load_eval_dataset
from src.generation.generator import generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]


def run_pipeline_on_dataset(dataset: Dataset, model: str | None = None) -> Dataset:
    """Run the full RAG pipeline on every question and populate answers + contexts."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for row in dataset:
        q = row["question"]
        chunks = retrieve(q, top_k=20)
        reranked = rerank(q, chunks, top_n=5)
        result = generate_answer(q, reranked, model=model)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["source_chunks"]])
        ground_truths.append(row["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def save_results_csv(scores: Any, dataset: Dataset, output_path: str) -> None:
    """Write per-question scores + metadata to a CSV file."""
    scores_df = scores.to_pandas()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        metric_names = [m.name for m in METRICS]
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "ground_truth"] + metric_names)
        writer.writeheader()

        for i, row in enumerate(dataset):
            record = {
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
            }
            for m in metric_names:
                record[m] = scores_df[m].iloc[i] if m in scores_df.columns else ""
            writer.writerow(record)


def evaluate_rag(
    json_path: str = "eval_data/test_questions.json",
    model: str | None = None,
    output_dir: str = "eval_results",
) -> dict[str, Any]:
    """
    Run a full RAGAS evaluation over all test questions.

    Parameters
    ----------
    json_path:
        Path to the test questions JSON file.
    model:
        Model selector passed to the generator (e.g. "openai", "anthropic").
        Defaults to DEFAULT_LLM from config.
    output_dir:
        Directory where the CSV results file will be written.

    Returns
    -------
    dict with:
      - "scores":  ragas EvaluationResult (printable, .to_pandas())
      - "dataset": the filled Dataset (questions + answers + contexts)
      - "csv_path": path to the saved CSV
    """
    dataset = load_eval_dataset(json_path)
    filled = run_pipeline_on_dataset(dataset, model=model)
    scores = evaluate(filled, metrics=METRICS, raise_exceptions=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = str(Path(output_dir) / f"eval_{timestamp}.csv")
    save_results_csv(scores, filled, csv_path)

    print(f"\n=== RAGAS Evaluation Results ===")
    print(scores)
    print(f"\nPer-question results saved to: {csv_path}")

    return {"scores": scores, "dataset": filled, "csv_path": csv_path}
