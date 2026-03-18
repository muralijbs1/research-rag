import json
from datasets import Dataset

def load_eval_dataset(json_path: str) -> Dataset:
    try:
        with open(json_path, "r") as f:
            questions = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find test questions file at: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"File at {json_path} is not valid JSON")

    if not questions:
        raise ValueError("test_questions.json is empty")

    for i, q in enumerate(questions):
        if "question" not in q:
            raise KeyError(f"Entry {i} is missing 'question' field")
        if "ground_truth" not in q:
            raise KeyError(f"Entry {i} is missing 'ground_truth' field")

    return Dataset.from_dict({
        "question": [q["question"] for q in questions],
        "ground_truth": [q["ground_truth"] for q in questions],
        "answer": ["" for _ in questions],
        "contexts": [[] for _ in questions]
    })