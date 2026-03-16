"""
RAGAS-based evaluation for CiteRAG.

Metrics computed:
  • answer_relevancy   – cosine similarity of answer embedding vs question
  • faithfulness       – fraction of answer claims supported by context
  • context_recall     – how much of the gold answer is covered by context
  • context_precision  – signal-to-noise in retrieved context

Thresholds are loaded from eval/config.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR     = PROJECT_ROOT / "docs"
DATASET_DIR  = PROJECT_ROOT / "dataset"
EVAL_DATASET = PROJECT_ROOT / "eval" / "dataset.jsonl"
EVAL_CONFIG  = PROJECT_ROOT / "eval" / "config.yaml"


def load_eval_dataset() -> List[dict]:
    rows: List[dict] = []
    with EVAL_DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config() -> dict:
    with EVAL_CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.slow
def test_rag_eval_ragas() -> None:
    """Run RAGAS evaluation and assert each metric meets its threshold."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError as e:
        pytest.skip(f"RAGAS or datasets not installed: {e}")

    from app.retrieval.pipeline import RetrievalPipeline

    assert DOCS_DIR.exists(),     "docs/ directory must exist"
    assert EVAL_DATASET.exists(), "eval/dataset.jsonl must exist"
    assert EVAL_CONFIG.exists(),  "eval/config.yaml must exist"

    cfg    = load_config()
    top_k  = int(cfg.get("top_k", 5))

    min_answer_relevancy   = float(cfg.get("min_answer_relevancy", 0.5))
    min_faithfulness       = float(cfg.get("min_faithfulness", 0.5))
    min_context_recall     = float(cfg.get("min_context_recall", 0.5))
    min_context_precision  = float(cfg.get("min_context_precision", 0.5))

    pipe   = RetrievalPipeline(docs_path=DOCS_DIR, dataset_path=DATASET_DIR)
    rows   = load_eval_dataset()

    questions:   List[str]       = []
    answers:     List[str]       = []
    contexts:    List[List[str]] = []
    ground_truths: List[str]     = []

    for row in rows:
        question       = row["question"]
        expected       = row.get("expected_answer", "")
        result         = pipe.answer(question, top_k=top_k)
        retrieved_ctx  = [c.text for c in result.citations]

        questions.append(question)
        answers.append(result.answer)
        contexts.append(retrieved_ctx if retrieved_ctx else [""])
        ground_truths.append(expected)

    ragas_dataset = Dataset.from_dict(
        {
            "question":      questions,
            "answer":        answers,
            "contexts":      contexts,
            "ground_truth":  ground_truths,
        }
    )

    result = evaluate(
        ragas_dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
    )

    scores = result.to_pandas().mean(numeric_only=True)

    ar  = float(scores.get("answer_relevancy",  0.0))
    ff  = float(scores.get("faithfulness",       0.0))
    cr  = float(scores.get("context_recall",     0.0))
    cp  = float(scores.get("context_precision",  0.0))

    print(
        f"\n[RAGAS] answer_relevancy={ar:.3f}  faithfulness={ff:.3f}  "
        f"context_recall={cr:.3f}  context_precision={cp:.3f}"
    )

    assert ar >= min_answer_relevancy,  f"answer_relevancy {ar:.3f} < {min_answer_relevancy}"
    assert ff >= min_faithfulness,      f"faithfulness {ff:.3f} < {min_faithfulness}"
    assert cr >= min_context_recall,    f"context_recall {cr:.3f} < {min_context_recall}"
    assert cp >= min_context_precision, f"context_precision {cp:.3f} < {min_context_precision}"
