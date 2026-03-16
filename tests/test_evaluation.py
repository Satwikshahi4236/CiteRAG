import json
from pathlib import Path
from typing import List, Tuple

import pytest
import yaml
from rapidfuzz.fuzz import token_set_ratio

from app.retrieval.pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
EVAL_DATASET = PROJECT_ROOT / "eval" / "dataset.jsonl"
EVAL_CONFIG = PROJECT_ROOT / "eval" / "config.yaml"


def load_dataset() -> List[dict]:
  lines: List[dict] = []
  with EVAL_DATASET.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      lines.append(json.loads(line))
  return lines


def compute_retrieval_metrics(
  gold_doc_ids: List[str], retrieved_doc_ids: List[str]
) -> Tuple[float, float]:
  """Return (recall, reciprocal_rank) for a single example."""
  if not gold_doc_ids:
    return 1.0, 1.0

  gold_set = set(gold_doc_ids)
  retrieved = list(retrieved_doc_ids)

  # recall@k
  hits = [doc_id for doc_id in retrieved if doc_id in gold_set]
  recall = len(hits) / len(gold_set)

  # MRR@k
  rr = 0.0
  for rank, doc_id in enumerate(retrieved, start=1):
    if doc_id in gold_set:
      rr = 1.0 / rank
      break

  return recall, rr


def compute_answer_f1(expected: str, predicted: str) -> float:
  """Use token-set similarity as a proxy F1."""
  if not expected.strip():
    return 1.0
  if not predicted.strip():
    return 0.0
  # rapidfuzz returns 0-100
  return token_set_ratio(expected, predicted) / 100.0


@pytest.mark.slow
def test_rag_eval_meets_thresholds() -> None:
  assert DOCS_DIR.exists(), "docs/ directory must exist"
  assert EVAL_DATASET.exists(), "eval/dataset.jsonl must exist"
  assert EVAL_CONFIG.exists(), "eval/config.yaml must exist"

  with EVAL_CONFIG.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  min_recall = float(cfg.get("min_recall_at_5", 0.0))
  min_mrr = float(cfg.get("min_mrr_at_5", 0.0))
  min_answer_f1 = float(cfg.get("min_answer_f1", 0.0))
  top_k = int(cfg.get("top_k", 5))

  pipeline = RetrievalPipeline(docs_path=DOCS_DIR)
  data = load_dataset()

  recalls: List[float] = []
  mrrs: List[float] = []
  f1s: List[float] = []

  for row in data:
    question = row["question"]
    gold_doc_ids = row.get("relevant_doc_ids", [])
    expected_answer = row.get("expected_answer", "")

    result = pipeline.answer(question, top_k=top_k)
    retrieved_doc_ids = [c.doc_id for c in result.citations]

    recall, rr = compute_retrieval_metrics(gold_doc_ids, retrieved_doc_ids)
    f1 = compute_answer_f1(expected_answer, result.answer)

    recalls.append(recall)
    mrrs.append(rr)
    f1s.append(f1)

  mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
  mean_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
  mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0

  print(f"RAG eval: recall@5={mean_recall:.3f}, mrr@5={mean_mrr:.3f}, f1={mean_f1:.3f}")

  assert (
    mean_recall >= min_recall
  ), f"Recall@5 {mean_recall:.3f} < threshold {min_recall:.3f}"
  assert mean_mrr >= min_mrr, f"MRR@5 {mean_mrr:.3f} < threshold {min_mrr:.3f}"
  assert mean_f1 >= min_answer_f1, f"Answer F1 {mean_f1:.3f} < threshold {min_answer_f1:.3f}"

