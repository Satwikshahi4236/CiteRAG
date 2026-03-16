from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.preprocessing import normalize



@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class CitationResult:
    doc_id: str
    text: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    citations: List[CitationResult]


class RetrievalPipeline:
    """
    Hybrid retrieval (BM25 + vector) with cross-encoder reranking and
    simple extractive answer generation that enforces citations.
    """

    def __init__(self, docs_path: Path, dataset_path: Optional[Path] = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.docs_path = docs_path
        self.dataset_path = dataset_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.documents: List[Document] = []

        self._bm25: BM25Okapi | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._embeddings: np.ndarray | None = None
        self.cache_dir = self.docs_path.parent / ".index_cache"

        if self._load_from_cache():
            return

        self._load_documents()
        self._build_indices()
        self._save_to_cache()

    # --- Indexing helpers -------------------------------------------------

    def _chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def _load_documents(self) -> None:
        docs: List[Document] = []
        for path in sorted(self.docs_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            # Normalize to POSIX-style separators so doc_ids are stable
            # across platforms (e.g., for evaluation datasets).
            doc_id = path.relative_to(self.docs_path).as_posix()
            
            for chunk in self._chunk_text(text):
                docs.append(Document(doc_id=doc_id, text=chunk))

        if self.dataset_path and self.dataset_path.exists():
            for path in sorted(self.dataset_path.rglob("*.jsonl")):
                count = 0
                batch_limit = 200 # Reduced limit so it loads quickly
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            if count >= batch_limit:
                                break
                            if not line.strip(): continue
                            try:
                                data = json.loads(line)
                                title = data.get("title", "")
                                abstract = data.get("abstract", "")
                                doc_id = data.get("id", data.get("abs_url", ""))
                                text = f"{title}\n{abstract}".strip()
                                if not text: continue
                                for chunk in self._chunk_text(text):
                                    docs.append(Document(doc_id=doc_id, text=chunk))
                                count += 1
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        self.documents = docs

    def _load_from_cache(self) -> bool:
        if not self.cache_dir.exists():
            return False
        docs_path = self.cache_dir / "documents.pkl"
        bm25_path = self.cache_dir / "bm25.pkl"
        faiss_path = self.cache_dir / "index.faiss"
        embeddings_path = self.cache_dir / "embeddings.npy"
        
        if not all(p.exists() for p in [docs_path, bm25_path, faiss_path, embeddings_path]):
            return False
            
        try:
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            self._faiss_index = faiss.read_index(str(faiss_path))
            self._embeddings = np.load(embeddings_path)
            return True
        except Exception:
            return False

    def _save_to_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_dir / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            with open(self.cache_dir / "bm25.pkl", "wb") as f:
                pickle.dump(self._bm25, f)
            faiss.write_index(self._faiss_index, str(self.cache_dir / "index.faiss"))
            np.save(self.cache_dir / "embeddings.npy", self._embeddings)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _build_indices(self) -> None:
        corpus = [d.text for d in self.documents]
        if not corpus:
            self._bm25 = BM25Okapi([["empty"]])
            dim = 384
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._embeddings = np.zeros((1, dim), dtype="float32")
            return

        tokenized_corpus = [doc.split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

        embeddings = self.embedding_model.encode(corpus, convert_to_numpy=True, show_progress_bar=False, batch_size=256)
        embeddings = normalize(embeddings, axis=1)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))

        self._faiss_index = index
        self._embeddings = embeddings.astype("float32")

    # --- Retrieval --------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        assert self._bm25 is not None
        scores = self._bm25.get_scores(query.split())
        idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idxs]

    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        assert self._faiss_index is not None
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_emb = normalize(query_emb, axis=1).astype("float32")
        scores, idxs = self._faiss_index.search(query_emb, k)
        return [(int(idx), float(score)) for idx, score in zip(idxs[0], scores[0])]

    def _hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        Combine BM25 and vector scores with simple interpolation:
        score = alpha * bm25_norm + (1 - alpha) * dense_norm
        """
        k = max(k, 1)
        bm25_results = self._bm25_search(query, k=k * 4)
        vec_results = self._vector_search(query, k=k * 4)

        bm25_dict = {idx: score for idx, score in bm25_results}
        vec_dict = {idx: score for idx, score in vec_results}

        all_idxs = sorted(set(bm25_dict.keys()) | set(vec_dict.keys()))
        if not all_idxs:
            return []

        bm25_scores = np.array([bm25_dict.get(i, 0.0) for i in all_idxs])
        vec_scores = np.array([vec_dict.get(i, 0.0) for i in all_idxs])

        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)
        if vec_scores.max() > 0:
            vec_scores = vec_scores / (vec_scores.max() + 1e-9)

        combined = alpha * bm25_scores + (1 - alpha) * vec_scores
        order = np.argsort(combined)[::-1][:k]
        return [(int(all_idxs[i]), float(combined[i])) for i in order]

    # --- Reranking & Answering -------------------------------------------

    def _rerank(self, query: str, candidate_idxs: List[int], top_k: int) -> List[Tuple[int, float]]:
        if not candidate_idxs:
            return []
        pairs = [(query, self.documents[idx].text) for idx in candidate_idxs]
        scores = self.cross_encoder.predict(pairs, batch_size=256)
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidate_idxs[i], float(scores[i])) for i in order]

    def answer(self, query: str, top_k: int = 5) -> AnswerResult:
        """
        Run hybrid retrieval, cross-encoder reranking, and then generate
        an extractive answer composed directly from top passages.
        """
        if not self.documents:
            return AnswerResult(
                answer="No documents are indexed yet. Please add files under the docs/ directory.",
                citations=[],
            )

        hybrid_candidates = self._hybrid_search(query, k=max(top_k * 4, 10))
        candidate_idxs = [idx for idx, _ in hybrid_candidates]
        reranked = self._rerank(query, candidate_idxs, top_k=top_k)

        citations: List[CitationResult] = []
        answer_parts: List[str] = []
        import re
        for idx, score in reranked:
            doc = self.documents[idx]
            snippet = doc.text.strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit("\n", 1)[0]
            # Strip markdown for cleaner metrics matching against expected flat text
            clean_snippet = re.sub(r'[*#_]', '', snippet)
            citations.append(CitationResult(doc_id=doc.doc_id, text=snippet, score=score))
            answer_parts.append(clean_snippet)

        if not answer_parts:
            return AnswerResult(
                answer="I could not find relevant content in the indexed documents for this query.",
                citations=[],
            )

        answer = "\n\n---\n\n".join(answer_parts)
        return AnswerResult(answer=answer, citations=citations)

