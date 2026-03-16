"""
CiteRAG – Retrieval Pipeline (v2)
====================================
Tech stack:
  • LangGraph  – stateful retrieval workflow (graph-based orchestration)
  • ChromaDB   – persistent vector store with versioned collections
  • CrossEncoder (sentence-transformers) – passage reranking
  • BM25 (rank-bm25) – lexical retrieval leg of hybrid search
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CitationResult:
    doc_id: str
    text: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    citations: List[CitationResult]
    pipeline_version: str = "v2"


# ---------------------------------------------------------------------------
# LangGraph state  (typed dict travels through nodes)
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    query: str
    top_k: int
    bm25_hits: List[dict]          # [{doc_id, text, score}]
    vector_hits: List[dict]        # [{doc_id, text, score}]
    hybrid_hits: List[dict]        # fused
    reranked: List[dict]           # after cross-encoder
    answer: str
    citations: List[CitationResult]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    LangGraph-orchestrated Hybrid RAG pipeline:
      retrieve_bm25  ──┐
                       ├─► fuse ──► rerank ──► generate
      retrieve_vector ─┘
    """

    COLLECTION_VERSION = "v2"   # bump to re-index when schema changes

    def __init__(
        self,
        docs_path: Path,
        dataset_path: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chroma_persist_dir: Optional[Path] = None,
        dataset_limit: int = 200,
    ) -> None:
        self.docs_path = docs_path
        self.dataset_path = dataset_path
        self.dataset_limit = dataset_limit
        self.chroma_persist_dir = chroma_persist_dir or docs_path.parent / ".chroma_store"

        # ------------------------------------------------------------------
        # Embedding model (LangChain wrapper around sentence-transformers)
        # ------------------------------------------------------------------
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # ------------------------------------------------------------------
        # Cross-encoder reranker
        # ------------------------------------------------------------------
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        # ------------------------------------------------------------------
        # ChromaDB persistent vector store (versioned collection)
        # ------------------------------------------------------------------
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection_name = f"citerag_{self.COLLECTION_VERSION}"
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ------------------------------------------------------------------
        # Load documents and build / refresh indices
        # ------------------------------------------------------------------
        self._raw_docs: List[LCDocument] = []
        self._bm25: Optional[BM25Okapi] = None

        needs_index = self._collection.count() == 0
        self._load_documents()
        if needs_index and self._raw_docs:
            self._build_chroma_index()
        self._build_bm25()

        # ------------------------------------------------------------------
        # Build LangGraph workflow
        # ------------------------------------------------------------------
        self._graph = self._build_graph()

    # -----------------------------------------------------------------------
    # Document loading
    # -----------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=80, separators=["\n\n", "\n", " "]
        )
        return splitter.split_text(text)

    def _load_documents(self) -> None:
        docs: List[LCDocument] = []

        # ── Plain docs (Markdown / txt) ────────────────────────────────────
        for path in sorted(self.docs_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            doc_id = path.relative_to(self.docs_path).as_posix()
            for chunk in self._chunk_text(raw):
                docs.append(LCDocument(page_content=chunk, metadata={"doc_id": doc_id}))

        # ── JSONL dataset ──────────────────────────────────────────────────
        if self.dataset_path and self.dataset_path.exists():
            for path in sorted(self.dataset_path.rglob("*.jsonl")):
                count = 0
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            if count >= self.dataset_limit:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                title = data.get("title", "")
                                abstract = data.get("abstract", "")
                                doc_id = data.get("id", data.get("abs_url", "unknown"))
                                text = f"{title}\n{abstract}".strip()
                                if not text:
                                    continue
                                for chunk in self._chunk_text(text):
                                    docs.append(
                                        LCDocument(
                                            page_content=chunk,
                                            metadata={"doc_id": doc_id},
                                        )
                                    )
                                count += 1
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"[CiteRAG] Error loading {path}: {e}")

        self._raw_docs = docs

    # -----------------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------------

    def _build_chroma_index(self) -> None:
        """Embed and upsert all documents into ChromaDB."""
        texts = [d.page_content for d in self._raw_docs]
        metadatas = [d.metadata for d in self._raw_docs]
        ids = [
            hashlib.md5(f"{m['doc_id']}::{t}".encode()).hexdigest()
            for m, t in zip(metadatas, texts)
        ]
        # Batch upsert (Chroma handles deduplication by ID)
        batch = 512
        for i in range(0, len(texts), batch):
            embeddings = self.embeddings.embed_documents(texts[i : i + batch])
            self._collection.upsert(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                embeddings=embeddings,
                metadatas=metadatas[i : i + batch],
            )

    def _build_bm25(self) -> None:
        """Build BM25 index over all raw docs."""
        if not self._raw_docs:
            self._bm25 = BM25Okapi([["empty"]])
            return
        corpus = [d.page_content for d in self._raw_docs]
        self._bm25 = BM25Okapi([doc.split() for doc in corpus])

    # -----------------------------------------------------------------------
    # LangGraph nodes
    # -----------------------------------------------------------------------

    def _node_retrieve_bm25(self, state: RAGState) -> RAGState:
        if not self._raw_docs:
            return {**state, "bm25_hits": []}
        scores = self._bm25.get_scores(state["query"].split())
        k = state["top_k"] * 4
        idxs = np.argsort(scores)[::-1][:k]
        hits = [
            {
                "doc_id": self._raw_docs[i].metadata["doc_id"],
                "text": self._raw_docs[i].page_content,
                "score": float(scores[i]),
            }
            for i in idxs
            if float(scores[i]) > 0
        ]
        return {**state, "bm25_hits": hits}

    def _node_retrieve_vector(self, state: RAGState) -> RAGState:
        k = state["top_k"] * 4
        query_emb = self.embeddings.embed_query(state["query"])
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "doc_id": meta.get("doc_id", "unknown"),
                    "text": doc,
                    "score": float(1 - dist),   # cosine distance → similarity
                }
            )
        return {**state, "vector_hits": hits}

    def _node_fuse(self, state: RAGState) -> RAGState:
        """Reciprocal Rank Fusion of BM25 + vector results."""
        k_rrf = 60
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        doc_ids: dict[str, str] = {}

        for rank, hit in enumerate(state["bm25_hits"]):
            key = hit["text"][:120]
            scores[key] = scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
            texts[key] = hit["text"]
            doc_ids[key] = hit["doc_id"]

        for rank, hit in enumerate(state["vector_hits"]):
            key = hit["text"][:120]
            scores[key] = scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
            texts[key] = hit["text"]
            doc_ids[key] = hit["doc_id"]

        top_k = state["top_k"] * 2
        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        hybrid_hits = [
            {"doc_id": doc_ids[k], "text": texts[k], "score": scores[k]}
            for k in sorted_keys
        ]
        return {**state, "hybrid_hits": hybrid_hits}

    def _node_rerank(self, state: RAGState) -> RAGState:
        hits = state["hybrid_hits"]
        if not hits:
            return {**state, "reranked": []}
        pairs = [(state["query"], h["text"]) for h in hits]
        ce_scores = self.cross_encoder.predict(pairs, batch_size=64)
        order = np.argsort(ce_scores)[::-1][: state["top_k"]]
        reranked = [
            {**hits[i], "score": float(ce_scores[i])}
            for i in order
        ]
        return {**state, "reranked": reranked}

    def _node_generate(self, state: RAGState) -> RAGState:
        """Extractive answer assembly with Markdown stripping."""
        hits = state["reranked"]
        if not hits:
            return {
                **state,
                "answer": "I could not find relevant content for your query.",
                "citations": [],
            }
        citations: List[CitationResult] = []
        parts: List[str] = []
        for h in hits:
            snippet = h["text"].strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit("\n", 1)[0]
            clean = re.sub(r"[*#_]", "", snippet)
            citations.append(CitationResult(doc_id=h["doc_id"], text=snippet, score=h["score"]))
            parts.append(clean)
        answer = "\n\n---\n\n".join(parts)
        return {**state, "answer": answer, "citations": citations}

    # -----------------------------------------------------------------------
    # Graph assembly
    # -----------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        g = StateGraph(RAGState)
        g.add_node("retrieve_bm25", self._node_retrieve_bm25)
        g.add_node("retrieve_vector", self._node_retrieve_vector)
        g.add_node("fuse", self._node_fuse)
        g.add_node("rerank", self._node_rerank)
        g.add_node("generate", self._node_generate)

        g.set_entry_point("retrieve_bm25")
        g.add_edge("retrieve_bm25", "retrieve_vector")
        g.add_edge("retrieve_vector", "fuse")
        g.add_edge("fuse", "rerank")
        g.add_edge("rerank", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def answer(self, query: str, top_k: int = 5) -> AnswerResult:
        if not self._raw_docs:
            return AnswerResult(
                answer="No documents indexed yet. Add files to docs/ or dataset/.",
                citations=[],
            )
        initial_state: RAGState = {
            "query": query,
            "top_k": top_k,
            "bm25_hits": [],
            "vector_hits": [],
            "hybrid_hits": [],
            "reranked": [],
            "answer": "",
            "citations": [],
        }
        final_state = self._graph.invoke(initial_state)
        return AnswerResult(
            answer=final_state["answer"],
            citations=final_state["citations"],
        )

    @property
    def document_count(self) -> int:
        return len(self._raw_docs)

    @property
    def collection_version(self) -> str:
        return self.COLLECTION_VERSION
