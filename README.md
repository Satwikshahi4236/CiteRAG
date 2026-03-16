# 🔍 CiteRAG – Ask My Docs

> *Your intelligent, citation-enforced document assistant — powered by a modern RAG pipeline.*

Ever wished you could just *ask* your documentation a question and get a precise, sourced answer in seconds? That's CiteRAG. It reads your internal docs and research papers, understands your questions using a state-of-the-art hybrid retrieval pipeline, and hands you back clean, accurate answers — complete with citations so you always know exactly where the information came from.

---

## ✨ What Makes CiteRAG Different

Most RAG systems are a single vector search + an LLM call. CiteRAG is a proper, production-quality pipeline with multiple retrieval signals, smart fusion, and a full evaluation harness:

- **🔀 Hybrid Retrieval** — Combines keyword-based BM25 (great for exact terms) with dense vector search via ChromaDB (great for semantic similarity). Neither alone is as good as both together.
- **🏆 Reciprocal Rank Fusion (RRF)** — A mathematically principled method for merging ranked results from both retrieval legs into one clean, ordered candidate list.
- **🎯 CrossEncoder Reranking** — After retrieval, a sentence-transformer cross-encoder re-scores every candidate passage against your query to surface the most relevant ones.
- **📑 Enforced Citations** — Every answer is assembled *only* from retrieved passages. No hallucinations, no invented facts — every claim traces back to a source document.
- **🧠 LangGraph Orchestration** — The entire pipeline is modeled as an explicit, inspectable state graph, making each step visible, debuggable, and easy to extend.
- **📊 RAGAS Evaluation** — Built-in evaluation using the RAGAS framework measures faithfulness, answer relevancy, context recall, and context precision — so you always know how good your answers are.
- **⚡ Versioned Vector Store** — ChromaDB collections are versioned. Bumping the version tag triggers a clean re-index on the next startup. No more stale caches to hunt down.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (persistent, versioned) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Lexical Search** | BM25 (`rank-bm25`) |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) |
| **API** | FastAPI |
| **Dataset Format** | Markdown, `.txt`, `.jsonl` |

---

## 🚀 Getting Started

### 1. Set Up Your Environment

Make sure you have **Python 3.10+** installed, then run:

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

### 2. Add Your Knowledge

Drop your files into the right place:
- **Markdown / `.txt` files** → `docs/` (great for internal wikis, runbooks, SLA documents)
- **Research papers** → `dataset/` (expects `.jsonl` files with `title`, `abstract`, `id` fields)

CiteRAG will pick them up automatically on the next startup.

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Then open **`http://localhost:8000`** in your browser. The pipeline builds its ChromaDB index on first run (this takes a minute), then caches it — subsequent starts are instant.

### 4. Check the Pipeline Status

Hit the `/pipeline-info` endpoint to see how many documents are indexed and which collection version is active:

```bash
curl http://localhost:8000/pipeline-info
```

---

## 🐳 Docker

```bash
docker build -t citerag .
docker run -p 8000:8000 citerag
```

---

## 🧪 Running Evaluations

CiteRAG ships with a two-tier test suite:

```bash
# Fast: just the health check (runs in seconds)
pytest tests/test_health.py -v

# Full: RAGAS evaluation suite (takes a few minutes)
pytest tests/test_evaluation.py -v -s -m slow
```

The RAGAS evaluation measures four metrics against your ground-truth dataset in `eval/dataset.jsonl`:

| Metric | What It Checks |
|---|---|
| **Answer Relevancy** | Is the answer actually about the question? |
| **Faithfulness** | Are all claims grounded in retrieved passages? |
| **Context Recall** | Does the retrieved context cover the gold answer? |
| **Context Precision** | Is the context focused, or full of noise? |

Thresholds are configured in `eval/config.yaml`. CI fails if any metric drops below its threshold.

---

## ⚙️ Version Control for the Index

Want to force a full re-index (e.g., after adding new documents or changing the embedding model)?

Just bump the `COLLECTION_VERSION` constant in `app/retrieval/pipeline.py`:

```python
COLLECTION_VERSION = "v3"   # was "v2" — changing this triggers re-indexing on next startup
```

ChromaDB will create a fresh collection under the new name. Old collections remain on disk and can be deleted with `rm -rf .chroma_store/`.

---

## 📁 Project Layout

```
citerag/
├── app/
│   ├── main.py                  # FastAPI routes + startup
│   ├── retrieval/
│   │   └── pipeline.py          # LangGraph RAG pipeline (the core)
│   └── static/
│       └── index.html           # Query UI
├── docs/                        # Your Markdown / text documents
├── dataset/                     # JSONL research paper datasets
├── dataset_compressed/          # Compressed dataset archives (for GitHub)
├── eval/
│   ├── dataset.jsonl            # Ground-truth Q&A pairs
│   └── config.yaml              # Evaluation thresholds
├── tests/
│   ├── test_health.py           # Fast smoke test
│   └── test_evaluation.py      # RAGAS evaluation (gated)
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔄 How the Pipeline Works

```
User Query
    │
    ├──► BM25 Retrieval (keyword matching)
    │
    ├──► ChromaDB Vector Search (semantic similarity)
    │
    └──► Reciprocal Rank Fusion
              │
              └──► CrossEncoder Reranking
                        │
                        └──► Extractive Answer + Citations
```

Every step is a node in a **LangGraph StateGraph**, making the full execution trace inspectable and each component independently replaceable.

---

*Built to make navigating large knowledge bases feel effortless. Ask anything. Get cited answers.*
