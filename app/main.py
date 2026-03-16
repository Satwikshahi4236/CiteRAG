from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .retrieval.pipeline import RetrievalPipeline, AnswerResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
DATASET_DIR = PROJECT_ROOT / "dataset"


app = FastAPI(title="Ask My Docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


pipeline: Optional[RetrievalPipeline] = None


def _ensure_docs_dir() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event() -> None:
    global pipeline
    _ensure_docs_dir()
    pipeline = RetrievalPipeline(docs_path=DOCS_DIR, dataset_path=DATASET_DIR)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    index_path = PROJECT_ROOT / "app" / "static" / "index.html"
    if not index_path.exists():
        return "<html><body><h1>Ask My Docs</h1><p>UI not found.</p></body></html>"
    return index_path.read_text(encoding="utf-8")


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")

    result: AnswerResult = pipeline.answer(request.query, top_k=request.top_k)
    citations = [
        Citation(doc_id=c.doc_id, score=c.score, text=c.text) for c in result.citations
    ]
    return QueryResponse(answer=result.answer, citations=citations)


@app.get("/health")
async def health() -> dict:
    ready = pipeline is not None
    return {"status": "ok" if ready else "initializing"}

