# 📚 Ask My Docs
*Your Intelligent, Domain-Specific Document Assistant*

Welcome to **Ask My Docs**! Have you ever wished you could just *ask* your documentation a question and get an instant, accurate answer? That's exactly what this project does.

Built with a state-of-the-art **Hybrid RAG (Retrieval-Augmented Generation)** architecture, it reads through your internal documents and datasets, understands your questions, and returns precise answers—complete with exact citations so you know *exactly* where the info came from!

## ✨ Key Features
- **🧠 Smart Hybrid Retrieval:** Combines the best of old-school keyword search (BM25) with modern AI vector search (FAISS + Sentence Transformers).
- **🎯 Laser-Accurate Reranking:** Uses a powerful cross-encoder to guarantee the most relevant paragraphs bubble up to the top.
- **📑 Enforced Citations:** No AI hallucinations here! Every answer is derived strictly from your documents and clearly referenced.
- **✅ Automated Quality Control:** A built-in evaluation pipeline (with CI gates) ensures your answers are always hitting top metrics (Recall@5 & MRR).
- **⚡ Super Fast Caching:** Automatically indexes and caches massive document datasets locally for lightning-fast startups!

## 🚀 Getting Started

Want to spin this up on your local machine? It's easy!

### 1. Set Up Your Environment
Make sure you have Python 3.10+ installed. Open your terminal, navigate to the project folder, and run:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Add Your Knowledge
Simply drop your Markdown (`.md`), Text (`.txt`), or JSON Lines (`.jsonl`) files into the `docs/` or `dataset/` directories. Ask My Docs will automatically find them and learn your data on startup.

### 3. Start the Server
Run the FastAPI backend:
```bash
uvicorn app.main:app --reload
```
Then, open your favorite web browser and navigate to `http://localhost:8000` to start asking questions!

## 🐳 Running with Docker
If you prefer Docker, you can build and run the container in just two steps:
```bash
docker build -t ask-my-docs .
docker run -p 8000:8000 ask-my-docs
```

## 🧪 Testing and Evaluation
We take accuracy seriously. You can run our robust evaluation suite to ensure the RAG system is performing perfectly:
```bash
pytest -q
```
*Note: GitHub Actions will also run this suite automatically on every push!*

## 📁 Project Structure
- `app/` – The brains of the operation (FastAPI, hybrid retrieval pipeline, models).
- `docs/` & `dataset/` – Where your domain-specific knowledge lives.
- `eval/` – Evaluation datasets and configuration thresholds.
- `tests/` – Unit tests to ensure everything stays polished.
- `Dockerfile` & `docker-compose.yml` – Your containerization config.

---
*Built with ❤️ to make navigating knowledge bases a breeze.*
