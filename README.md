# 📚 CiteRAG
*Your Intelligent, Domain-Specific Document Assistant*

Welcome to **CiteRAG**! Have you ever wished you could just *ask* your documentation a question and get an instant, accurate answer? That's exactly what this project does.

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
### 2. Add Your Knowledg
### 3. Start the Server
## 🐳 Running with Docker
## 🧪 Testing and Evaluation

## 📁 Project Structure
- `app/` – The brains of the operation (FastAPI, hybrid retrieval pipeline, models).
- `docs/` & `dataset/` – Where your domain-specific knowledge lives.
- `eval/` – Evaluation datasets and configuration thresholds.
- `tests/` – Unit tests to ensure everything stays polished.
- `Dockerfile` & `docker-compose.yml` – Project containerization config.

---
*Built with ❤️ to make navigating knowledge bases a breeze.*
