# Ask My Docs – Internal Overview

Our "Ask My Docs" system provides question answering over our internal documentation.

Key properties:

- Hybrid retrieval: BM25 + dense vector search
- Cross-encoder reranking over top candidates
- Citation-enforced answers composed from retrieved passages

Intended usage:

- Quickly answer questions about internal processes and SLAs
- Provide traceable answers with explicit citations to source documents

