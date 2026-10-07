# Agentic RAG Copilot

> Source-grounded, tool-augmented AI assistant — Phase 1: RAG Pipeline

## Project structure

```
rag_copilot/
├── data/
│   ├── pdfs/          ← drop your research PDFs here
│   └── chroma_db/     ← auto-created, persisted vector store
├── src/
│   ├── config.py      ← all settings in one place
│   ├── ingestor.py    ← PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py   ← semantic search + retrieval self-check
│   └── pipeline.py    ← wires everything together
└── tests/
    └── test_pipeline.py
```

## Setup

```bash
# 1. create venv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. install deps
pip install -r requirements.txt

# 3. (optional) copy env file for LLM keys
cp .env.example .env
# edit .env: set LLM_PROVIDER, OPENAI_API_KEY, etc.
```

## Quickstart

```bash
# Step 1 — ingest your PDFs
#   drop PDFs into data/pdfs/, then:
python src/ingestor.py

# Step 2 — test retrieval
python src/retriever.py "What methods reduce hallucination in RAG?"

# Step 3 — run the pipeline smoke test
python src/pipeline.py

# Step 4 — run unit tests
pytest tests/ -v
```

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Local, free, 384-dim, cosine-ready |
| Vector DB | ChromaDB (local) | Zero infra, offline demo, persisted |
| Chunking | RecursiveCharacterTextSplitter | Respects paragraph/sentence boundaries |
| Diversity | MMR re-ranking | Avoids redundant chunks in context |
| Self-check | Score threshold + count check | Enables agent re-planning on weak retrieval |

## What's next (Phase 2)

- `src/agent.py` — LangGraph Think→Act→Observe→Decide loop
- `src/tools/` — MCP-based arXiv search, PDF reader, calculator
- `src/generator.py` — citation-grounded LLM response layer
- `src/eval/` — RAGAS evaluation harness
