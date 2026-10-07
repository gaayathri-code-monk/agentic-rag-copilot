"""
tests/test_pipeline.py
Unit tests for the RAG pipeline components.
Run: pytest tests/ -v
"""

import sys
sys.path.insert(0, str(__file__.replace("tests/test_pipeline.py", "src")))

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestor import chunk_documents
from retriever import Retriever, RetrievedChunk, RetrievalResult
from langchain_core.documents import Document


# ── Chunking tests ─────────────────────────────────────────────────────────────

class TestChunking:
    def _make_doc(self, text):
        return Document(page_content=text, metadata={"source_file": "test.pdf", "page": 1})

    def test_short_doc_stays_single_chunk(self):
        doc = self._make_doc("Short paper abstract.")
        chunks = chunk_documents([doc])
        assert len(chunks) == 1

    def test_long_doc_splits_into_multiple_chunks(self):
        text = "word " * 600   # well above CHUNK_SIZE=512
        doc = self._make_doc(text)
        chunks = chunk_documents([doc])
        assert len(chunks) > 1

    def test_empty_chunks_are_dropped(self):
        doc = self._make_doc("   \n\n   ")
        chunks = chunk_documents([doc])
        assert len(chunks) == 0

    def test_metadata_preserved(self):
        doc = self._make_doc("Some content about RAG systems.")
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.pdf"


# ── Retrieval self-check tests ─────────────────────────────────────────────────

class TestRetrievalSelfCheck:
    def _make_retriever(self):
        """Retriever with mocked ChromaDB — no real DB needed."""
        with patch("retriever.Chroma"), patch("retriever.HuggingFaceEmbeddings"):
            r = Retriever.__new__(Retriever)
            r._store = MagicMock()
            return r

    def _chunks(self, scores):
        return [
            RetrievedChunk("content", "test.pdf", i, score)
            for i, score in enumerate(scores)
        ]

    def test_no_chunks_is_weak(self):
        r = self._make_retriever()
        is_strong, reason = r._self_check([], k=5)
        assert not is_strong
        assert reason is not None

    def test_too_few_chunks_is_weak(self):
        r = self._make_retriever()
        chunks = self._chunks([0.8])   # only 1 chunk when k=5 → < k/2
        is_strong, reason = r._self_check(chunks, k=5)
        assert not is_strong

    def test_low_scores_is_weak(self):
        r = self._make_retriever()
        chunks = self._chunks([0.1, 0.15, 0.12, 0.09, 0.11])
        is_strong, reason = r._self_check(chunks, k=5)
        assert not is_strong

    def test_good_scores_is_strong(self):
        r = self._make_retriever()
        chunks = self._chunks([0.7, 0.65, 0.68, 0.71, 0.66])
        is_strong, reason = r._self_check(chunks, k=5)
        assert is_strong
        assert reason is None


# ── RetrievalResult helpers ────────────────────────────────────────────────────

class TestRetrievalResult:
    def _result(self, scores):
        chunks = [
            RetrievedChunk(f"Content {i}", "paper.pdf", i, s)
            for i, s in enumerate(scores)
        ]
        return RetrievalResult(
            chunks=chunks, query="test", is_strong=True
        )

    def test_context_text_has_all_chunks(self):
        result = self._result([0.8, 0.7])
        assert "Content 0" in result.context_text
        assert "Content 1" in result.context_text

    def test_citations_correct_length(self):
        result = self._result([0.8, 0.7, 0.6])
        assert len(result.citations) == 3

    def test_citations_have_required_fields(self):
        result = self._result([0.75])
        c = result.citations[0]
        assert "source" in c
        assert "page"   in c
        assert "score"  in c
