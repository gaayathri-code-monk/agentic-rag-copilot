"""
retriever.py
Wraps ChromaDB with:
  - semantic similarity search
  - retrieval self-check (scores chunks, flags weak results)
  - MMR (Maximal Marginal Relevance) to reduce redundant chunks

This is the component the agentic loop will call and evaluate.
"""

from dataclasses import dataclass
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    RETRIEVAL_SCORE_THRESHOLD,
    TOP_K,
)

console = Console()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    content:     str
    source_file: str
    page:        int
    score:       float      # cosine similarity [0, 1]

    def __repr__(self):
        return (
            f"[score={self.score:.3f}] "
            f"{self.source_file} p.{self.page} — "
            f"{self.content[:80]}..."
        )


@dataclass
class RetrievalResult:
    chunks:       list[RetrievedChunk]
    query:        str
    is_strong:    bool    # True if at least TOP_K/2 chunks pass the score threshold
    weak_reason:  Optional[str] = None   # human-readable reason if is_strong=False

    @property
    def context_text(self) -> str:
        """Concatenated chunk content, ready to inject into the LLM prompt."""
        return "\n\n---\n\n".join(
            f"[Source: {c.source_file}, p.{c.page}]\n{c.content}"
            for c in self.chunks
        )

    @property
    def citations(self) -> list[dict]:
        return [
            {"source": c.source_file, "page": c.page, "score": round(c.score, 3)}
            for c in self.chunks
        ]


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Semantic retriever over the ChromaDB collection.
    Call retrieve(query) → RetrievalResult.
    The agentic loop checks result.is_strong before proceeding.
    """

    def __init__(self):
        embed_fn = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embed_fn,
            persist_directory=str(CHROMA_DIR),
        )

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        use_mmr: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks for a query.

        use_mmr=True  → Maximal Marginal Relevance:
                         balances relevance vs. diversity, reduces repeated info.
        use_mmr=False → pure similarity search (faster, can be repetitive).
        """
        if use_mmr:
            # MMR: fetch 2*k candidates, re-rank for diversity
            docs_and_scores = self._store.max_marginal_relevance_search_with_score(
                query, k=k, fetch_k=k * 2
            )
        else:
            docs_and_scores = self._store.similarity_search_with_score(query, k=k)

        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source_file=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page", 0),
                score=float(score),
            )
            for doc, score in docs_and_scores
        ]

        # ── Retrieval self-check ───────────────────────────────────────────────
        # The agentic loop uses this to decide whether to re-plan or proceed.
        is_strong, weak_reason = self._self_check(chunks, k)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            is_strong=is_strong,
            weak_reason=weak_reason,
        )

    def _self_check(
        self, chunks: list[RetrievedChunk], k: int
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate retrieval quality.
        Returns (is_strong, reason_if_weak).

        Rules:
          - No chunks at all          → definitely weak
          - Fewer than k/2 chunks     → weak (sparse corpus)
          - Mean score < threshold    → weak (semantically far)
          - Top score < threshold     → weak (best match is still poor)
        """
        if not chunks:
            return False, "No chunks retrieved — corpus may be empty or query is out of domain."

        if len(chunks) < max(1, k // 2):
            return False, (
                f"Only {len(chunks)} chunk(s) retrieved (expected ≥{k // 2}). "
                "Corpus may be too small or query too specific."
            )

        scores     = [c.score for c in chunks]
        mean_score = sum(scores) / len(scores)
        top_score  = max(scores)

        if top_score < RETRIEVAL_SCORE_THRESHOLD:
            return False, (
                f"Best chunk score {top_score:.3f} < threshold {RETRIEVAL_SCORE_THRESHOLD}. "
                "Query may be out of scope for the current corpus."
            )

        if mean_score < RETRIEVAL_SCORE_THRESHOLD * 0.8:
            return False, (
                f"Mean chunk score {mean_score:.3f} is low. "
                "Retrieved chunks may not be directly relevant."
            )

        return True, None

    def collection_stats(self) -> dict:
        """Quick info about what's in the DB — useful for debugging."""
        count = self._store._collection.count()
        return {"total_chunks": count, "collection": COLLECTION_NAME}


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What methods reduce hallucination in RAG systems?"

    retriever = Retriever()
    stats = retriever.collection_stats()
    console.print(f"\n[cyan]Collection:[/cyan] {stats['total_chunks']} chunks in DB")
    console.print(f"[cyan]Query:[/cyan] {query}\n")

    result = retriever.retrieve(query)

    console.print(f"[bold]Retrieved {len(result.chunks)} chunks[/bold]")
    console.print(f"[bold]Strong retrieval:[/bold] {result.is_strong}")
    if not result.is_strong:
        console.print(f"[yellow]Weak reason:[/yellow] {result.weak_reason}")

    console.rule("Chunks")
    for i, chunk in enumerate(result.chunks, 1):
        console.print(f"\n[{i}] {chunk}")
