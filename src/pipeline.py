"""
pipeline.py
Top-level RAG pipeline object.
The LangGraph agentic loop will import and call this.

Usage:
    from pipeline import RAGPipeline
    rag = RAGPipeline()
    result = rag.query("What is retrieval-augmented generation?")
    print(result.context_text)
    print(result.citations)
"""

from pathlib import Path

from rich.console import Console

from ingestor import ingest
from retriever import Retriever, RetrievalResult
from config import DATA_DIR

console = Console()


class RAGPipeline:
    """
    Thin orchestrator that combines:
      1. Ingestor  — build/update the ChromaDB index
      2. Retriever — query it, get grounded chunks with self-check
    """

    def __init__(self, auto_ingest: bool = False, reset_db: bool = False):
        """
        auto_ingest=True  → ingest all PDFs in data/pdfs/ on startup
        reset_db=True     → wipe and re-index (pass with auto_ingest=True)
        """
        if auto_ingest:
            console.print("[cyan]Auto-ingesting PDFs...[/cyan]")
            ingest(reset=reset_db)

        self.retriever = Retriever()

    def ingest_file(self, path: Path):
        """Ingest a single PDF without rebuilding the whole index."""
        ingest(files=[path])
        # reload retriever so it sees the new chunks
        self.retriever = Retriever()

    def query(self, question: str, k: int = 5, use_mmr: bool = True) -> RetrievalResult:
        """
        Retrieve relevant chunks for a question.
        Returns a RetrievalResult — check .is_strong before generating.
        """
        return self.retriever.retrieve(question, k=k, use_mmr=use_mmr)

    def stats(self) -> dict:
        return self.retriever.collection_stats()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = RAGPipeline()
    s = rag.stats()
    console.print(f"[green]Pipeline ready.[/green] DB has {s['total_chunks']} chunks.")

    if s["total_chunks"] == 0:
        console.print(
            "[yellow]DB is empty — drop PDFs into data/pdfs/ and run:[/yellow]\n"
            "  python src/ingestor.py"
        )
    else:
        question = "What are the main challenges in retrieval-augmented generation?"
        result = rag.query(question)
        console.print(f"\nQuery: [bold]{question}[/bold]")
        console.print(f"Strong: [bold]{result.is_strong}[/bold]  |  "
                      f"Chunks: {len(result.chunks)}")
        if result.citations:
            console.print("\nCitations:")
            for c in result.citations:
                console.print(f"  {c['source']}  p.{c['page']}  score={c['score']}")
