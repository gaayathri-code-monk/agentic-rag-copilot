"""
ingestor.py
Handles the full ingestion pipeline:
  PDF files  →  cleaned text  →  chunks  →  embeddings  →  ChromaDB

Run directly:
    python src/ingestor.py                  # ingest everything in data/pdfs/
    python src/ingestor.py --file paper.pdf # ingest one file
    python src/ingestor.py --reset          # wipe and re-index
"""

import argparse
import hashlib
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
)

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 of file content — used to skip already-indexed files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pdf(path: Path) -> list:
    """Load a PDF and return LangChain Document objects (one per page)."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # attach source metadata
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_hash"]   = file_hash(path)
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraph → sentence → word
    boundaries before hard-cutting at CHUNK_SIZE.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # drop empty / whitespace-only chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    return chunks


def get_embedding_fn():
    """Load the sentence-transformer embedding model (cached after first load)."""
    console.log(f"[cyan]Loading embedding model:[/cyan] {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )


def get_vector_store(embedding_fn, reset: bool = False) -> Chroma:
    """Open (or create) the persisted ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        console.log("[yellow]ChromaDB wiped — starting fresh.[/yellow]")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_DIR),
    )


def already_indexed(store: Chroma, file_hash_val: str) -> bool:
    """
    Check if a file (by hash) already has chunks in the store.
    Avoids re-embedding the same file on repeated runs.
    """
    results = store.get(where={"file_hash": file_hash_val}, limit=1)
    return len(results["ids"]) > 0


# ── Main ingest logic ─────────────────────────────────────────────────────────

def ingest(files: list[Path] | None = None, reset: bool = False):
    """
    Ingest PDFs into ChromaDB.
    - files: specific files to ingest; None = all PDFs in DATA_DIR
    - reset: wipe the collection first
    """
    start = time.time()

    # resolve file list
    if files is None:
        files = sorted(DATA_DIR.glob("*.pdf"))
    if not files:
        console.print(f"[red]No PDFs found in {DATA_DIR}. Drop some papers in there first.[/red]")
        return

    console.rule("[bold cyan]Agentic RAG Copilot — Ingestion Pipeline[/bold cyan]")
    console.print(f"  Found [bold]{len(files)}[/bold] PDF(s) to process\n")

    # load models
    embed_fn = get_embedding_fn()
    store    = get_vector_store(embed_fn, reset=reset)

    stats = {"skipped": 0, "ingested": 0, "chunks": 0, "errors": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting PDFs...", total=len(files))

        for pdf_path in files:
            progress.update(task, description=f"  {pdf_path.name[:40]}")

            try:
                # skip if already indexed (idempotent)
                fh = file_hash(pdf_path)
                if already_indexed(store, fh):
                    console.log(f"[dim]  SKIP (already indexed): {pdf_path.name}[/dim]")
                    stats["skipped"] += 1
                    progress.advance(task)
                    continue

                # load → chunk → store
                docs   = load_pdf(pdf_path)
                chunks = chunk_documents(docs)

                if not chunks:
                    console.log(f"[yellow]  WARN: no extractable text in {pdf_path.name}[/yellow]")
                    progress.advance(task)
                    continue

                # add to ChromaDB in batches of 100
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    store.add_documents(chunks[i : i + batch_size])

                console.log(
                    f"[green]  OK:[/green] {pdf_path.name} "
                    f"([dim]{len(docs)} pages → {len(chunks)} chunks[/dim])"
                )
                stats["ingested"] += 1
                stats["chunks"]   += len(chunks)

            except Exception as exc:
                console.log(f"[red]  ERROR: {pdf_path.name} — {exc}[/red]")
                stats["errors"] += 1

            progress.advance(task)

    # summary table
    elapsed = time.time() - start
    t = Table(title="\nIngestion Summary", show_header=True, header_style="bold cyan")
    t.add_column("Metric");  t.add_column("Value", justify="right")
    t.add_row("PDFs ingested",   str(stats["ingested"]))
    t.add_row("PDFs skipped",    str(stats["skipped"]))
    t.add_row("Errors",          str(stats["errors"]))
    t.add_row("Total chunks",    str(stats["chunks"]))
    t.add_row("Time elapsed",    f"{elapsed:.1f}s")
    console.print(t)

    collection = store._collection
    total_in_db = collection.count()
    console.print(f"\n[bold green]ChromaDB now holds {total_in_db} chunks total.[/bold green]")
    return store


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--file",  type=str, help="Single PDF filename (in data/pdfs/)")
    parser.add_argument("--reset", action="store_true", help="Wipe DB and re-index everything")
    args = parser.parse_args()

    files = None
    if args.file:
        p = DATA_DIR / args.file
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise SystemExit(1)
        files = [p]

    ingest(files=files, reset=args.reset)
