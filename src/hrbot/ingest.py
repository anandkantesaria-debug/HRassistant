from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hrbot.config import Settings
from hrbot.embeddings import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    source: str
    page: int
    doc_title: str
    text: str


@dataclass
class LoadResult:
    pages: list[PageData]
    total_pages: int
    skipped_pages: int


@dataclass
class IngestSummary:
    files: int
    pages: int
    chunks: int
    skipped_pages: int
    elapsed_seconds: float


def find_pdfs(path: str | Path) -> list[Path]:
    """Recursively find all *.pdf files under path, sorted for determinism."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    pdfs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs)


def load_pdf(pdf_path: str | Path) -> LoadResult:
    """Extract text page by page with PyMuPDF.

    Pages with no extractable text are logged as warnings and counted in skipped_pages.
    """
    path = Path(pdf_path)
    source = path.name
    pages: list[PageData] = []
    skipped_pages = 0

    with pymupdf.open(path) as doc:
        meta_title = (doc.metadata or {}).get("title") or ""
        meta_title = meta_title.strip()
        # reportlab sets default title to 'untitled' if not specified; treat empty or 'untitled' carefully?
        # Wait: let's check what reportlab sets by default!
        doc_title = meta_title if meta_title and meta_title.lower() != "untitled" else source
        total_pages = len(doc)

        for idx, page in enumerate(doc):
            page_num = idx + 1
            raw_text = page.get_text("text")
            text = raw_text.strip()
            if not text:
                logger.warning(
                    "No extractable text on page %d of %s; skipping page.",
                    page_num,
                    source,
                )
                skipped_pages += 1
                continue

            pages.append(
                PageData(
                    source=source,
                    page=page_num,
                    doc_title=doc_title,
                    text=text,
                )
            )

    return LoadResult(
        pages=pages,
        total_pages=total_pages,
        skipped_pages=skipped_pages,
    )


def compute_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Compute a deterministic SHA-1 ID from source, page, chunk_index, and text."""
    payload = f"{source}\x00{page}\x00{chunk_index}\x00{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def chunk_pages(
    pages: list[PageData],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split each page independently into Document chunks with required metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []

    for page_data in pages:
        split_texts = splitter.split_text(page_data.text)
        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_id = compute_chunk_id(
                source=page_data.source,
                page=page_data.page,
                chunk_index=chunk_index,
                text=chunk_text,
            )
            metadata = {
                "source": page_data.source,
                "page": page_data.page,
                "chunk_index": chunk_index,
                "doc_title": page_data.doc_title,
                "chunk_id": chunk_id,
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    return chunks


def index_chunks(
    chunks: list[Document],
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    reset: bool = False,
) -> Chroma:
    """Store chunk embeddings in Chroma using cosine distance and deterministic IDs."""
    if settings is None:
        settings = Settings()
    if embeddings is None:
        embeddings = get_embeddings(settings=settings)

    vectorstore = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    if reset:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    if chunks:
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        vectorstore.add_documents(documents=chunks, ids=ids)

    return vectorstore


def ingest_directory(
    path: str | Path,
    reset: bool = False,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> IngestSummary:
    """Run full ingestion pipeline: discover PDFs, load pages, chunk, and index into Chroma."""
    start_time = time.perf_counter()

    if settings is None:
        settings = Settings()

    pdf_files = find_pdfs(path)
    all_pages: list[PageData] = []
    total_pages = 0
    skipped_pages = 0

    for pdf_file in pdf_files:
        res = load_pdf(pdf_file)
        all_pages.extend(res.pages)
        total_pages += res.total_pages
        skipped_pages += res.skipped_pages

    chunks = chunk_pages(
        all_pages,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    index_chunks(
        chunks=chunks,
        settings=settings,
        embeddings=embeddings,
        reset=reset,
    )

    elapsed = time.perf_counter() - start_time
    summary = IngestSummary(
        files=len(pdf_files),
        pages=total_pages,
        chunks=len(chunks),
        skipped_pages=skipped_pages,
        elapsed_seconds=elapsed,
    )

    print(
        f"Ingestion complete: files={summary.files}, pages={summary.pages}, "
        f"chunks={summary.chunks}, skipped_pages={summary.skipped_pages}, "
        f"elapsed={summary.elapsed_seconds:.2f}s"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest HR policy PDFs into Chroma.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory containing policy PDFs.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma collection before indexing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ingest_directory(path=args.path, reset=args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
