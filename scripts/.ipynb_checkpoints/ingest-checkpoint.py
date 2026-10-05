from pathlib import Path
import shutil
import time

from src.localrag.config import (
    PAPERS_DIR,
    INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from .document_loader import load_papers
from .chunker import chunk_sections
from .embeddings import embed_texts
from .indexer import build_indexes


def clear_previous_corpus():
    """
    Remove the previous uploaded documents and indexes.
    """

    if PAPERS_DIR.exists():
        for item in PAPERS_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    if INDEX_DIR.exists():
        for item in INDEX_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_uploaded_files(uploaded_files):
    """
    Save Streamlit UploadedFile objects to disk.
    """

    saved_files = []

    for uploaded_file in uploaded_files:

        filename = Path(
            uploaded_file.name
        ).name

        if not filename.lower().endswith(".pdf"):
            continue

        output_path = PAPERS_DIR / filename

        with open(output_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        saved_files.append(
            output_path
        )

    return saved_files


def create_paper_metadata(pdf_paths):
    """
    Convert locally uploaded PDFs into the metadata
    format expected by document_loader.py.
    """

    papers = []

    for path in pdf_paths:

        papers.append({
            "source": "User Upload",
            "external_id": path.stem,
            "doi": None,
            "title": path.stem,
            "year": None,
            "authors": [],
            "abstract": "",
            "citation_count": 0,
            "venue": None,
            "is_oa": True,
            "pdf_url": None,
            "landing_url": None,
            "type": "uploaded_pdf",
            "local_pdf": str(path),
        })

    return papers


def ingest_uploaded_pdfs(
    uploaded_files,
    progress_callback=None
):
    """
    Complete ingestion pipeline:

    Streamlit upload
        ↓
    Save PDFs
        ↓
    Extract text
        ↓
    Chunk documents
        ↓
    Generate Ollama embeddings
        ↓
    Build FAISS index
        ↓
    Build BM25 index
    """

    if not uploaded_files:
        raise ValueError(
            "No PDF files were uploaded."
        )

    total_start = time.perf_counter()

    def progress(value, message):
        if progress_callback:
            progress_callback(
                min(max(value, 0.0), 1.0),
                message
            )

    # ---------------------------------------------------------
    # 1. Clear previous corpus
    # ---------------------------------------------------------

    progress(
        0.02,
        "Clearing previous corpus..."
    )

    clear_previous_corpus()

    # ---------------------------------------------------------
    # 2. Save uploaded PDFs
    # ---------------------------------------------------------

    progress(
        0.08,
        "Saving uploaded PDFs..."
    )

    pdf_paths = save_uploaded_files(
        uploaded_files
    )

    if not pdf_paths:
        raise ValueError(
            "No valid PDF files were uploaded."
        )

    papers = create_paper_metadata(
        pdf_paths
    )

    # ---------------------------------------------------------
    # 3. Extract text
    # ---------------------------------------------------------

    progress(
        0.20,
        "Extracting text from PDFs..."
    )

    sections = load_papers(
        papers
    )

    if not sections:
        raise ValueError(
            "No text could be extracted from "
            "the uploaded PDFs."
        )

    # ---------------------------------------------------------
    # 4. Chunk documents
    # ---------------------------------------------------------

    progress(
        0.35,
        "Creating document chunks..."
    )

    chunks = chunk_sections(
        sections,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    # ---------------------------------------------------------
    # 5. Generate embeddings
    # ---------------------------------------------------------

    progress(
        0.45,
        "Generating Ollama embeddings..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    # ---------------------------------------------------------
    # 6. Build FAISS + BM25
    # ---------------------------------------------------------

    progress(
        0.85,
        "Building FAISS and BM25 indexes..."
    )

    build_indexes(
        chunks,
        embeddings
    )

    # ---------------------------------------------------------
    # 7. Finished
    # ---------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    progress(
        1.0,
        "Corpus ready."
    )

    return {
        "documents": len(pdf_paths),
        "pages": len(sections),
        "chunks": len(chunks),
        "embedding_count": len(embeddings),
        "processing_time": total_time,
        "files": [
            path.name
            for path in pdf_paths
        ],
    }