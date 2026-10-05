from pathlib import Path
import json
import shutil
import time

from .config import (
    PAPERS_DIR,
    INDEX_DIR,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PAPERS_JSON,
    CHUNKS_PATH,
)

from .document_loader import load_documents
from .chunker import chunk_documents
from .embeddings import embed_texts
from .vector_store import (
    build_faiss_index,
    add_chunks_to_index
)

from .bm25_store import (
    build_bm25_index
)

from .retriever import retrieve

def clear_previous_corpus():
    """
    Remove all uploaded PDFs and retrieval indexes.

    This is a hard reset. It's called once, automatically, when
    the app starts (in case a previous run left a corrupt or
    partial corpus on disk), and can also be triggered manually.
    It is NOT called on every ingestion anymore — ingestion is
    additive, see ingest_uploaded_pdfs().
    """

    # Clear DATA directory
    if PAPERS_DIR.exists():

        for item in PAPERS_DIR.iterdir():

            if item.is_file():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)

    # Clear index directory
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


# A marker file, not st.cache_resource, guards the one-time
# startup wipe. st.cache_resource keys on the cached function's
# own bytecode — every time app.py (or any module it imports)
# changes and Streamlit hot-reloads, the cache is treated as
# "new" and the wipe fires again, silently deleting whatever you
# just built. A file on disk survives hot-reloads, reruns, and
# full app restarts, and only ever fires once per data directory.
STARTUP_MARKER = DATA_DIR / ".startup_cleanup_done"


def run_startup_cleanup_once():
    """
    Wipe any stale/partial corpus left behind by a previous
    crashed run — but only the very first time this data
    directory has ever seen the app. Safe to call on every
    script rerun; it's a no-op once the marker exists.
    """

    if STARTUP_MARKER.exists():
        return False

    clear_previous_corpus()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STARTUP_MARKER.write_text(
        "Startup cleanup already ran for this data directory.\n",
        encoding="utf-8"
    )

    return True


def get_indexed_filenames():
    """
    Filenames already present in the persisted corpus metadata.

    Used so re-ingesting doesn't re-save or re-embed a PDF that's
    already part of the index.
    """

    if not PAPERS_JSON.exists():
        return set()

    try:

        existing = json.loads(
            PAPERS_JSON.read_text(encoding="utf-8")
        )

    except json.JSONDecodeError:
        return set()

    return {
        Path(paper["local_pdf"]).name
        for paper in existing
        if paper.get("local_pdf")
    }


def save_uploaded_files(
    uploaded_files,
    already_indexed
):
    """
    Save Streamlit UploadedFile objects to data/documents/.

    Files whose name is already in already_indexed are skipped,
    so reselecting the same PDF doesn't duplicate it in the index.
    """

    saved_paths = []

    skipped_files = []

    for uploaded_file in uploaded_files:

        filename = Path(
            uploaded_file.name
        ).name

        # Only accept PDFs
        if not filename.lower().endswith(".pdf"):
            continue

        if filename in already_indexed:

            skipped_files.append(
                filename
            )

            continue

        output_path = PAPERS_DIR / filename

        with open(output_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )

        saved_paths.append(
            output_path
        )

    return saved_paths, skipped_files


def create_metadata(pdf_paths):
    """
    Create metadata records compatible with the
    existing document_loader.py.
    """

    DATA = []

    for pdf_path in pdf_paths:

        DATA.append({
            "source": "User Upload",

            "external_id": pdf_path.stem,

            "doi": None,

            "title": pdf_path.stem,

            "year": None,

            "authors": [],

            "abstract": "",

            "citation_count": 0,

            "venue": None,

            "is_oa": True,

            "pdf_url": None,

            "landing_url": None,

            "type": "uploaded_pdf",

            "local_pdf": str(pdf_path),
        })

    return DATA


def append_metadata(new_papers):
    """
    Merge newly ingested paper metadata into papers.json instead
    of overwriting whatever's already there.
    """

    PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if PAPERS_JSON.exists():

        try:

            existing = json.loads(
                PAPERS_JSON.read_text(encoding="utf-8")
            )

        except json.JSONDecodeError:
            existing = []

    else:
        existing = []

    all_papers = existing + new_papers

    PAPERS_JSON.write_text(
        json.dumps(
            all_papers,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return all_papers

def _current_corpus_totals():
    """
    Read whatever's currently persisted, without ingesting
    anything. Used when a build has nothing new to add, so we
    can still report accurate totals instead of erroring out.
    """

    if PAPERS_JSON.exists():

        try:

            papers = json.loads(
                PAPERS_JSON.read_text(encoding="utf-8")
            )

        except json.JSONDecodeError:
            papers = []

    else:
        papers = []

    if CHUNKS_PATH.exists():

        chunks = json.loads(
            CHUNKS_PATH.read_text(encoding="utf-8")
        )

    else:
        chunks = []

    return papers, chunks


def ingest_uploaded_pdfs(
    uploaded_files,
    progress_callback=None
):
    """
    Additive ResearchMate ingestion pipeline.

    Newly uploaded PDFs
        ↓
    Save only the ones not already indexed
        ↓
    Extract page-level text (new PDFs only)
        ↓
    Chunk text (ids continue on from the existing corpus)
        ↓
    Generate Ollama embeddings (new chunks only)
        ↓
    Add to the existing FAISS index (or create one)
        ↓
    Rebuild BM25 over the full, merged chunk set
        ↓
    Merge metadata into papers.json

    This does NOT clear the existing corpus — that only happens
    once, automatically, when the app starts. Every call here
    grows the corpus instead of replacing it.
    """

    if not uploaded_files:

        raise ValueError(
            "No PDF files were uploaded."
        )

    start_time = time.perf_counter()

    def update_progress(
        value,
        message
    ):
        """
        Send progress information back to Streamlit.
        """

        value = max(
            0.0,
            min(1.0, value)
        )

        if progress_callback:

            progress_callback(
                value,
                message
            )

    PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # STEP 1 — Skip PDFs already in the corpus
    # ========================================================

    update_progress(
        0.05,
        "Checking existing corpus..."
    )

    already_indexed = get_indexed_filenames()

    # ========================================================
    # STEP 2 — Save only the new PDFs
    # ========================================================

    update_progress(
        0.10,
        "Saving uploaded PDFs..."
    )

    pdf_paths, skipped_files = save_uploaded_files(
        uploaded_files,
        already_indexed
    )

    if not pdf_paths:

        if skipped_files:

            # Nothing new to do — this isn't a failure, it's
            # just a no-op. Report current totals instead of
            # raising, so the UI doesn't show a scary error for
            # what is, in effect, a successful "already done".
            papers, chunks = _current_corpus_totals()

            update_progress(
                1.0,
                "Nothing new to add — all selected PDFs are "
                "already indexed."
            )

            return {
                "documents": len(papers),

                "chunks": len(chunks),

                "files": [
                    Path(paper["local_pdf"]).name
                    for paper in papers
                    if paper.get("local_pdf")
                ],

                "new_documents": 0,

                "new_pages": 0,

                "new_chunks": 0,

                "skipped_files": skipped_files,

                "processing_time": (
                    time.perf_counter()
                    - start_time
                ),
            }

        raise ValueError(
            "No valid PDF files were uploaded."
        )

    # ========================================================
    # STEP 3 — Record metadata for the new PDFs
    # ========================================================

    update_progress(
        0.15,
        "Recording document metadata..."
    )

    new_papers = create_metadata(
        pdf_paths
    )

    all_papers = append_metadata(
        new_papers
    )

    # ========================================================
    # STEP 4 — Extract text (new PDFs only)
    # ========================================================

    update_progress(
        0.20,
        "Extracting text from new PDFs..."
    )

    sections = load_documents(
        PAPERS_DIR,
        only_files=pdf_paths
    )

    if not sections:

        raise ValueError(
            "No text could be extracted from the newly "
            "uploaded PDFs. Make sure the PDFs contain "
            "machine-readable text."
        )

    # ========================================================
    # STEP 5 — Chunk new documents
    # ========================================================

    update_progress(
        0.35,
        "Splitting new documents into chunks..."
    )

    if CHUNKS_PATH.exists():

        existing_chunk_count = len(
            json.loads(
                CHUNKS_PATH.read_text(encoding="utf-8")
            )
        )

    else:
        existing_chunk_count = 0

    new_chunks = chunk_documents(
        sections,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        start_id=existing_chunk_count
    )

    if not new_chunks:

        raise ValueError(
            "No document chunks were created."
        )

    # ========================================================
    # STEP 6 — Generate embeddings (new chunks only)
    # ========================================================

    update_progress(
        0.45,
        "Generating Ollama embeddings..."
    )

    texts = [
        chunk["text"]
        for chunk in new_chunks
    ]

    embeddings = embed_texts(
        texts
    )

    if not embeddings:

        raise RuntimeError(
            "Ollama returned no embeddings."
        )

    if len(embeddings) != len(new_chunks):

        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of new document chunks."
        )

    # ========================================================
    # STEP 7 — Add to FAISS, rebuild BM25 over everything
    # ========================================================

    update_progress(
        0.85,
        "Updating FAISS and BM25 indexes..."
    )

    _, all_chunks = add_chunks_to_index(
        new_chunks,
        embeddings
    )

    # rank_bm25 has no incremental API, so BM25 is rebuilt from
    # the full merged chunk set — this is cheap (no embeddings
    # involved), unlike the FAISS/embedding steps.
    build_bm25_index(
        all_chunks
    )

    # ========================================================
    # STEP 8 — Finish
    # ========================================================

    processing_time = (
        time.perf_counter()
        - start_time
    )

    update_progress(
        1.0,
        "Research corpus is ready."
    )

    return {
        # Cumulative totals across the whole corpus
        "documents": len(all_papers),

        "chunks": len(all_chunks),

        "files": [
            Path(paper["local_pdf"]).name
            for paper in all_papers
            if paper.get("local_pdf")
        ],

        # This batch only
        "new_documents": len(pdf_paths),

        "new_pages": len(sections),

        "new_chunks": len(new_chunks),

        "skipped_files": skipped_files,

        "processing_time": processing_time,
    }