import sys
import time
from pathlib import Path


# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.localrag.config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from src.localrag.document_loader import (
    load_documents
)

from src.localrag.chunker import (
    chunk_documents
)

from src.localrag.embeddings import (
    embed_texts
)

from src.localrag.vector_store import (
    build_faiss_index
)

from src.localrag.bm25_store import (
    build_bm25_index
)


def main():

    print("=" * 70)
    print("LocalRAG Document Ingestion")
    print("=" * 70)

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        f"\nLoading documents from:\n"
        f"{DOCUMENTS_DIR}"
    )

    documents = load_documents(
        DOCUMENTS_DIR
    )

    if not documents:

        print(
            "\nNo PDF, TXT or Markdown documents found."
        )

        print(
            "Add documents to data/documents/"
        )

        return

    print(
        f"Loaded {len(documents)} document sections."
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print("\nChunking documents...")

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings with Ollama..."
    )

    start = time.perf_counter()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_texts(
        texts
    )

    embedding_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f}s"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print(
        "\nBuilding FAISS index..."
    )

    build_faiss_index(
        chunks,
        embeddings
    )

    print(
        "FAISS index saved."
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    build_bm25_index(
        chunks
    )

    print(
        "BM25 index saved."
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks:    {len(chunks)}"
    )

    print(
        f"Embedding: {embedding_time:.2f}s"
    )


if __name__ == "__main__":
    main()