import json

import faiss
import numpy as np

from .config import (
    FAISS_PATH,
    CHUNKS_PATH,
    INDEX_DIR
)


def build_faiss_index(
    chunks,
    embeddings
):
    """
    Build and persist a FAISS index.

    Embeddings are L2-normalized, so inner product
    corresponds to cosine similarity.
    """

    if not chunks:
        raise ValueError(
            "No chunks supplied."
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalize embeddings
    faiss.normalize_L2(matrix)

    dimension = matrix.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(matrix)

    # Save FAISS index
    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    # Save chunk metadata
    CHUNKS_PATH.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return index


def add_chunks_to_index(
    new_chunks,
    new_embeddings
):
    """
    Add new chunks + embeddings into the existing FAISS index
    (creating one if none exists yet), and merge the new chunk
    metadata into chunks.json rather than overwriting it.

    Unlike build_faiss_index(), this is additive: it's what
    ingestion uses so re-ingesting doesn't wipe out documents
    that are already indexed.
    """

    if not new_chunks:
        raise ValueError(
            "No chunks supplied."
        )

    if len(new_chunks) != len(new_embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix = np.asarray(
        new_embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(matrix)

    if FAISS_PATH.exists():

        index = faiss.read_index(
            str(FAISS_PATH)
        )

        if index.d != matrix.shape[1]:

            raise ValueError(
                "New embeddings don't match the existing index's "
                "dimension. If you've changed embedding models, "
                "clear the corpus and rebuild from scratch."
            )

    else:

        dimension = matrix.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

    index.add(matrix)

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    if CHUNKS_PATH.exists():

        existing_chunks = json.loads(
            CHUNKS_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        existing_chunks = []

    all_chunks = existing_chunks + new_chunks

    CHUNKS_PATH.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return index, all_chunks


def load_faiss_index():

    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {CHUNKS_PATH}"
        )

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    return index, chunks


def dense_search(
    index,
    chunks,
    query_embedding,
    k=10
):
    """
    Dense semantic retrieval.
    """

    if index.ntotal == 0:
        return []

    query = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    faiss.normalize_L2(query)

    k = min(
        k,
        index.ntotal
    )

    scores, ids = index.search(
        query,
        k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        ids[0]
    ):

        if index_id < 0:
            continue

        results.append(
            {
                "chunk": chunks[int(index_id)],
                "score": float(score)
            }
        )

    return results