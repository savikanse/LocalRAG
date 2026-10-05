from .config import (
    DENSE_K,
    BM25_K,
    RRF_K
)
from .reranker import rerank

from .embeddings import embed_query

from .vector_store import dense_search

from .bm25_store import bm25_search


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_k=RRF_K
):
    """
    Reciprocal Rank Fusion.

    RRF score:

        score(d) = Σ 1 / (k + rank)

    This combines rankings without requiring
    dense and BM25 scores to be on the same scale.
    """

    scores = {}

    chunks_by_id = {}

    # --------------------------------------------------------
    # Dense ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk = result["chunk"]

        chunk_id = chunk["id"]

        chunks_by_id[chunk_id] = chunk

        scores.setdefault(
            chunk_id,
            0.0
        )

        scores[chunk_id] += (
            1.0 /
            (rrf_k + rank)
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk = result["chunk"]

        chunk_id = chunk["id"]

        chunks_by_id[chunk_id] = chunk

        scores.setdefault(
            chunk_id,
            0.0
        )

        scores[chunk_id] += (
            1.0 /
            (rrf_k + rank)
        )

    # --------------------------------------------------------
    # Sort fused ranking
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        {
            "chunk": chunks_by_id[chunk_id],
            "score": score
        }
        for chunk_id, score in ranked
    ]


def retrieve(
    query,
    index,
    chunks,
    bm25,
    dense_k=DENSE_K,
    bm25_k=BM25_K
):

    query_embedding = embed_query(query)

    dense_results = dense_search(
        index,
        chunks,
        query_embedding,
        k=dense_k
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        k=bm25_k
    )

    # RRF combines FAISS + BM25
    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    # Cross-encoder reranks the RRF results
    reranked_results = rerank(
        query,
        fused_results
    )

    return reranked_results