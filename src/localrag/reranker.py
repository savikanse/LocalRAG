from sentence_transformers import CrossEncoder

from .config import (
    RERANK_MODEL,
    RERANK_K
)


_model = None


def get_reranker():

    global _model

    if _model is None:

        print(
            f"Loading reranker: {RERANK_MODEL}"
        )

        _model = CrossEncoder(
            RERANK_MODEL
        )

    return _model


def rerank(
    query,
    fused_results,
    k=RERANK_K
):
    """
    Cross-encoder reranking.

    The cross-encoder directly scores:
        (query, passage)

    rather than comparing independently generated
    embeddings.
    """

    if not fused_results:
        return []

    model = get_reranker()

    pairs = [
        (
            query,
            result["chunk"]["text"]
        )
        for result in fused_results
    ]

    scores = model.predict(
        pairs
    )

    results = []

    for result, score in zip(
        fused_results,
        scores
    ):

        results.append(
            {
                "chunk": result["chunk"],
                "score": float(score),
                "rrf_score": float(
                    result["score"]
                )
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:k]