from sentence_transformers import CrossEncoder

from .config import (
    RERANK_MODEL,
    RERANK_K,
    RERANK_CANDIDATES
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

    Input:
        fused_results from RRF

    Expected RRF result:
        {
            "chunk": {...},
            "score": <RRF score>
        }

    Output:
        {
            "chunk": {...},
            "rrf_score": <RRF score>,
            "rerank_score": <cross-encoder score>
        }
    """

    if not fused_results:
        return []

    model = get_reranker()

    # --------------------------------------------------------
    # Only score the top RERANK_CANDIDATES fused results.
    # RRF has already ranked these, so candidates far down the
    # list are extremely unlikely to end up in the final top K.
    # Scoring fewer pairs is the single biggest latency lever
    # in this pipeline, since cross-encoder inference dominates
    # end-to-end query time.
    # --------------------------------------------------------

    candidates = fused_results[:RERANK_CANDIDATES]

    # --------------------------------------------------------
    # Prepare query-passage pairs
    # --------------------------------------------------------

    pairs = [
        (
            query,
            result["chunk"]["text"]
        )
        for result in candidates
    ]

    # --------------------------------------------------------
    # Cross-encoder scores
    # --------------------------------------------------------

    scores = model.predict(
        pairs
    )

    # --------------------------------------------------------
    # Build final results
    # --------------------------------------------------------

    results = []

    for result, score in zip(
        candidates,
        scores
    ):

        # Get RRF score
        rrf_score = result.get(
            "score",
            result.get("rrf_score", 0.0)
        )

        results.append(
            {
                "chunk": result["chunk"],
                "rrf_score": float(rrf_score),
                "rerank_score": float(score)
            }
        )

    # --------------------------------------------------------
    # Sort by cross-encoder score
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:k]