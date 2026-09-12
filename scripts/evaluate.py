import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.localrag.config import (
    FAISS_PATH,
    CHUNKS_PATH,
    BM25_PATH
)

from src.localrag.vector_store import (
    load_faiss_index
)

from src.localrag.bm25_store import (
    load_bm25_index
)

from src.localrag.retriever import (
    retrieve
)


def reciprocal_rank(
    retrieved_sources,
    relevant_sources
):
    """
    Reciprocal Rank:

        1 / rank

    of the first relevant document.
    """

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):

        if source in relevant_sources:

            return 1.0 / rank

    return 0.0


def main():

    questions_path = (
        ROOT
        / "evaluation"
        / "questions.json"
    )

    questions = json.loads(
        questions_path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # Check index
    # --------------------------------------------------------

    if not (
        FAISS_PATH.exists()
        and CHUNKS_PATH.exists()
        and BM25_PATH.exists()
    ):

        print(
            "No index found."
        )

        print(
            "Run:"
        )

        print(
            "python scripts/ingest.py"
        )

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    index, chunks = (
        load_faiss_index()
    )

    bm25 = (
        load_bm25_index()
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    hit_count = 0

    reciprocal_ranks = []

    latencies = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for item in questions:

        question = (
            item["question"]
        )

        relevant_sources = set(
            item["relevant_sources"]
        )

        start = (
            time.perf_counter()
        )

        results = retrieve(
            question,
            index,
            chunks,
            bm25
        )

        latency = (
            time.perf_counter()
            - start
        )

        latencies.append(
            latency
        )

        retrieved_sources = [
            result["chunk"]["source"]
            for result in results
        ]

        # Hit@K
        if any(
            source in relevant_sources
            for source in retrieved_sources
        ):

            hit_count += 1

        # MRR
        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_sources,
                relevant_sources
            )
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    n = len(questions)

    if n == 0:

        print(
            "No evaluation questions found."
        )

        return

    hit_at_k = (
        hit_count / n
    )

    mrr = (
        sum(reciprocal_ranks)
        / n
    )

    average_latency = (
        sum(latencies)
        / n
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("LocalRAG Retrieval Evaluation")
    print("=" * 60)

    print(
        f"Questions:              {n}"
    )

    print(
        f"Hit@K:                  "
        f"{hit_at_k:.3f}"
    )

    print(
        f"MRR:                    "
        f"{mrr:.3f}"
    )

    print(
        f"Avg retrieval latency:  "
        f"{average_latency * 1000:.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()