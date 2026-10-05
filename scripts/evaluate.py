import json
import sys
import time
from pathlib import Path
import unicodedata


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# Imports
# ============================================================

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


# ============================================================
# Evaluation settings
# ============================================================

K = 3

CURRENT_DOCUMENT = "__CURRENT_DOCUMENT__"


# ============================================================
# Source normalization
# ============================================================

def normalize_source(source):

    if not source:
        return ""

    source = str(source)

    source = unicodedata.normalize(
        "NFKC",
        source
    )

    source = source.strip().lower()

    source = " ".join(
        source.split()
    )

    return source


# ============================================================
# Reciprocal Rank
# ============================================================

def reciprocal_rank(
    retrieved_sources,
    relevant_sources
):

    relevant_sources = {
        normalize_source(source)
        for source in relevant_sources
    }

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):

        if (
            normalize_source(source)
            in relevant_sources
        ):

            return 1.0 / rank

    return 0.0


# ============================================================
# Resolve relevant sources
# ============================================================

def resolve_relevant_sources(
    configured_sources,
    available_sources
):

    available_lookup = {
        normalize_source(source): source
        for source in available_sources
    }

    resolved_sources = set()

    invalid_sources = set()

    for source in configured_sources:

        # ----------------------------------------------------
        # __CURRENT_DOCUMENT__
        # ----------------------------------------------------

        if source == CURRENT_DOCUMENT:

            if len(available_sources) == 1:

                resolved_sources.add(
                    available_sources[0]
                )

            else:

                print()
                print(
                    "ERROR: __CURRENT_DOCUMENT__ "
                    "requires exactly one indexed document."
                )

                print(
                    f"Currently indexed documents: "
                    f"{len(available_sources)}"
                )

                print(
                    "Available filenames:"
                )

                for filename in available_sources:

                    print(
                        f"  - {filename}"
                    )

                invalid_sources.add(
                    source
                )

            continue

        # ----------------------------------------------------
        # Normal explicit filename
        # ----------------------------------------------------

        normalized = normalize_source(
            source
        )

        if normalized in available_lookup:

            resolved_sources.add(
                available_lookup[normalized]
            )

        else:

            invalid_sources.add(
                source
            )

    return (
        resolved_sources,
        invalid_sources
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("LocalRAG Retrieval Evaluation")
    print("=" * 60)

    # ========================================================
    # Load questions
    # ========================================================

    questions_path = (
        ROOT
        / "evaluation"
        / "questions.json"
    )

    if not questions_path.exists():

        print()
        print(
            f"ERROR: questions.json not found:"
        )

        print(
            questions_path
        )

        return

    try:

        questions = json.loads(
            questions_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:

        print()
        print(
            "ERROR: Invalid questions.json"
        )

        print(error)

        return

    if not questions:

        print()
        print(
            "ERROR: No evaluation questions found."
        )

        return

    print()
    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    # ========================================================
    # Check index
    # ========================================================

    if not (
        FAISS_PATH.exists()
        and CHUNKS_PATH.exists()
        and BM25_PATH.exists()
    ):

        print()
        print(
            "ERROR: Retrieval index not found."
        )

        print()
        print(
            "Run:"
        )

        print(
            "python scripts/ingest.py"
        )

        return

    # ========================================================
    # Load retrieval system
    # ========================================================

    print()
    print(
        "Loading retrieval index..."
    )

    index, chunks = load_faiss_index()

    bm25 = load_bm25_index()

    # ========================================================
    # Find documents currently indexed
    # ========================================================

    available_sources = sorted({
        chunk["source"]
        for chunk in chunks
        if chunk.get("source")
    })

    if not available_sources:

        print()
        print(
            "ERROR: No document sources found "
            "in the current index."
        )

        return

    print()
    print(
        "Indexed documents:"
    )

    for source in available_sources:

        print(
            f"  - {source}"
        )

    print()
    print(
        f"Corpus size: "
        f"{len(available_sources)} document(s), "
        f"{len(chunks)} chunk(s)"
    )

    # ========================================================
    # Metrics
    # ========================================================

    hit_count = 0

    hit_at_1_count = 0

    reciprocal_ranks = []

    latencies = []

    # ========================================================
    # Evaluate each question
    # ========================================================

    for question_number, item in enumerate(
        questions,
        start=1
    ):

        question = item.get(
            "question",
            ""
        ).strip()

        configured_sources = item.get(
            "relevant_sources",
            []
        )

        print()
        print("-" * 60)

        print(
            f"Question {question_number}:"
        )

        print(question)

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:

            print(
                "ERROR: Question is empty."
            )

            continue

        if not configured_sources:

            print(
                "ERROR: No relevant_sources specified."
            )

            continue

        # ----------------------------------------------------
        # Resolve relevant sources
        # ----------------------------------------------------

        (
            relevant_sources,
            invalid_sources
        ) = resolve_relevant_sources(
            configured_sources,
            available_sources
        )

        print()
        print(
            "Configured relevant sources:"
        )

        for source in configured_sources:

            print(
                f"  - {source}"
            )

        print()
        print(
            "Resolved relevant sources:"
        )

        if relevant_sources:

            for source in relevant_sources:

                print(
                    f"  - {source}"
                )

        else:

            print(
                "  NONE"
            )

        # ----------------------------------------------------
        # Invalid source handling
        # ----------------------------------------------------

        if invalid_sources:

            print()
            print(
                "ERROR: Invalid relevant source(s):"
            )

            for source in invalid_sources:

                print(
                    f"  - {source}"
                )

            print()
            print(
                "Available filenames:"
            )

            for source in available_sources:

                print(
                    f"  - {source}"
                )

            print()
            print(
                "This question will not be counted "
                "as a retrieval hit."
            )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        start = time.perf_counter()

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

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        top_results = results[:K]

        retrieved_sources = [
            result["chunk"]["source"]
            for result in top_results
        ]

        # ----------------------------------------------------
        # Display retrieved sources
        # ----------------------------------------------------

        print()
        print(
            f"Retrieved sources (Top {K}):"
        )

        if retrieved_sources:

            for rank, source in enumerate(
                retrieved_sources,
                start=1
            ):

                print(
                    f"  {rank}. {source}"
                )

        else:

            print(
                "  No results returned."
            )

        # ----------------------------------------------------
        # Hit@K
        # ----------------------------------------------------

        relevant_normalized = {
            normalize_source(source)
            for source in relevant_sources
        }

        hit = any(
            normalize_source(source)
            in relevant_normalized
            for source in retrieved_sources
        )

        if hit:

            hit_count += 1

        # ----------------------------------------------------
        # Hit@1
        # ----------------------------------------------------

        hit_at_1 = (
            bool(retrieved_sources)
            and normalize_source(retrieved_sources[0])
            in relevant_normalized
        )

        if hit_at_1:

            hit_at_1_count += 1

        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------

        rr = reciprocal_rank(
            retrieved_sources,
            relevant_sources
        )

        reciprocal_ranks.append(
            rr
        )

        # ----------------------------------------------------
        # Per-question output
        # ----------------------------------------------------

        print()
        print(
            f"Hit@1: "
            f"{1 if hit_at_1 else 0}"
        )

        print(
            f"Hit@{K}: "
            f"{1 if hit else 0}"
        )

        print(
            f"Reciprocal Rank: "
            f"{rr:.3f}"
        )

        print(
            f"Retrieval latency: "
            f"{latency * 1000:.2f} ms"
        )
        print("=" * 60)

    # ========================================================
    # Aggregate summary
    # ========================================================

    num_evaluated = len(reciprocal_ranks)

    if num_evaluated == 0:

        print()
        print(
            "No questions were successfully evaluated."
        )

        return

    avg_latency_ms = (
        sum(latencies) / len(latencies)
    ) * 1000

    sorted_latencies = sorted(latencies)

    mid = len(sorted_latencies) // 2

    if len(sorted_latencies) % 2 == 0:

        median_latency_ms = (
            (sorted_latencies[mid - 1] + sorted_latencies[mid])
            / 2
        ) * 1000

    else:

        median_latency_ms = sorted_latencies[mid] * 1000

    print()
    print("=" * 60)
    print("Retrieval")
    print("-" * 60)

    print(
        f"Corpus size: "
        f"{len(available_sources)} document(s), "
        f"{len(chunks)} chunk(s)"
    )

    print()

    print(
        f"Questions evaluated: {num_evaluated}"
    )

    print(
        f"Hit@1: "
        f"{hit_at_1_count}/{num_evaluated} "
        f"({100 * hit_at_1_count / num_evaluated:.1f}%)"
    )

    print(
        f"Hit@{K}: "
        f"{hit_count}/{num_evaluated} "
        f"({100 * hit_count / num_evaluated:.1f}%)"
    )

    print(
        f"MRR: "
        f"{sum(reciprocal_ranks) / num_evaluated:.3f}"
    )

    print(
        f"Avg retrieval time: "
        f"{avg_latency_ms:.2f} ms"
    )

    print(
        f"Median retrieval time: "
        f"{median_latency_ms:.2f} ms"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()