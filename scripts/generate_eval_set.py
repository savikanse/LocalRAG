import argparse
import json
import random
import sys
from pathlib import Path

import ollama


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
    CHUNKS_PATH,
    OLLAMA_HOST,
    LLM_MODEL,
)


# ============================================================
# Settings
# ============================================================

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "evaluation"
    / "generated_questions.json"
)

# Skip near-empty chunks (page headers, references, etc.) —
# they rarely produce a meaningful, answerable question.
MIN_CHUNK_CHARS = 200

MAX_CHUNK_CHARS_FOR_PROMPT = 3000

QUESTION_PROMPT = """You are creating a retrieval-evaluation question from a document excerpt.

Read the excerpt below and write ONE specific, factual question that:
- can be directly and fully answered using ONLY this excerpt
- is specific enough that it probably would NOT be answerable from a
  different, unrelated document
- does not refer to "the excerpt", "the document", "the text", or
  similar meta-references

Respond with ONLY the question itself. No preamble, no quotes, no
numbering.

Excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""


# ============================================================
# Corpus loading
# ============================================================

def load_chunks():

    if not CHUNKS_PATH.exists():

        raise FileNotFoundError(
            f"No indexed chunks found at {CHUNKS_PATH}. "
            "Build a corpus first."
        )

    return json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )


def group_by_source(chunks):

    grouped = {}

    for chunk in chunks:

        source = chunk.get("source")

        if not source:
            continue

        grouped.setdefault(
            source,
            []
        ).append(chunk)

    return grouped


def pick_sample_chunks(
    chunks_for_doc,
    per_document,
    seed
):
    """
    Prefer substantial chunks over near-empty ones, and sample
    deterministically so re-runs with the same seed are
    reproducible.
    """

    usable = [
        chunk
        for chunk in chunks_for_doc
        if len(chunk["text"]) >= MIN_CHUNK_CHARS
    ]

    pool = usable or chunks_for_doc

    if len(pool) <= per_document:
        return pool

    rng = random.Random(seed)

    return rng.sample(
        pool,
        per_document
    )


# ============================================================
# Question generation
# ============================================================

def generate_question(
    client,
    chunk_text
):

    prompt = QUESTION_PROMPT.format(
        chunk_text=chunk_text[:MAX_CHUNK_CHARS_FOR_PROMPT]
    )

    response = client.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options={
            "temperature": 0.3,
            "num_predict": 100,
        },

        keep_alive="10m"
    )

    question = response["message"]["content"].strip()

    # Strip stray wrapping quotes some models add.
    question = question.strip('"').strip("'").strip()

    return question


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Auto-generate an offline eval question set from "
            "the currently indexed corpus, so every indexed "
            "document gets test coverage without hand-writing "
            "questions for it."
        )
    )

    parser.add_argument(
        "--per-document",
        type=int,
        default=3,
        help="How many questions to generate per document (default: 3)."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Where to write the generated questions.json. "
            "Defaults to evaluation/generated_questions.json."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling (default: 42)."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_PATH
    )

    print()
    print("=" * 60)
    print("Synthetic Offline-Eval Question Generation")
    print("=" * 60)

    chunks = load_chunks()

    grouped = group_by_source(chunks)

    if not grouped:

        print()
        print("ERROR: No documents found in the current index.")

        return

    print()
    print(
        f"Found {len(grouped)} indexed document(s). "
        f"Generating up to {args.per_document} question(s) each..."
    )

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    generated = []

    for source, doc_chunks in sorted(grouped.items()):

        sample = pick_sample_chunks(
            doc_chunks,
            args.per_document,
            args.seed
        )

        print()
        print(f"{source} — generating {len(sample)} question(s)")

        for chunk in sample:

            try:

                question = generate_question(
                    client,
                    chunk["text"]
                )

            except Exception as error:

                print(
                    f"  ERROR generating question: {error}"
                )

                continue

            if not question:
                continue

            print(
                f"  - {question}"
            )

            generated.append({
                "question": question,

                # The chunk this question was generated from —
                # the ground-truth label, for free.
                "relevant_sources": [source],
            })

    if not generated:

        print()
        print("ERROR: No questions were generated.")

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            generated,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(
        f"Wrote {len(generated)} generated question(s) to:"
    )
    print(
        f"  {output_path}"
    )
    print()
    print("Run the evaluation against it with:")
    print(
        f"  python scripts/evaluate.py --questions "
        f"{output_path.relative_to(ROOT)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()