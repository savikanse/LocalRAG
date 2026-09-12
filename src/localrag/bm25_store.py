import pickle

from rank_bm25 import BM25Okapi

from .config import (
    BM25_PATH,
    INDEX_DIR
)


def tokenize(text):
    """
    Simple BM25 tokenizer.
    """

    return text.lower().split()


def build_bm25_index(chunks):

    if not chunks:
        raise ValueError(
            "No chunks supplied."
        )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        corpus
    )

    with BM25_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            bm25,
            file
        )

    return bm25


def load_bm25_index():

    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_PATH}"
        )

    with BM25_PATH.open(
        "rb"
    ) as file:

        return pickle.load(file)


def bm25_search(
    bm25,
    chunks,
    query,
    k=10
):
    """
    Lexical retrieval using BM25.
    """

    if not chunks:
        return []

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []

    for index in ranked_indices:

        results.append(
            {
                "chunk": chunks[index],
                "score": float(scores[index])
            }
        )

    return results