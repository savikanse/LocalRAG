import ollama

from .config import (
    OLLAMA_HOST,
    EMBED_MODEL
)


client = ollama.Client(
    host=OLLAMA_HOST
)


def embed_texts(
    texts,
    batch_size=32
):
    """
    Generate embeddings for multiple documents.
    """

    if not texts:
        return []

    embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        response = client.embed(
            model=EMBED_MODEL,
            input=batch
        )

        embeddings.extend(
            response["embeddings"]
        )

    return embeddings


def embed_query(query):
    """
    Generate embedding for a single query.
    """

    response = client.embed(
        model=EMBED_MODEL,
        input=query
    )

    return response["embeddings"][0]