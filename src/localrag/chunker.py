import re


def normalize_text(text):
    """
    Normalize whitespace.
    """

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text,
    chunk_size=800,
    overlap=120
):
    """
    Split text into overlapping word-based chunks.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "chunk overlap must be smaller than chunk size"
        )

    text = normalize_text(text)

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_documents(
    documents,
    chunk_size=800,
    overlap=120
):
    """
    Chunk loaded documents while preserving metadata.
    """

    chunks = []

    chunk_id = 0

    for document in documents:

        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for text_chunk in text_chunks:

            chunks.append(
                {
                    "id": chunk_id,
                    "text": text_chunk,
                    "source": document["source"],
                    "page": document.get("page")
                }
            )

            chunk_id += 1

    return chunks