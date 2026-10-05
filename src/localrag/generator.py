import ollama

from .config import (
    OLLAMA_HOST,
    LLM_MODEL
)


client = ollama.Client(
    host=OLLAMA_HOST
)


SYSTEM_PROMPT = """
You are a grounded document question-answering assistant.

Answer using ONLY the provided retrieved context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context is insufficient, say that the information is not available in the provided documents.
4. Keep the answer concise and useful.
5. Cite important claims using:
   [Source: filename, page X]
"""


# Limit how much retrieved material reaches the LLM
MAX_CONTEXT_CHUNKS = 3
MAX_CHARS_PER_CHUNK = 3500


def build_context(retrieved_results):

    context_parts = []

    # Only send the top reranked chunks
    selected_results = retrieved_results[:MAX_CONTEXT_CHUNKS]

    for result in selected_results:

        chunk = result["chunk"]

        source = chunk["source"]
        page = chunk.get("page")

        if page is not None:
            citation = (
                f"[Source: {source}, page {page}]"
            )
        else:
            citation = (
                f"[Source: {source}]"
            )

        # Prevent very large chunks from reaching Ollama
        text = chunk["text"][:MAX_CHARS_PER_CHUNK]

        context_parts.append(
            f"{citation}\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question,
    retrieved_results
):
    """
    Generate a grounded response using Ollama.
    """

    if not retrieved_results:
        return (
            "The information is not available "
            "in the provided documents."
        )

    context = build_context(
        retrieved_results
    )

    response = client.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n\n"
                    f"{context}\n\n"
                    f"User question:\n\n"
                    f"{question}\n\n"
                    f"Answer using only the retrieved context."
                )
            }
        ],

        options={
            "temperature": 0.1,

            # Prevent unnecessarily long answers
            "num_predict": 300,

            # Keep the context window reasonably small
            "num_ctx": 4096,
        },

        # Keep model loaded between questions
        keep_alive="10m"
    )

    return response["message"]["content"].strip()