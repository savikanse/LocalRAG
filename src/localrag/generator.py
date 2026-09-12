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

Your job is to answer questions using ONLY the
provided retrieved context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   explicitly say that the information is not available
   in the provided documents.
4. Keep answers concise but useful.
5. Cite the source supporting each important claim.
6. Use this citation format:

   [Source: filename, page X]

7. For documents without page numbers, use:

   [Source: filename]
"""


def build_context(retrieved_results):

    context_parts = []

    for result in retrieved_results:

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

        context_parts.append(
            f"{citation}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(
        context_parts
    )


def generate_answer(
    question,
    retrieved_results
):
    """
    Generate a grounded response using Ollama.
    """

    context = build_context(
        retrieved_results
    )

    prompt = f"""
{SYSTEM_PROMPT}

Retrieved context:

{context}

User question:

{question}

Answer using only the retrieved context.
Include source citations.
"""

    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]