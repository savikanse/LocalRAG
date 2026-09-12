import sys
import time
from pathlib import Path


import streamlit as st


# Add project root
ROOT = Path(__file__).resolve().parent

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

from src.localrag.reranker import (
    rerank
)

from src.localrag.generator import (
    generate_answer
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="LocalRAG",
    page_icon="📚",
    layout="wide"
)


st.title(
    "📚 LocalRAG"
)

st.caption(
    "Private document intelligence powered by local retrieval and generation."
)


# ============================================================
# CHECK INDEX
# ============================================================

required_files = [
    FAISS_PATH,
    CHUNKS_PATH,
    BM25_PATH
]


if not all(
    path.exists()
    for path in required_files
):

    st.warning(
        """
        No retrieval index found.

        Add PDF/TXT/Markdown files to:

        `data/documents/`

        Then run:

        `python scripts/ingest.py`
        """
    )

    st.stop()


# ============================================================
# LOAD INDEXES
# ============================================================

@st.cache_resource
def load_indexes():

    index, chunks = (
        load_faiss_index()
    )

    bm25 = (
        load_bm25_index()
    )

    return (
        index,
        chunks,
        bm25
    )


index, chunks, bm25 = (
    load_indexes()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System"
    )

    st.write(
        f"Indexed chunks: **{len(chunks)}**"
    )

    st.write(
        "Dense retrieval: **FAISS**"
    )

    st.write(
        "Lexical retrieval: **BM25**"
    )

    st.write(
        "Fusion: **RRF**"
    )

    st.write(
        "Reranking: **Cross-Encoder**"
    )

    st.divider()

    st.info(
        "All LLM inference runs locally through Ollama."
    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_start = (
        time.perf_counter()
    )

    with st.spinner(
        "Searching documents..."
    ):

        fused_results = retrieve(
            question,
            index,
            chunks,
            bm25
        )

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    with st.spinner(
        "Reranking passages..."
    ):

        reranked_results = rerank(
            question,
            fused_results
        )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    generation_start = (
        time.perf_counter()
    )

    with st.spinner(
        "Generating grounded answer..."
    ):

        answer = generate_answer(
            question,
            reranked_results
        )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        st.caption(
            f"Retrieval + reranking: "
            f"{retrieval_time:.2f}s | "
            f"Generation: "
            f"{generation_time:.2f}s"
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        with st.expander(
            "Retrieved sources"
        ):

            for rank, result in enumerate(
                reranked_results,
                start=1
            ):

                chunk = result["chunk"]

                source = (
                    chunk["source"]
                )

                page = (
                    chunk.get("page")
                )

                if page is not None:

                    location = (
                        f"{source}, page {page}"
                    )

                else:

                    location = source

                st.markdown(
                    f"### {rank}. {location}"
                )

                st.write(
                    f"Reranker score: "
                    f"`{result['score']:.4f}`"
                )

                st.write(
                    chunk["text"]
                )