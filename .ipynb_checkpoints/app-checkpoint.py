import sys
import time
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(
    __file__
).resolve().parent

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT)
    )


# ============================================================
# LOCALRAG IMPORTS
# ============================================================

from src.localrag.config import (
    FAISS_PATH,
    CHUNKS_PATH,
    BM25_PATH,
    EMBED_MODEL,
    LLM_MODEL,
)

from src.localrag.ingestion import (
    ingest_uploaded_pdfs,
    clear_previous_corpus,
    run_startup_cleanup_once,
)

from src.localrag.retriever import (
    retrieve,
)

from src.localrag.generator import (
    generate_answer,
)
from src.localrag.vector_store import (
    load_faiss_index
)

from src.localrag.bm25_store import (
    load_bm25_index
)

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind",
    layout="wide",
)


# ============================================================
# ONE-TIME STARTUP CLEANUP
# ============================================================
# Wipes any stale/partial data left behind by a previous crashed
# run. Guarded by a marker file on disk (see run_startup_cleanup_
# once), not @st.cache_resource — cache_resource keys on the
# cached function's bytecode, so it re-fires (and re-wipes your
# corpus) every time this file changes during development. The
# marker file survives hot-reloads and app restarts. Ingestion
# itself is additive from here on and never wipes the corpus.

run_startup_cleanup_once()


# ============================================================
# SESSION STATE
# ============================================================

if "corpus_ready" not in st.session_state:

    st.session_state.corpus_ready = False


if "corpus_stats" not in st.session_state:

    st.session_state.corpus_stats = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def indexes_exist():

    return (
        FAISS_PATH.exists()
        and CHUNKS_PATH.exists()
        and BM25_PATH.exists()
    )


@st.cache_resource
def load_retrieval_system():

    """
    Load FAISS index, chunks and BM25 index.

    Streamlit caches this so we don't repeatedly reload
    the retrieval system for every question.
    """
    index, chunks = load_faiss_index()
    bm25 = load_bm25_index()

    return index,chunks,  bm25


def clear_index_cache():

    """
    Force Streamlit to reload the newly generated indexes.
    """

    load_retrieval_system.clear()


def process_uploads(uploaded_files):

    """
    Run the complete ingestion pipeline and show
    progress in the Streamlit UI.
    """

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    def progress_callback(
        value,
        message
    ):

        progress_bar.progress(
            int(value * 100)
        )

        status_text.info(
            message
        )

    try:

        stats = ingest_uploaded_pdfs(
            uploaded_files,
            progress_callback=progress_callback
        )

        has_new_content = bool(stats.get("new_documents"))

        if has_new_content:

            # Important:
            # The FAISS/BM25 files have changed, so remove
            # the cached old retrieval system.
            clear_index_cache()

            st.session_state.messages = []

        st.session_state.corpus_ready = True

        st.session_state.corpus_stats = stats

        progress_bar.progress(
            100
        )

        if has_new_content:

            status_text.success(
                "Document Corpus built successfully."
            )

        else:

            status_text.info(
                "Nothing new to add — those PDFs are already "
                "indexed."
            )

        return True

    except Exception as error:

        progress_bar.empty()

        status_text.error(
            f"Ingestion failed: {error}"
        )

        return False


# ============================================================
# HEADER
# ============================================================

st.title(
    "DocuMind AI"
)

st.caption(
    """
DocuMind AI is a Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over your documents, using natural language with AI-powered responses backed by actual document sources.
"""
)

st.divider()
# ============================================================
# SIDEBAR
# ============================================================





with st.sidebar:
    st.markdown("### Documents")
    if indexes_exist():
        stats = st.session_state.corpus_stats
        if stats:
            st.caption(f"{stats['documents']} documents")
            st.caption(f"{stats['chunks']} chunks")
        else:
            st.caption("Documents are indexed")
    else:
        st.caption("No documents indexed")


    st.divider()
    with st.expander("System information"):
        st.markdown("**Embedding model**")
        st.code(EMBED_MODEL, language=None)
        st.markdown("**Language model**")
        st.code(LLM_MODEL, language=None)
        st.caption("Retrieval uses semantic and lexical search with reranking.")
    
    st.divider()
    st.caption("Local inference")
    st.caption("Your documents and queries are processed through the local application.")

    st.divider()
    st.markdown("### Corpus management")
    st.caption(
        "New uploads are added to what's already indexed. "
        "Use this to start over from scratch instead."
    )
    if st.button("🗑️ Clear entire corpus", use_container_width=True):
        clear_previous_corpus()
        clear_index_cache()
        st.session_state.corpus_ready = False
        st.session_state.corpus_stats = None
        st.session_state.messages = []
        st.rerun()
    
    





# ============================================================
# STEP 1 — UPLOAD
# ============================================================

st.header(
    "Upload Documents"
)

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "Upload research papers, lecture notes, reports, "
        "or other text-based PDFs."
    ),
)


# ============================================================
# SHOW SELECTED FILES
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) selected."
    )

    with st.expander(
        "📄 Selected documents",
        expanded=True
    ):

        for uploaded_file in uploaded_files:

            size_mb = (
                uploaded_file.size
                / (1024 * 1024)
            )

            st.write(
                f"**{uploaded_file.name}** "
                f"— {size_mb:.2f} MB"
            )


# ============================================================
# STEP 2 — INGESTION
# ============================================================

if uploaded_files:

    st.header(
        "Build Corpus"
    )

    st.write(
        "Click the button below to add these PDFs to your "
        "corpus. This adds to what's already indexed — it "
        "won't remove documents you've already uploaded."
    )


    build_button = st.button(
        "Build Corpus",
        type="primary",
        use_container_width=True,
    )

    if build_button:

        successful = process_uploads(
            uploaded_files
        )

        if successful:

            st.rerun()


# ============================================================
# STEP 3 — CORPUS STATUS
# ============================================================

if indexes_exist():

    st.header(
        "Corpus"
    )

    stats = (
        st.session_state.corpus_stats
    )

    if stats:

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Documents (total)",
                stats["documents"],
                delta=(
                    f"+{stats['new_documents']} this build"
                    if stats.get("new_documents")
                    else None
                )
            )

        with col2:

            st.metric(
                "Chunks (total)",
                stats["chunks"],
                delta=(
                    f"+{stats['new_chunks']} this build"
                    if stats.get("new_chunks")
                    else None
                )
            )

        with col3:

            st.metric(
                "New pages this build",
                stats.get("new_pages", 0)
            )

        with col4:

            st.metric(
                "Processing time",
                f"{stats['processing_time']:.1f}s"
            )

        if stats.get("skipped_files"):

            st.info(
                "Already indexed, skipped: "
                + ", ".join(stats["skipped_files"])
            )

        with st.expander(
            "Indexed documents"
        ):

            for filename in stats["files"]:

                st.write(
                    f"📄 {filename}"
                )

    else:
    
        st.caption(
            "No documents have been uploaded yet."
        )


# ============================================================
# STEP 4 — QUESTION ANSWERING
# ============================================================

st.header(
    "Query your documents"
)


if not indexes_exist():

    st.info(
        "Upload documents and build a corpus "
        "to start asking questions."
    )

else:

    # --------------------------------------------------------
    # LOAD RETRIEVAL SYSTEM
    # --------------------------------------------------------

    try:

        index, chunks, bm25 = (
            load_retrieval_system()
        )

    except Exception as error:

        st.error(
            f"Unable to load retrieval index: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # Display evidence for assistant answers
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "Retrieved evidence"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        chunk = source["chunk"]

                        st.markdown(
                            f"### Evidence {i}"
                        )

                        st.markdown(
                            f"**{chunk['source']}**"
                        )

                       
                        st.write(
                            chunk["text"]
                        )


    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your documents..."
    )


    if question:

        # ----------------------------------------------------
        # STORE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            status = st.empty()

            # These stay populated no matter which branch below
            # runs, so the exchange is always saved to chat
            # history instead of vanishing when something fails.
            answer = None
            sources = []
            retrieval_failed = False


            # =================================================
            # RETRIEVAL
            # =================================================

            status.info(
                "🔎 Searching the research corpus..."
            )

            retrieval_start = (
                time.perf_counter()
            )

            try:

                results = retrieve(
                    question,
                    index,
                    chunks,
                    bm25
                )

            except Exception as error:

                status.empty()

                st.error(
                    f"Retrieval failed: {error}"
                )

                answer = f"Retrieval failed: {error}"

                retrieval_failed = True

                results = []

            retrieval_time = (
                time.perf_counter()
                - retrieval_start
            )


            if not retrieval_failed and not results:

                status.empty()

                st.warning(
                    "No relevant evidence was found."
                )

                answer = (
                    "No relevant evidence was found in the "
                    "research corpus for this question."
                )


            # =================================================
            # GENERATION
            # =================================================

            if not retrieval_failed and results:

                status.info(
                    "🤖 Generating an evidence-grounded answer..."
                )

                generation_start = (
                    time.perf_counter()
                )

                try:

                    answer = generate_answer(
                        question,
                        results
                    )

                except Exception as error:

                    status.empty()

                    st.error(
                        f"Generation failed: {error}"
                    )

                    answer = f"Generation failed: {error}"

                    results = []

                generation_time = (
                    time.perf_counter()
                    - generation_start
                )

                status.empty()


                # =============================================
                # ANSWER
                # =============================================

                st.markdown(
                    answer
                )

                st.caption(
                    f"Retrieval + reranking: "
                    f"{retrieval_time:.2f}s | "
                    f"Generation: "
                    f"{generation_time:.2f}s"
                )


                # =============================================
                # SOURCES
                # =============================================

                for result in results[:3]:

                    sources.append({
                        "chunk": result["chunk"],
                        "rrf_score": result["rrf_score"],
                        "rerank_score": result["rerank_score"]
                    })


                if sources:

                    st.subheader(
                        "📑 Sources"
                    )

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        chunk = source["chunk"]

                        title = chunk[
                            "source"
                        ]

                        page = chunk.get(
                            "page"
                        )

                        page_label = (
                            f" — Page {page}"
                            if page is not None
                            else ""
                        )

                        st.markdown(
                            f"**[{i}] {title}**"
                            f"{page_label}"
                        )


                    # ---------------------------------------------
                    # FULL EVIDENCE
                    # ---------------------------------------------

                    with st.expander(
                        "View retrieved passages"
                    ):

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            chunk = source["chunk"]

                            page = chunk.get(
                                "page"
                            )

                            page_label = (
                                f" — Page {page}"
                                if page is not None
                                else ""
                            )

                            st.markdown(
                                f"### [{i}] "
                                f"{chunk['source']}"
                                f"{page_label}"
                            )

                            st.write(
                                chunk["text"]
                            )


            # =================================================
            # STORE ASSISTANT MESSAGE
            # =================================================
            # Always stored, even on failure or empty results,
            # so the chat history stays in sync with what the
            # user actually saw.

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })