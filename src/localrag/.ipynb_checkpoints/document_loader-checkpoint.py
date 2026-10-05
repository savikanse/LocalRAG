from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md"
}


def load_documents(directory, only_files=None):
    """
    Load PDF, TXT and Markdown documents.

    PDF page numbers are preserved as metadata.

    If only_files is given (an iterable of paths), just those
    files are processed instead of rescanning the whole
    directory — used so re-ingesting doesn't reprocess PDFs
    that are already indexed.
    """

    directory = Path(directory)

    documents = []

    if only_files is not None:
        candidates = sorted(Path(path) for path in only_files)
    else:
        candidates = sorted(directory.rglob("*"))

    for path in candidates:

        if not path.is_file():
            continue

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == ".pdf":

            reader = PdfReader(str(path))

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                text = text.strip()

                if not text:
                    continue

                documents.append(
                    {
                        "source": path.name,
                        "path": str(path),
                        "page": page_number,
                        "text": text
                    }
                )

        # ----------------------------------------------------
        # TXT / MARKDOWN
        # ----------------------------------------------------

        else:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            text = text.strip()

            if not text:
                continue

            documents.append(
                {
                    "source": path.name,
                    "path": str(path),
                    "page": None,
                    "text": text
                }
            )

    return documents