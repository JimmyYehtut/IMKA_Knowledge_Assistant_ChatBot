import os
from typing import List
from langchain_core.documents import Document

# Formats MarkItDown converts to Markdown with the extras installed via
# requirements.txt. Legacy binary .doc is not supported by MarkItDown.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
    ".html", ".htm", ".txt", ".md", ".json", ".xml", ".epub", ".msg",
}

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from markitdown import MarkItDown
        _converter = MarkItDown()
    return _converter


def convert_to_markdown(path: str) -> str:
    """Convert any supported document to Markdown. PDF page boundaries are
    kept as form-feed characters (\\f) so the chunker can recover page numbers."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext} for path: {path}")
    return _get_converter().convert(path).markdown


def load_documents_from_paths(paths: List[str]) -> List[Document]:
    """One Markdown Document per file; files that yield no text are skipped."""
    docs = []
    for path in paths:
        markdown = convert_to_markdown(path)
        if not markdown.strip():
            print(f"⚠️ No text extracted from {path}, skipping.")
            continue
        ext = os.path.splitext(path)[1].lower()
        docs.append(Document(page_content=markdown, metadata={"source": path, "format": ext.lstrip(".")}))
    return docs
