import re
from bisect import bisect_right
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PAGE_BREAK = "\f"
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _page_starts(text: str) -> list[int] | None:
    """Offsets where each page begins, or None for non-paginated sources."""
    if PAGE_BREAK not in text:
        return None
    return [0] + [i + 1 for i, ch in enumerate(text) if ch == PAGE_BREAK]


def _section_path_at(headings: list[tuple[int, int, str]], offset: int) -> str:
    """'H1 > H2 > H3' trail of the Markdown headings in effect at offset."""
    stack: list[tuple[int, str]] = []
    for start, level, title in headings:
        if start > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join(title for _, title in stack)


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split MarkItDown output on Markdown structure (headings, code blocks,
    paragraphs) and tag each chunk with a 1-indexed page_no (PDFs only) and
    the section_path of the headings above it."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = []
    for doc in docs:
        page_starts = _page_starts(doc.page_content)
        # Same-length replacement keeps start_index offsets aligned with page_starts.
        text = doc.page_content.replace(PAGE_BREAK, "\n")
        headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(text)]

        for chunk in splitter.split_documents([Document(page_content=text, metadata=dict(doc.metadata))]):
            start = chunk.metadata.get("start_index", -1)
            if start >= 0:
                if page_starts:
                    chunk.metadata["page_no"] = bisect_right(page_starts, start)
                section_path = _section_path_at(headings, start)
                if section_path:
                    chunk.metadata["section_path"] = section_path
            chunks.append(chunk)
    return chunks
