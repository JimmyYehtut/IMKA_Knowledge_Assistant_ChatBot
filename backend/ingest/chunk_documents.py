import re
from bisect import bisect_right
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PAGE_BREAK = "\f"
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
# Length of the text probes matched against the page index, and the shortest usable one.
_PROBE_CHARS = 40
_MIN_PROBE_CHARS = 16


def _page_starts(text: str) -> list[int] | None:
    """Offsets where each page begins, or None for non-paginated sources."""
    if PAGE_BREAK not in text:
        return None
    return [0] + [i + 1 for i, ch in enumerate(text) if ch == PAGE_BREAK]


def _alnum(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.lower())


class _PdfPageLocator:
    """Finds the page a chunk starts on when the Markdown carries no page breaks.

    MarkItDown only keeps pdfminer's \\f page breaks when it extracts a PDF in one
    pass. As soon as any page looks like a form or table it reads page by page with
    pdfplumber and joins the pages with blank lines, so the breaks are lost. This
    rebuilds the page boundaries from pdfplumber's per-page text, reduced to
    lowercase letters and digits so Markdown table pipes, padding and line wrapping
    do not matter, and looks up a probe from the chunk's start (then middle, then
    end). Lookups move forward through the document with the chunks, so a repeated
    phrase resolves to the occurrence near the previous chunk.
    """

    def __init__(self, path: str):
        import pdfplumber

        parts, self.starts, offset = [], [], 0
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = _alnum(page.extract_text() or "")
                self.starts.append(offset)
                parts.append(text)
                offset += len(text)
                page.close()
        self.index = "".join(parts)
        self.cursor = 0

    def page_of(self, chunk_text: str) -> int | None:
        key = _alnum(chunk_text)
        middle = len(key) // 2
        for probe in (key[:_PROBE_CHARS], key[middle:middle + _PROBE_CHARS], key[-_PROBE_CHARS:]):
            if len(probe) < _MIN_PROBE_CHARS:
                continue
            found = self.index.find(probe, max(0, self.cursor - 5000))
            if found < 0:
                found = self.index.find(probe)
            if found >= 0:
                self.cursor = found
                return bisect_right(self.starts, found)
        return None


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
        locator = None
        if page_starts is None and doc.metadata.get("format") == "pdf" and doc.metadata.get("source"):
            locator = _PdfPageLocator(doc.metadata["source"])
        # Same-length replacement keeps start_index offsets aligned with page_starts.
        text = doc.page_content.replace(PAGE_BREAK, "\n")
        headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(text)]

        for chunk in splitter.split_documents([Document(page_content=text, metadata=dict(doc.metadata))]):
            start = chunk.metadata.get("start_index", -1)
            if start >= 0:
                if page_starts:
                    chunk.metadata["page_no"] = bisect_right(page_starts, start)
                elif locator:
                    page_no = locator.page_of(chunk.page_content)
                    if page_no:
                        chunk.metadata["page_no"] = page_no
                section_path = _section_path_at(headings, start)
                if section_path:
                    chunk.metadata["section_path"] = section_path
            chunks.append(chunk)
    return chunks
