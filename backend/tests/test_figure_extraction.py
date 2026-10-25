"""Docling figure extraction: which pictures end up in which figure chunk.

Builds small DoclingDocuments in memory that reproduce the layouts of the ABB
manual that went wrong (cover photo glued onto Figure 1, a warning sign from
p. 74 glued onto Figure 33 on p. 81) and the ones that must keep working
(multi-panel figures, captions Docling labels as headings).
"""
import pytest

docling_doc = pytest.importorskip("docling_core.types.doc")
from PIL import Image  # noqa: E402

import ingest.embed_and_store_docling as docling_ingest  # noqa: E402

BoundingBox = docling_doc.BoundingBox
DocItemLabel = docling_doc.DocItemLabel
DoclingDocument = docling_doc.DoclingDocument
ImageRef = docling_doc.ImageRef
ProvenanceItem = docling_doc.ProvenanceItem
Size = docling_doc.Size


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=10, r=10, b=0), charspan=(0, 0))


class _Builder:
    def __init__(self, pages: int):
        self.doc = DoclingDocument(name="test")
        for n in range(1, pages + 1):
            self.doc.add_page(page_no=n, size=Size(width=100, height=100))

    def picture(self, page: int, caption: str | None = None, color: str = "red"):
        image = ImageRef.from_pil(Image.new("RGB", (8, 8), color), dpi=72)
        picture = self.doc.add_picture(prov=_prov(page), image=image)
        if caption:
            text = self.doc.add_text(label=DocItemLabel.CAPTION, text=caption, prov=_prov(page))
            picture.captions.append(text.get_ref())
        return self

    def text(self, page: int, text: str, label=DocItemLabel.TEXT):
        self.doc.add_text(label=label, text=text, prov=_prov(page))
        return self


@pytest.fixture
def extract(tmp_path, monkeypatch):
    monkeypatch.setattr(docling_ingest, "DIAGRAMS_DIR", tmp_path)

    def run(doc):
        chunks = docling_ingest._extract_figures(doc, source_path="manual.pdf", document_id="doc1")
        return {c.metadata["figure_number"]: c for c in chunks}, tmp_path

    return run


def test_uncaptioned_pictures_on_earlier_pages_are_not_glued_onto_the_next_figure(extract):
    doc = (
        _Builder(pages=20)
        .picture(1)  # cover photo
        .picture(1)  # logo
        .picture(5)  # unrelated sketch
        .picture(19, caption="Figure 1 Rating plate for direct on line machines")
        .doc
    )
    figures, _ = extract(doc)

    assert figures["1"].metadata["image_paths"] == ["diagrams/doc1/fig1_1.png"]
    assert figures["1"].metadata["page_no"] == 19


def test_warning_sign_pages_before_the_caption_is_dropped(extract):
    doc = (
        _Builder(pages=81)
        .picture(74)  # magnet warning sign, never captioned
        .text(81, "Figure 33 Measuring points", label=DocItemLabel.CAPTION)
        .doc
    )
    figures, _ = extract(doc)
    assert "33" not in figures  # its only candidate picture is 7 pages away

    doc = _Builder(pages=81).picture(74).picture(81, caption="Figure 33 Measuring points").doc
    figures, _ = extract(doc)
    assert figures["33"].metadata["image_paths"] == ["diagrams/doc1/fig33_1.png"]
    assert figures["33"].metadata["page_no"] == 81


def test_multi_panel_figure_keeps_all_panels_on_the_same_page(extract):
    doc = (
        _Builder(pages=25)
        .picture(25)
        .picture(25)
        .picture(25, caption="Figure 9 Lifting of unpacked machines")
        .doc
    )
    figures, out_dir = extract(doc)

    assert figures["9"].metadata["image_paths"] == [f"diagrams/doc1/fig9_{k}.png" for k in (1, 2, 3)]
    assert all((out_dir / "doc1" / f"fig9_{k}.png").is_file() for k in (1, 2, 3))


def test_picture_at_foot_of_previous_page_joins_caption_on_next_page(extract):
    doc = (
        _Builder(pages=45)
        .picture(44)
        .text(45, "Figure 24 Mounting of the bracket plate", DocItemLabel.SECTION_HEADER)
        .doc
    )
    figures, _ = extract(doc)
    assert figures["24"].metadata["page_no"] == 44
    assert figures["24"].metadata["image_paths"] == ["diagrams/doc1/fig24_1.png"]


def test_picture_at_foot_of_page_leaves_next_page_caption_to_its_own_picture(extract):
    doc = (
        _Builder(pages=45)
        .picture(44)  # warning sign at the foot of p. 44
        .picture(45)
        .text(45, "Figure 24 Mounting of the bracket plate", DocItemLabel.CAPTION)
        .doc
    )
    figures, _ = extract(doc)
    assert figures["24"].metadata["page_no"] == 45
    assert figures["24"].metadata["image_paths"] == ["diagrams/doc1/fig24_1.png"]


def test_caption_labelled_as_heading_still_captions_the_picture_above_it(extract):
    doc = (
        _Builder(pages=91)
        .text(91, "7.6.4.1 Conversion of measured insulation resistance", DocItemLabel.SECTION_HEADER)
        .picture(91)
        .text(91, "Figure 38 Correlation between the insulation resistance and the temperature",
              DocItemLabel.SECTION_HEADER)
        .doc
    )
    figures, _ = extract(doc)
    chunk = figures["38"]
    assert chunk.page_content.startswith("Figure 38 Correlation")
    assert chunk.metadata["chunk_type"] == "figure"
    assert chunk.metadata["figure_caption"] == chunk.page_content
    assert chunk.metadata["section_path"] == "7.6.4.1 Conversion of measured insulation resistance"


class _FakeVLM:
    """Stands in for the OpenAI client: records each request, returns a canned reply."""

    def __init__(self, reply: str, finish_reason: str = "stop"):
        self.requests = []
        message = type("Message", (), {"content": reply})()
        choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
        self.response = type("Response", (), {"choices": [choice]})()
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


@pytest.fixture
def openai_captioner(monkeypatch):
    monkeypatch.setattr(docling_ingest, "VLM_CAPTION_MODEL", "gpt-5.6-luna")


def test_figure_description_is_trimmed_to_the_last_full_sentence_when_cut_off(openai_captioner):
    vlm = _FakeVLM("The graph converts R to R40 with k ≈ 0.25. The manual cautions that", "length")
    text = docling_ingest._describe_image(vlm, Image.new("RGB", (8, 8)))
    assert text == "The graph converts R to R40 with k ≈ 0.25."


def test_complete_figure_description_is_kept_whole(openai_captioner):
    vlm = _FakeVLM("Callout 1 is the terminal box. Callout 2 is the earthing screw.")
    text = docling_ingest._describe_image(vlm, Image.new("RGB", (8, 8)))
    assert text == "Callout 1 is the terminal box. Callout 2 is the earthing screw."
