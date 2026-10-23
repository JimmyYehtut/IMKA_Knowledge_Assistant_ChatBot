"""Query time: how retrieved figure passages reach the answer prompt and the citations."""
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import rag.pipeline as pipeline

DOC = "cdbdf9f5-8397-485f-a095-563fa702be5f"


def figure_chunk(number: int, caption: str, page: int, panels: int = 1) -> Document:
    return Document(
        page_content=caption,
        metadata={
            "document_id": DOC,
            "document_name": "ABB_Manual.pdf",
            "chunk_type": "figure",
            "figure_number": number,
            "figure_caption": caption,
            "image_paths": [f"diagrams/{DOC}/fig{number}_{k}.png" for k in range(1, panels + 1)],
            "page_no": page,
            "section_path": "2.2.3 Lifting an unpacked machine",
        },
    )


def text_chunk(text: str, page: int, figure_refs=None) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": DOC,
            "document_name": "ABB_Manual.pdf",
            "chunk_type": "text",
            "page_no": page,
            "figure_refs": figure_refs or [],
        },
    )


class FakeQdrant:
    """Answers the one scroll() call _figures_for_chunks makes."""

    def __init__(self, figures: list[Document], fail: bool = False):
        self.figures, self.fail, self.calls = figures, fail, []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("qdrant down")
        points = [SimpleNamespace(payload={"metadata": f.metadata}) for f in self.figures]
        return points, None


@pytest.fixture
def qdrant(monkeypatch):
    def install(figures, fail=False):
        fake = FakeQdrant(figures, fail)
        monkeypatch.setattr(pipeline, "get_qdrant_client", lambda: fake)
        return fake

    return install


def test_figure_passage_offers_its_own_images(qdrant):
    fake = qdrant([])
    chunk = figure_chunk(9, "Figure 9 Lifting of unpacked machines", 25, panels=3)
    [figures] = pipeline._figures_for_chunks([chunk])
    assert figures == [{
        "number": 9,
        "caption": "Figure 9 Lifting of unpacked machines",
        "image_paths": [f"diagrams/{DOC}/fig9_{k}.png" for k in (1, 2, 3)],
    }]
    assert fake.calls == []  # no lookup needed


def test_text_passage_offers_the_figures_it_mentions(qdrant):
    fake = qdrant([figure_chunk(9, "Figure 9 Lifting of unpacked machines", 25),
                   figure_chunk(20, "Figure 20 Typical foundation stud assembly", 40)])
    chunk = text_chunk("Lift the machine as shown in Figure 9.", 25, figure_refs=[9, 99])
    [figures] = pipeline._figures_for_chunks([chunk])
    assert [f["number"] for f in figures] == [9]  # 99 does not exist, 20 is not mentioned
    assert len(fake.calls) == 1


def test_figure_lookup_failure_still_answers_without_diagrams(qdrant):
    qdrant([], fail=True)
    chunk = text_chunk("See Figure 9.", 25, figure_refs=[9])
    assert pipeline._figures_for_chunks([chunk]) == [[]]


def test_context_lists_each_figure_with_its_exact_image_path(qdrant):
    qdrant([figure_chunk(9, "Figure 9 Lifting of unpacked machines", 25)])
    chunks = [text_chunk("Lift by crane, see Figure 9.", 25, figure_refs=[9]),
              text_chunk("Check the weight.", 23)]
    context = pipeline._stage4_assemble_context(chunks, pipeline._figures_for_chunks(chunks))

    first, second = context.split("\n\n---\n\n")
    assert first.startswith("[1] ABB Manual, p. 25")
    assert f"Figure 9 Lifting of unpacked machines — image path: diagrams/{DOC}/fig9_1.png" in first
    assert second.startswith("[2] ABB Manual, p. 23")
    assert "image path" not in second


def test_query_rewrite_stays_in_the_equipment_domain():
    captured = {}

    class FakeLLM:
        def invoke(self, prompt_value, config=None, **kwargs):
            captured["messages"] = prompt_value.to_messages()
            from langchain_core.messages import AIMessage
            return AIMessage(content=" check carbon brush pressure on slip rings ")

    from langchain_core.runnables import RunnableLambda
    llm = RunnableLambda(lambda pv, config=None: FakeLLM().invoke(pv))
    rewritten = pipeline._stage1_process_query("How do I check the brush pressure?", llm)
    system = captured["messages"][0].content
    assert rewritten == "check carbon brush pressure on slip rings"
    assert "industrial equipment" in system and "never change the subject" in system


def test_answer_prompt_tells_the_model_how_to_embed_diagrams():
    template = pipeline._RAG_TEMPLATE
    assert "Figures for this passage" in template
    assert "![Figure 7 Lifting the machine](diagrams/" in template
    assert "never" in template and "image path that is not in the context" in template


def test_citation_of_a_figure_carries_its_image_and_caption():
    chunks = [text_chunk("Check the weight.", 23),
              figure_chunk(38, "Figure 38 Correlation between the insulation resistance and the temperature", 91)]
    citations = pipeline._stage6_build_citations(chunks)

    assert citations[0]["image_path"] is None and citations[0]["figure_caption"] is None
    assert citations[1] == {
        "number": 2,
        "document_id": DOC,
        "document_name": "ABB Manual",
        "chunk_index": 1,
        "bm25_score": None,
        "page": 91,
        "section_path": "2.2.3 Lifting an unpacked machine",
        "chunk_type": "figure",
        "image_path": f"diagrams/{DOC}/fig38_1.png",
        "figure_caption": "Figure 38 Correlation between the insulation resistance and the temperature",
    }


def test_only_citations_used_in_the_answer_are_kept():
    citations = [{"number": n} for n in (1, 2, 3, 4)]
    answer = "Lift by crane [1][3].\n\n![Figure 9 Lifting](diagrams/x/fig9_1.png)\n\nSee also [2, 3]."
    assert [c["number"] for c in pipeline.select_cited(answer, citations)] == [1, 2, 3]


def test_rerank_keeps_a_compact_table_that_holds_the_answer():
    prose = text_chunk("lifting " * 60 + "the machine lifting eyes lifting lugs lifting", 23)
    table = Document(page_content="| Frame size | Lifting |\n|---|---|\n| 560-1000 | shackles |",
                     metadata={"document_id": DOC, "chunk_type": "table"})
    others = [text_chunk(f"unrelated passage {i}", 30 + i) for i in range(6)]
    ranked = pipeline._stage3_rerank("lifting shackles frame size 560", [table, prose, *others])
    assert table in ranked[:2]
