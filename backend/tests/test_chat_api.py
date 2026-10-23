"""Chat API: an answer's diagrams and figure citations reach the client, and
diagram files are served from data/diagrams only."""
import json
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

import api.openai_compat as openai_compat
from api.schemas import Citation
from rag.intent import Intent

DOC = "cdbdf9f5-8397-485f-a095-563fa702be5f"
IMAGE = f"diagrams/{DOC}/fig9_1.png"
QUESTION = "How do I lift an unpacked machine?"
ANSWER = (
    "1. Lift the machine by crane from the lifting eyes. [1]\n"
    f"![Figure 9 Lifting of unpacked machines]({IMAGE})\n"
    "2. Do not use a forklift. [1]"
)
CITATIONS = [
    {"number": 1, "document_id": DOC, "document_name": "ABB Manual", "chunk_index": 0, "bm25_score": 1.2,
     "page": 25, "section_path": "2.2.3", "chunk_type": "figure", "image_path": IMAGE,
     "figure_caption": "Figure 9 Lifting of unpacked machines"},
    {"number": 2, "document_id": DOC, "document_name": "ABB Manual", "chunk_index": 1, "bm25_score": 0.4,
     "page": 23, "section_path": None, "chunk_type": None, "image_path": None, "figure_caption": None},
]


class FakeChain:
    async def ainvoke(self, inputs, config=None):
        assert "image path" in inputs["context"]
        assert inputs["question"] == QUESTION  # the user's words, not the retrieval rewrite
        return ANSWER

    async def astream(self, inputs, config=None):
        assert inputs["question"] == QUESTION
        for i in range(0, len(ANSWER), 7):  # arrives in pieces, splitting the image Markdown
            yield ANSWER[i:i + 7]


@pytest.fixture
def fake_rag(monkeypatch):
    async def classify(query, metadata=None):
        return Intent(intent="knowledge", topic="Unpacked machine lifting")

    async def prepare(query, metadata=None):
        context = f"[1] ABB Manual, p. 25\nLift by crane.\n\nFigures for this passage:\n- Figure 9 — image path: {IMAGE}"
        return context, [dict(c) for c in CITATIONS], "lift unpacked machine"

    monkeypatch.setattr(openai_compat, "classify_intent", classify)
    monkeypatch.setattr(openai_compat, "prepare_context", prepare)
    monkeypatch.setattr(openai_compat, "build_answer_chain", lambda: FakeChain())


async def test_answer_keeps_its_diagram_and_only_the_cited_figure(fake_rag):
    answer, citations, topic = await openai_compat._generate_answer(QUESTION)
    assert f"]({IMAGE})" in answer
    assert [c["number"] for c in citations] == [1]
    assert citations[0]["image_path"] == IMAGE
    assert citations[0]["figure_caption"] == "Figure 9 Lifting of unpacked machines"
    assert topic == "Unpacked machine lifting"


async def test_streamed_answer_reassembles_the_diagram_and_sends_figure_citations(fake_rag, monkeypatch):
    saved = {}

    async def persist(db, conversation, query, answer, citations, topic=""):
        saved.update(answer=answer, citations=citations)
        return "msg-1"

    monkeypatch.setattr(openai_compat, "_persist_exchange", persist)
    conversation = type("C", (), {"id": "conv-1", "persona": "engineer"})()

    events = [e async for e in openai_compat._stream_sse(QUESTION, conversation, db=None)]
    chunks = [json.loads(e[6:]) for e in events if e.startswith("data: {")]
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)

    assert streamed == ANSWER == saved["answer"]
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert [c["image_path"] for c in final["citations"]] == [IMAGE]
    assert saved["citations"] == final["citations"]


def test_history_schema_returns_the_figure_caption():
    out = Citation(**CITATIONS[0]).model_dump()
    assert out["figure_caption"] == "Figure 9 Lifting of unpacked machines"
    assert out["image_path"] == IMAGE
    # Citations stored before figure_caption existed still load.
    legacy = {k: v for k, v in CITATIONS[0].items() if k != "figure_caption"}
    assert Citation(**legacy).figure_caption is None


@pytest.fixture
def client_with_diagram():
    from api.main import DATA_DIR, app

    folder = DATA_DIR / "diagrams" / f"__pytest_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    (folder / "fig1_1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
    try:
        yield TestClient(app), f"diagrams/{folder.name}/fig1_1.png"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_diagram_files_are_served_as_png(client_with_diagram):
    client, path = client_with_diagram
    response = client.get(f"/data/{path}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_missing_diagram_is_404(client_with_diagram):
    client, _ = client_with_diagram
    assert client.get("/data/diagrams/nope/fig1_1.png").status_code == 404


@pytest.mark.parametrize("path", ["/data/../.env", "/data/%2e%2e/.env", "/data/..%2f.env", "/data/diagrams/../../.env"])
def test_static_mount_does_not_escape_the_data_folder(client_with_diagram, path):
    client, _ = client_with_diagram
    response = client.get(path)
    assert response.status_code == 404 or "OPENAI_API_KEY" not in response.text
