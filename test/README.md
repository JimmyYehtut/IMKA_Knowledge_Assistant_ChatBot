# IMKA Knowledge Assistant — RAG evaluation (test-A … test-H)

This folder holds the code and the complete results of the IMKA retrieval-augmented generation (RAG) evaluation.
The plan, the hypotheses and the discussion of the results are in [../report/notes.md](../report/notes.md).

## What is evaluated

The assistant answers maintenance questions from five equipment manuals (ABB induction motors and generators,
Siemens vertical induction motors, Kohler KD27V12 generator, EMSA diesel generator, and the Nidec U.S. MOTORS
product service manual). Each run asks **170 questions** — 150 answerable from one of the
PDFs, 20 that none of them answers — against all five PDFs at once, **three times**, and scores retrieval (no LLM)
and the answers (Claude as judge).

Three ingestion configurations, each adding one technique:

| Configuration | Ingestion |
| --- | --- |
| C1 | standard pipeline: MarkItDown + 1,000-character splitter |
| C2 | Docling layout-aware, table-atomic chunking |
| C3 | C2 + a vision-model description of every figure |

Each model does everything in its own stack — the model that answers also writes the C3 figure descriptions:

| Answering + captioning model | C1 | C2 | C3 |
| --- | --- | --- | --- |
| gpt-4o-mini | test-A | test-B | test-C |
| gpt-5.6-luna | test-D | test-E | test-F |
| Qwen3.8-27B, fully local (own embeddings) | – | – | test-G |
| *hybrid strategy: luna for all ingestion, gpt-4o-mini answering* | | | test-H |

## Folder layout

```text
test/
  code/
    imka_eval.py       runs, comparisons, backend client, retrieval metrics, Claude judge, bootstrap
    run.py             command line: questions | documents | answer | judge | report
    notebooks.py       writes and executes output/<run>/evaluate.ipynb and output/report.ipynb
    ingest.sh          ingests one configuration for all five PDFs, a log per PDF
    questions.json     the 170 questions with reference answers and source pages
    requirements.txt   the evaluation venv (test/.venv)
    .env.example       IMKA login for the evaluation account
  output/
    report.ipynb       every run and every comparison side by side (executed)
    test-<X>/
      documents.json   the five document_ids the run searched (pinned)
      answers-r1..3.json  every answer with its retrieved chunks, per repeat
      judge-r1..3.json    every score, per repeat, with the Claude spend
      answer.log, judge.log
      evaluate.ipynb   that run's results (executed)
    ingest/            ingestion logs, one per stage and PDF, and status.log with timings
```

## Results

The executed notebooks show the results; `output/report.ipynb` has the comparisons. Summary — (pending until all
runs are done).

## Reproducing

**Requirements.** The IMKA backend running (`cd backend && uvicorn api.main:app --port 8000`) with Qdrant and
Postgres (`docker compose up -d`); `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `backend/.env`; for test-G the
RunPod Qwen endpoint (`RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`).

```bat
python -m venv test\.venv
test\.venv\Scripts\python.exe -m pip install -r test\code\requirements.txt
copy test\code\.env.example test\code\.env      & REM then fill in IMKA_EMAIL / IMKA_PASSWORD
```

**1. Ingest** (Git Bash, repository root). With no PDF named, a stage ingests all five; from scratch:

```bash
bash test/code/ingest.sh C1          # test-A, test-D
bash test/code/ingest.sh C2          # test-B, test-E
bash test/code/ingest.sh C3-mini     # test-C
bash test/code/ingest.sh C3-luna     # test-F, test-H
bash test/code/ingest.sh C3-qwen     # test-G (Qwen descriptions and embeddings, collection ragindex_qwen)
```

This evaluation reused the other four manuals' documents of 23 Oct 2026, so it ran only:

```bash
for s in C1 C2 C3-luna C3-qwen; do bash test/code/ingest.sh $s "Nidec Motor Manual.pdf"; done
bash test/code/ingest.sh C3-mini
```

`python run.py documents <run>` shows the five documents a run will use and pins them in
`output/<run>/documents.json`.

**2. Answer.** Set the answering model in `backend/.env`, restart the backend, then from `test/code`:

| Runs | `backend/.env` |
| --- | --- |
| test-A, B, C, H | `LLM_PROVIDER="openai"`, `OPENAI_CHAT_MODEL="gpt-4o-mini"`, `EMBEDDING_PROVIDER="openai"`, `QDRANT_COLLECTION_NAME="ragindex"` |
| test-D, E, F | as above with `OPENAI_CHAT_MODEL="gpt-5.6-luna"` |
| test-G | `LLM_PROVIDER="qwen"`, `EMBEDDING_PROVIDER="qwen"`, `QDRANT_COLLECTION_NAME="ragindex_qwen"` |

`OPENAI_TEMPERATURE` stays unset (each model's default).

```bat
..\.venv\Scripts\python.exe run.py answer test-F
..\.venv\Scripts\python.exe run.py answer test-G --concurrent 5
```

Every answer is checked against the run's answering and embedding model before it is saved, so a backend on the
wrong settings stops the run. Every step resumes where it stopped.

**3. Judge** (Claude; needs no backend):

```bat
..\.venv\Scripts\python.exe run.py judge test-F --repeats 1 --check
..\.venv\Scripts\python.exe run.py judge test-F --repeats 2 3
```

| Metric | Judge |
| --- | --- |
| answer accuracy | Sonnet 5.5 |
| faithfulness, answer relevancy, context recall | Haiku 5.5 |
| refused (out-of-scope questions) | Haiku 5.5 |
| context recall, second opinion (`--check`, repeat 1 of test-F and test-G) | Sonnet 5.5 |

**4. Notebooks and report:**

```bat
..\.venv\Scripts\python.exe notebooks.py --all --report
..\.venv\Scripts\python.exe run.py report
```

## Provenance

Every run is answered and judged on the five-manual corpus; none is carried over from the earlier evaluation of
23 Oct 2026 ([../test-7](../test-7)), which used the Keljik *AC/DC Motors* textbook in place of the Nidec manual
and included NER.

Documents: for the ABB, Siemens, Kohler and EMSA manuals, the C1, C2, C3-luna and C3-Qwen documents ingested on
23 Oct 2026 are reused, and only the Nidec manual is ingested for those configurations; C3-mini is ingested for all
five. test-G's ABB, Siemens, Kohler and EMSA documents were ingested with Qwen NER, then had the entities removed
from every chunk (they never touched chunk text or embeddings), which leaves C3 documents with Qwen figure
descriptions.

Ingestion logs in `output/ingest/`: the reused documents' logs of 23 Oct 2026 (`C1-*`, `C2-*`, `C3-*` for luna
descriptions, `C4q-*` for test-G's Qwen documents) with their timings in `status-2026-10-23.log`; this evaluation's
ingestions (Nidec for each configuration, and `C3-mini`) are added with their timings in `status.log`.
