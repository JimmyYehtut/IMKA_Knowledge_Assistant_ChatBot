# IMKA Knowledge Assistant — RAG evaluation (test-A … test-G)

This folder holds the code and the complete results of the IMKA retrieval-augmented generation (RAG) evaluation.
The plan, the hypotheses and the discussion of the results are in [../report/notes.md](../report/notes.md).

## What is evaluated

The assistant answers maintenance questions from five equipment manuals (ABB induction motors and generators,
Siemens vertical induction motors, Kohler KD27V12 generator, EMSA diesel generator, and the Nidec U.S. MOTORS
product service manual). Each run asks **170 questions** — 150 answerable from one of the PDFs, 20 that none of
them answers — against all five PDFs at once, **three times**, and scores retrieval (no LLM) and the answers
(Claude as judge).

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
| *hybrid strategy: luna for all ingestion, gpt-4o-mini answering* | | | test-G |

## Folder layout

```text
test/
  code/
    imka_eval.py       runs, comparisons, backend client, retrieval metrics, Claude judge, bootstrap, cost
    run.py             command line: questions | documents | answer | judge | report
    notebooks.py       writes and executes output/<run>/evaluate.ipynb and output/report.ipynb
    ingest.sh          ingests one configuration for the five PDFs, a log per PDF
    questions.json     the 170 questions with reference answers and source pages
    requirements.txt   the evaluation venv (test/.venv)
    .env.example       IMKA login for the evaluation account
  output/
    report.ipynb       every run and every comparison side by side (executed)
    report.txt         the same as text (`run.py report`); notebooks.log: the notebook build
    test-<X>/
      documents.json   the five document_ids the run searched (pinned)
      answers-r1..3.json  every answer with its retrieved chunks and model usage, per repeat
      judge-r1..3.json    every score, per repeat, with the Claude spend
      answer.log, judge.log
      evaluate.ipynb   that run's results (executed)
    ingest/            ingestion logs, one per stage and PDF, and status.log with timings
```

## Results

The executed notebooks show the results; `output/report.ipynb` has the comparisons and `output/report.txt` the
same tables as text. The discussion is in [../report/notes.md](../report/notes.md). Means over 3 repeats, 150
in-scope questions:

| Run | Configuration | Answering | Answer accuracy | Figures | Context recall | $ / 1,000 questions | s / question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test-A | C1 | gpt-4o-mini | 0.586 | 0.167 | 0.640 | 0.44 | 4.2 |
| test-B | C2 | gpt-4o-mini | 0.744 | 0.380 | 0.810 | 0.46 | 4.6 |
| test-C | C3, gpt-4o-mini descriptions | gpt-4o-mini | 0.789 | 0.567 | 0.856 | 0.47 | 4.7 |
| test-G | C3, luna descriptions (hybrid) | gpt-4o-mini | 0.799 | 0.700 | 0.881 | 0.47 | 4.6 |
| test-D | C1 | gpt-5.6-luna | 0.622 | 0.280 | 0.656 | 0.95 | 7.4 |
| test-E | C2 | gpt-5.6-luna | 0.788 | 0.347 | 0.811 | 0.95 | 7.4 |
| test-F | C3, luna descriptions | gpt-5.6-luna | 0.855 | 0.727 | 0.878 | 0.98 | 7.4 |

- Docling chunking is the largest gain: context recall on tables + topical +0.17 [0.04, 0.29] (gpt-4o-mini) and
  +0.19 [0.08, 0.30] (luna).
- Figure descriptions lift figure answers by +0.19 (gpt-4o-mini's) to +0.38 (luna's); luna's descriptions beat
  gpt-4o-mini's by +0.13 [0.01, 0.28] with the same answering model.
- The hybrid test-G keeps nearly all of the figure gain (0.70 vs 0.73) at gpt-4o-mini's price; luna's answering
  adds +0.056 [0.020, 0.096] over all in-scope questions at ~2.1× the cost per question.
- 419 of 420 out-of-scope answers were declined. Haiku and Sonnet agreed on context recall for 97% of 300 checked
  answers (mean absolute difference 0.018). Spend: $26.94 Claude (judging), ~$2.40 OpenAI (answering).

## Reproducing

**Requirements.** The IMKA backend running (`cd backend && uvicorn api.main:app --port 8000`) with Qdrant and
Postgres (`docker compose up -d`); `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `backend/.env`.

```bat
python -m venv test\.venv
test\.venv\Scripts\python.exe -m pip install -r test\code\requirements.txt
copy test\code\.env.example test\code\.env      & REM then fill in IMKA_EMAIL / IMKA_PASSWORD
```

**1. Ingest** (Git Bash, repository root). With no PDF named, a stage ingests all five:

```bash
bash test/code/ingest.sh C1          # test-A, test-D
bash test/code/ingest.sh C2          # test-B, test-E
bash test/code/ingest.sh C3-mini     # test-C
bash test/code/ingest.sh C3-luna     # test-F, test-G
```

`python run.py documents <run>` shows the five documents a run will use and pins them in
`output/<run>/documents.json`.

**2. Answer.** Set the answering model in `backend/.env`, restart the backend, then from `test/code`:

| Runs | `backend/.env` |
| --- | --- |
| test-A, B, C, G | `LLM_PROVIDER="openai"`, `OPENAI_CHAT_MODEL="gpt-4o-mini"`, `EMBEDDING_PROVIDER="openai"`, `QDRANT_COLLECTION_NAME="ragindex"` |
| test-D, E, F | as above with `OPENAI_CHAT_MODEL="gpt-5.6-luna"` |

`OPENAI_TEMPERATURE` stays unset (each model's default).

```bat
..\.venv\Scripts\python.exe run.py answer test-F
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
| context recall, second opinion (`--check`, repeat 1 of test-C and test-F) | Sonnet 5.5 |

**4. Notebooks and report:**

```bat
..\.venv\Scripts\python.exe notebooks.py --all --report
..\.venv\Scripts\python.exe run.py report
```

## Provenance

Every run is answered and judged on the five-manual corpus with the current backend; none is carried over from
the earlier evaluation of 23 Oct 2026 ([../test-7](../test-7)), which used the Keljik *AC/DC Motors* textbook in
place of the Nidec manual and included NER. Each `answers-r*.json` records the backend commit it ran on
(`pinned.backend_commit`).

Documents: every document — C1, C2 and C3, all five manuals — is ingested with the current backend on 25 Oct
2026, whose figure matching and figure-description prompt changed after 23 Oct 2026. (The 23 Oct C1 documents
were meant to be reused, but a team merge replaced the document registry, so they were ingested again too.)

Ingestion logs in `output/ingest/`: one per stage and PDF, with their timings in `status.log`;
`status-2026-10-23.log` holds the earlier evaluation's timings for comparison.
