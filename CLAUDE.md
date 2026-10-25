# IMKA Knowledge Assistant

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Contents

- [Project Overview](#project-overview)
- [Common Commands](#common-commands)
- [Architecture](#architecture)
- [Environment Configuration](#environment-configuration)

## Project overview

IMKA Knowledge Assistant ChatBot — a Retrieval-Augmented Generation (RAG) system with a FastAPI backend and a React dashboard/chat frontend. There is also a legacy Streamlit single-page app in `backend/app.py` that combines ingestion + chat in one UI.

```text
backend/     FastAPI API + Streamlit app, RAG pipeline (LangChain, Qdrant, OpenAI, Postgres)
frontend/    React + TypeScript + Vite dashboard/chat UI
```

## Common commands

### Infrastructure (Qdrant vector store + Postgres chat/user DB)

```powershell
cd backend
docker compose up -d
```

- Compose project `imka`: containers `imka-qdrant` and `imka-postgres`, volumes `imka-qdrant-data` and `imka-postgres-data`, `restart: unless-stopped`
- Qdrant: `http://localhost:6333` (dashboard at `/dashboard`)
- Postgres: `localhost:5433`, database `imka`
- Port 6333 is shared with the user's Capstone `mediq-qdrant` container, which must be stopped before `docker compose up -d` will bind

### Backend

```powershell
cd backend
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY etc.

uvicorn api.main:app --reload --port 8000   # FastAPI (serves the React frontend), docs at /docs
streamlit run app.py                         # legacy single-page ingestion+chat UI, http://localhost:8501
```

Direct CLI ingestion (bypasses the UI):

```powershell
python -m ingest.embed_and_store path/to/document1.pdf path/to/document2.docx
python -m ingest.embed_and_store_docling path/to/document.pdf   # Docling pipeline, PDF only
```

Tests and lint (dev tools in `backend/requirements-dev.txt`, config in `backend/pyproject.toml`):

```powershell
pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q                       # unit tests: no network, OpenAI/Qdrant/Postgres faked
python -m evals.run_figure_eval           # live diagram eval: needs the running stack, spends OpenAI tokens
python -m evals.audit_figures <pdf|parse.json> [--document-id ID]   # dry-run figure extraction, no writes
```

`tests/conftest.py` blocks outbound connections and turns LangSmith tracing off. A test that reaches OpenAI fails instead of hanging. Tests that need the stack are marked `live` and run only with `IMKA_LIVE=1`.

RAG quality is evaluated by `evaluate.ipynb` in a numbered test folder (helpers in its `imka_eval.py`, own venv `.venv`, see its `README.md`): per registry `document_id`, it asks that document's questions from `questions.json` through the running backend, then scores retrieval (Hit@K/MRR) and RAGAS (judge `gpt-5.6-luna`); `compare.ipynb` sets C1–C4 side by side. `test-5/` holds the C1–C4 run answered by gpt-4o-mini, `test-6/` the C1–C4 run with Qwen3.8-27B (`LLM_PROVIDER=qwen`) doing ingestion captioning/NER and answering.

`test/` is the current evaluation (the plan and discussion are in `report/notes.md`; layout and how to run in `test/README.md`): runs test-A … test-H — gpt-4o-mini and gpt-5.6-luna each over C1 standard / C2 Docling / C3 + figure descriptions (each model describes its own figures), Qwen3.8-27B fully local at C3, and a cross-check run. A run answers the whole `test/code/questions.json` (170 questions, a `group` each) against one configuration's five documents, three times; `test/code/run.py` (`questions`, `documents`, `answer`, `judge`, `report`) drives `imka_eval.py`, which judges with Claude (Haiku 5.5 for most metrics, Sonnet 5.5 for answer accuracy, via RAGAS + Instructor JSON mode) and compares runs with a paired bootstrap; `notebooks.py` writes the executed per-run notebooks. Outputs: `test/output/<run>/`. `test-7/` is the frozen earlier evaluation (test-7 … test-12, with NER, which has since been removed from the backend); its `README.md` is that report, `HOWTO.md` how it ran.

### Frontend

```powershell
cd frontend
npm install
cp .env.example .env   # VITE_API_URL, default http://localhost:8000

npm run dev       # http://localhost:5173
npm run build      # tsc -b && vite build
npm run lint       # oxlint
npm test           # vitest (jsdom), tests beside the source as *.test.ts(x)
npm run preview
```

### Quality gates and the Claude Code harness

`python scripts/check.py [backend|frontend]` runs ruff, pytest, oxlint, tsc and vitest and prints a summary; CI (`.github/workflows/ci.yml`) runs the same plus `npm run build`.

- `.claude/settings.json`: a PostToolUse hook (`.claude/hooks/post_edit.py`) lints each edited file and runs the tests mapped to it (`BACKEND_TESTS` in the hook, `vitest related` for the frontend). Deny rules block edits to `.env` files and `backend/data/**`, and force-pushes.
- `.claude/agents/`: `rag-eval` (live diagram eval and failure diagnosis), `ingest-checker` (dry-run figure audit of a PDF), `frontend-reviewer` (review of `frontend/src` changes).
- `.claude/commands/`: `/check`, `/eval-diagrams`, `/audit-figures`, `/run-stack`.
- `backend/evals/figures.yaml` is the diagram golden set for the ABB manual. Add a case whenever a diagram question goes wrong.

### Windows convenience scripts

`start.bat` and `start.ps1` at the repository root launch the full stack. Check these scripts before assuming ports or services while debugging startup issues.

## Architecture

### Ingestion Pipelines

Both ingestion pipelines write to the same Qdrant collection and JSON document registry, so results from either pipeline are available together:

- Vector store: `rag.vectorstore.create_or_load_vectorstore`
- Document registry: `ingest/registry.py` -> `backend/data/documents_registry.json`

- **Standard** (`ingest/embed_and_store.py`): converts every file to Markdown with MarkItDown (`ingest/load_documents.py`; `SUPPORTED_EXTENSIONS` there is the single source of truth for allowed types: PDF, DOCX, PPTX, XLSX/XLS, CSV, HTML, TXT, MD, JSON, XML, EPUB, MSG). It then splits the Markdown with the Markdown-aware `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` (`chunk_size=1000`, `chunk_overlap=200`) in `ingest/chunk_documents.py`, and embeds and upserts the chunks. PDF chunks carry a 1-indexed `page_no`, recovered from pdfminer's `\f` page breaks — or, when MarkItDown takes its page-by-page pdfplumber path (any form/table-like page) and joins pages without breaks, by matching each chunk's text against pdfplumber's per-page text (`_PdfPageLocator` in `chunk_documents.py`). Chunks under Markdown headings carry `section_path`. Older chunks ingested by PyPDFLoader may still carry a 0-indexed `page`.
- **Docling** (`ingest/embed_and_store_docling.py`): processes PDF files with Docling's `DocumentConverter` and `HybridChunker` for layout-aware, table-aware, and header-aware chunking. Tables remain atomic and are serialized as compact Markdown tables (not the chunker's default triplets), section headings are tracked as `section_path`, and chunk metadata carries a 1-indexed `page_no`. Table-of-contents chunks and exact duplicate chunks are dropped. Figures: pictures are matched to the nearest figure caption below them on the same page, or else to the caption opening the next page when no picture sits above it there — "Figure 12", "Fig. 27:", "Figure-16a" and "FIGURE 25–1" are all recognized, and `_figure_id` turns the label into a string id ("12", "27", "16a", "25-1") used for `figure_number` and `figure_refs` (documents ingested before 23 Oct 2026 store integers) (`_group_figures`, by layout — Docling's own caption links misassign stacked figures), and every captioned picture is saved to `backend/data/diagrams/<document_id>/figN_k.png` (served at `/data/diagrams/...`, removed when the document is deleted) with one `chunk_type="figure"` chunk per figure (`figure_number`, `figure_caption`, `image_paths`; its text is the caption, then `Legend: ...` — the text lines printed between the picture and its caption, which name bare callout numbers and axes — then the VLM description, which is also given the legend); text chunks record the figure numbers they mention in `figure_refs`. `caption_figures=True` (default from `CAPTION_FIGURES=1` in `.env`, so it applies to both the CLI and uploads) additionally adds a VLM description to each figure chunk that transcribes its content — chart axes and curve readings, rating-plate/table values, callouts and dimensions (`_DESCRIBE_PROMPT`; model `DOCLING_CAPTION_MODEL`, default `gpt-5.6-luna`, chosen because gpt-4o/gpt-4o-mini invented axis titles and misread a log-scale curve; `"qwen"` sends the image to the RunPod Qwen endpoint instead). At query time `rag/pipeline.py` `_figures_for_chunks()` offers each retrieved chunk's figures (caption + image path) to the answer prompt, which embeds relevant ones as Markdown images. In the chat, `lib/diagrams.ts` allows only `diagrams/...` paths, and `DiagramImage.tsx` draws each one with a click-to-enlarge dialog and a caption fallback when the file is missing. `MarkdownContent.tsx` groups the panels of one figure under one caption, and `MessageBubble.tsx` shows cited figures the answer did not draw ("Diagrams cited in this answer").

`POST /api/documents` runs ingestion in a threadpool so the API keeps serving chat during a long ingest.

`POST /api/documents` selects the pipeline through the `pipeline` form field (`"standard"` or `"docling"`). Each pipeline has its own allowed file extensions.

Each registry entry records `pipeline` and the booleans `chunking` (Docling HybridChunker) and `captioning`, plus `models` (the LLM behind captioning, e.g. `{"captioning": "gpt-5.6-luna"}`) and `embedding` (model, dimension, collection), so evaluation runs can tell configurations of the same PDF apart. Entries from before NER was removed (23 Oct 2026) may still carry `ner`, and their chunks `entities`; both are ignored.

Both pipelines attach `document_id`, `document_name`, `version`, and caller-supplied metadata to every chunk. Downstream deletion (`DELETE /api/documents/{id}`) filters Qdrant points by `metadata.document_id`, and citation display relies on the same metadata.

### RAG Pipeline

Implementation: `backend/rag/pipeline.py`

The pipeline has six conceptual stages. Stage 5 is handled separately because it is the LLM streaming call:

1. Query rewrite (LLM cleans/disambiguates the query for semantic search, told it is searching industrial equipment manuals so terms keep their maintenance sense). The rewrite only drives retrieval; stage 5 answers the user's original question.
2. Retrieve — Qdrant similarity search, top `RETRIEVAL_TOP_K` (default 15)
3. Hybrid rerank — dedupe identical chunks, then fuse each candidate's semantic rank with its `rank_bm25` keyword rank by weighted Reciprocal Rank Fusion (`RERANK_SEMANTIC_WEIGHT`, default 2) down to `RERANK_TOP_N` (default 5). BM25 alone must not decide the order: it ranks long prose that repeats query words above a compact table that holds the answer. Tokenization is regex-based (`_tokenize`) so table cells and punctuated words match.
4. Context assembly — chunks numbered and labeled `[n] Document Name, p. N · section path (type)` (page resolved via `_page_number`, which prefers `page_no` over PyPDFLoader's 0-indexed `page`)
5. Answer generation — separate LCEL chain (`build_answer_chain()`), streamed token-by-token via `.astream()`. The prompt makes inline `[n]` citation markers mandatory.
6. Citations — one per reranked chunk, carrying the same `number` as its stage 4 label plus `page`, `section_path`, `chunk_type`, and for figure passages `image_path` and `figure_caption`. After the answer is complete, `select_cited()` (called in `api/openai_compat.py`) keeps only the citations whose `[n]` appears in the answer; that cited-only list is what is persisted and returned. The frontend renders the markers as badges (`MarkdownContent.tsx`) that jump to the source chip under the same message.

Before the pipeline runs, `rag/intent.py` classifies the query as `greeting` / `capability` / `knowledge`; only `knowledge` triggers the full retrieval pipeline — the other two get a short LLM-written reply directly, skipping retrieval.

Everything is wired with LangSmith tracing metadata (`conversation_id`, `user_id`) passed through as `config={"metadata": ...}` on every chain invoke — enable via `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` in `.env`.

### Chat API

`backend/api/openai_compat.py` exposes the following OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions` with optional SSE streaming

The frontend also uses these non-standard extensions:

- Request: `conversation_id`, and `persona` (`technician` | `engineer` | `specialist`). The persona is stored on the `Conversation` when it is created and used for every later answer in it; the value on later requests is ignored. Persona prompt instructions live in `PERSONAS` in `rag/pipeline.py`; the matching cards and the starter questions shown on the new-chat screen live in `frontend/src/lib/personas.ts` (ids must match).
- Evaluation-only request fields: `document_ids` restricts stage 2 retrieval to those documents; `include_context: true` (non-streaming) adds a `rag` object — `intent`, `model`, `rewritten_query`, `candidates` (stage 2), `chunks` (reranked, with `document_id`/`page`/`section_path`/`text`) and `contexts` — filled via the `trace` dict passed to `prepare_context()`.
- Response: `message_id`, `citations`, and `topic` — a 2-6 word label of the user's question produced by the intent classifier (`rag/intent.py`, no extra LLM call), stored on the user `Message` row (`""` for greetings) and listed in the chat page's "Session topics" card.

`GET /api/topics?persona=` (`rag/topics.py`) returns the topic catalogue shown on the new-chat screen: second-level numbered section headings read from chunk metadata in Qdrant, each assigned to personas by one LLM call. The result is cached in `backend/data/topics_cache.json` and rebuilt when the set of document ids in the registry changes.

Every exchange is persisted to Postgres as `Conversation` and `Message` records in `api/db_models.py`. Citations are JSON-encoded. `api/chat.py` only exposes the read path, `GET /api/history`; chat itself always goes through `POST /v1/chat/completions`.

### Authentication

JWT bearer authentication is implemented in `api/auth.py` with `python-jose` and `bcrypt`.

- `get_current_user` decodes the bearer token and is used by the chat and history endpoints.
- Tokens contain `user_id`, `email`, and `name`.
- The default token expiry is 1,440 minutes (`JWT_EXPIRE_MINUTES`).
- There is no refresh-token flow.

### Frontend Structure

- `lib/api.ts` — API client (talks to `VITE_API_URL`, defaults to `http://localhost:8000`)
- `lib/diagrams.ts` — which image paths may load, diagram URLs, cited-but-not-drawn figures, hiding half-streamed image Markdown
- `lib/AuthContext.tsx`, `lib/ChatHistoryContext.tsx`, `lib/ThemeContext.tsx` — app-wide React contexts
- `components/layout/RequireAuth.tsx` — route guard wrapping all authenticated routes in `App.tsx`
- `components/chat/` — chat UI incl. `MarkdownContent.tsx` (renders assistant answers: `react-markdown` + `remark-gfm`/`remark-math` + `rehype-katex` for LaTeX, since the backend prompt instructs the LLM to emit Markdown/LaTeX) and `CitationsCard.tsx` (renders the `citations` array from chat responses)
- `components/dashboard/` — analytics/stats cards, currently backed by `data/mockData.ts` (not yet wired to a real backend endpoint — check before assuming dashboard data is live)
- `components/ui/` — shadcn-style primitives (Base UI + Tailwind + `class-variance-authority`)

The path alias `@/*` maps to `frontend/src/*` and is configured in `tsconfig.app.json` and `vite.config.ts`.

## Environment configuration

### Backend

See `backend/.env.example` for the complete template.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required OpenAI authentication key |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME` | Qdrant connection and collection settings |
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` | JWT configuration |
| `OPENAI_EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` | Embedding model and vector size |
| `OPENAI_CHAT_MODEL` | Chat completion model |
| `LLM_PROVIDER` | `openai` (default, `OPENAI_CHAT_MODEL`) or `qwen`: every `chat_llm()` call — intent, query rewrite, answers, topics — goes to Qwen3.8-27B on the RunPod serverless llama.cpp endpoint (`rag/qwen_client.py`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`). The endpoint has no streaming or JSON mode, so `ChatQwenRunpod` returns the answer as one stream chunk and does structured output in the prompt (schema in the system turn, reply validated, one retry). Embeddings are set separately by `EMBEDDING_PROVIDER`. |
| `EMBEDDING_PROVIDER` | `openai` (default, `OPENAI_EMBEDDING_MODEL`) or `qwen`: Qwen3-Embedding-4B on the same RunPod endpoint (`QwenRunpodEmbeddings` in `rag/qwen_client.py`, an `embed` job; queries get the `Instruct: ...
Query:` prefix), cut to `EMBEDDING_DIMENSION`. A different provider or dimension needs its own `QDRANT_COLLECTION_NAME`. |
| `OPENAI_TEMPERATURE` | Optional. Sent to the chat model only when set (`rag/llm.py` `chat_llm()`, used by every chat call). Leave unset for models that only accept their default temperature. |
| `ALLOWED_ORIGINS` | CORS configuration |
| `LANGCHAIN_*` | Optional LangSmith tracing configuration |

`EMBEDDING_DIMENSION` must match the dimension used when the Qdrant collection was created. Changing it requires recreating the collection.

### Frontend

The frontend `.env` file requires `VITE_API_URL`.
