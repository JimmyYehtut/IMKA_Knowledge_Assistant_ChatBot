IMKA Knowledge Assistant is a RAG chatbot with a Python backend and a React frontend.

**Backend**
- **Language and API:** Python 3.12 with FastAPI, served by Uvicorn. There is also a legacy Streamlit single-page app (`backend/app.py`).
- **RAG framework:** LangChain (`langchain`, `langchain-openai`, `langchain-qdrant`, `langchain-community`), with LangSmith for optional tracing.
- **LLM and embeddings:** OpenAI. The chat model comes from `OPENAI_CHAT_MODEL` (currently `gpt-6-luna` in your `.env`; the code default is `gpt-4o-mini`). Embeddings use `text-embedding-3-small` at 1024 dimensions.
- **Vector store:** Qdrant, collection `ragindex`.
- **Reranking:** `rank-bm25` keyword rerank on top of the vector search.
- **Document parsing:**
  - MarkItDown for the standard pipeline (PDF, Office, HTML, CSV, text).
  - Docling for layout-aware PDF parsing and chunking.
- **Relational database:** PostgreSQL 16 for users, conversations and messages, accessed through async SQLAlchemy and `asyncpg`.
- **Auth:** JWT bearer tokens with `python-jose` and `bcrypt`.
- **Other:** `python-dotenv` for config; `truststore` and `pip-system-certs` for Windows certificate trust.

**Frontend**
- **Framework:** React with TypeScript, built with Vite.
- **Styling and components:** Tailwind CSS with shadcn-style primitives (Base UI, `class-variance-authority`) and `lucide-react` icons.
- **Routing:** `react-router-dom`.
- **Answer rendering:** `react-markdown` with `remark-gfm`, `remark-math` and `rehype-katex` for Markdown tables and LaTeX.
- **Linting:** oxlint. No test runner is configured on either side.

**Infrastructure**
- Docker Compose project `imka`, running `imka-qdrant` (ports 6333/6334) and `imka-postgres` (port 5433).
- Backend on port 8000, frontend dev server on port 5173.
- `start.ps1` and `start.bat` launch the stack on Windows.