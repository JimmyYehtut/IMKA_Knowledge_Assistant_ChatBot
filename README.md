# IMKA Knowledge Assistant ChatBot

A Retrieval-Augmented Generation (RAG) knowledge assistant with a chat interface, built with **LangChain**, **Qdrant**, **OpenAI**, **FastAPI**, and a **React** dashboard.

## Project Structure

```text
.
├── backend/     # FastAPI API, RAG pipeline (LangChain, Qdrant, OpenAI)
├── frontend/    # React + TypeScript + Vite dashboard/chat UI
├── Archive/     # Reference screenshots
└── UI Mockup/   # UI design mockups
```

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- [Docker](https://www.docker.com/) (for running Qdrant locally)
- An OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JimmyYehtut/IMKA_Knowledge_Assistant_ChatBot.git
cd IMKA_Knowledge_Assistant_ChatBot
```

### 2. Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# then edit .env and add your OpenAI API key / Qdrant settings
```

Start Qdrant (vector database) and Postgres via Docker:

```bash
docker compose up -d
```

This creates the `imka` compose project with the containers `imka-qdrant` and `imka-postgres` (volumes `imka-qdrant-data`, `imka-postgres-data`). Both use `restart: unless-stopped`, so they come back automatically with Docker Desktop.

Qdrant will be reachable at `http://localhost:6333` (dashboard at `http://localhost:6333/dashboard`) and Postgres at `localhost:5433` (database `imka`). If another project's container already holds port 6333 or 5433, stop it first.

### 3. Frontend setup

```bash
cd frontend

# Install Node dependencies
npm install

# Configure environment variables
cp .env.example .env
# then edit .env with the backend API URL, etc.
```

## Running the App

### Backend — FastAPI (serves the React frontend)

```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

API reachable at `http://localhost:8000` (interactive docs at `/docs`).

### Frontend — React dashboard

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:5173` by default.

Document ingestion is done through the React frontend's Documents page (which calls `POST /api/documents` on the FastAPI backend).

## Direct CLI Ingestion (alternative to the UI)

```bash
cd backend
python -m ingest.embed_and_store path/to/document1.pdf path/to/document2.docx
python -m ingest.embed_and_store_docling path/to/document.pdf   # Docling pipeline, PDF only
```

## Tech Stack

- **Backend**: FastAPI, LangChain, Qdrant, OpenAI, SQLAlchemy
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS
- **Vector Store**: Qdrant (via Docker)
