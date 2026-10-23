---
description: Check the IMKA dev stack (Docker, Qdrant, Postgres, API, frontend) and report what is down
disable-model-invocation: true
---

Check each service and report one line per service: up, down, or misconfigured.

1. Docker engine: `docker info` succeeds.
2. Containers: `docker ps` shows `imka-qdrant` and `imka-postgres` running. Port 6333 is shared with the user's Capstone `mediq-qdrant` container; if that one is running, say so, because `imka-qdrant` cannot bind while it is.
3. Qdrant: `curl -s http://localhost:6333/collections/ragindex` returns 200.
4. API: `curl -s http://localhost:8000/api/health` reports `qdrant_reachable` and `openai_configured` true.
5. Frontend: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/` returns 200.

Do not start, stop or restart anything. For each service that is down, give the exact command to bring it up, taken from `start.ps1` or `CLAUDE.md`: `docker compose up -d` in `backend/`, `uvicorn api.main:app --reload --port 8000` in `backend/` with the venv, `npm run dev` in `frontend/`, or the whole stack with `.\start.ps1`.
