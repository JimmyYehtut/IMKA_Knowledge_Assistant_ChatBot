---
description: Dry-run Docling figure extraction on a PDF and check every figure's images and page, without ingesting
argument-hint: "<path to PDF or saved parse JSON> [document id]"
disable-model-invocation: true
---

Use the `ingest-checker` agent on: $ARGUMENTS

Nothing may be written to Qdrant, the document registry or `backend/data/`. Relay the agent's table of problem figures, whether the index differs from a fresh extraction, and whether re-ingesting is needed. Leave the re-ingest decision to me.
