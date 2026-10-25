---
description: Run all quality gates (backend ruff + pytest, frontend oxlint + tsc + vitest) and fix what fails
argument-hint: "[backend|frontend]"
---

Run `python scripts/check.py $ARGUMENTS` from the repository root.

If every step passes, reply with the summary table only.

If a step fails, read the failure, fix the cause in the source (not by loosening a test, a lint rule or a type), and run the failing step again until it passes. Then rerun the full `python scripts/check.py $ARGUMENTS` once and report what was wrong and what you changed.
