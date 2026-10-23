"""Claude Code PostToolUse hook: lint the file just edited and run the tests that cover it.

Wired in .claude/settings.json for Edit|Write|MultiEdit. Reads the hook event
JSON on stdin. Exit 0 = fine (or nothing to check); exit 2 = problems, and
stderr is shown to Claude so it fixes them before moving on. Fast by design:
one file is linted, and only the test files mapped to it run (not the full
suite — `python scripts/check.py` does that).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

# Backend source -> test files that exercise it. Anything else under backend/ is lint-only.
BACKEND_TESTS = {
    "ingest/embed_and_store_docling.py": ["tests/test_figure_extraction.py"],
    "rag/pipeline.py": ["tests/test_pipeline_figures.py", "tests/test_chat_api.py"],
    "api/openai_compat.py": ["tests/test_chat_api.py"],
    "api/schemas.py": ["tests/test_chat_api.py"],
    "api/main.py": ["tests/test_chat_api.py"],
    "api/chat.py": ["tests/test_chat_api.py"],
}


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "FORCE_COLOR": "0", "NO_COLOR": "1"}
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def backend_python() -> str:
    for candidate in (BACKEND / ".venv" / "Scripts" / "python.exe", BACKEND / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check_backend(rel: str) -> list[str]:
    problems = []
    py = backend_python()
    code, out = run([py, "-m", "ruff", "check", rel], BACKEND)
    if code:
        problems.append(f"ruff found problems in backend/{rel}:\n{out}")
    tests = [rel] if rel.startswith("tests/test_") else BACKEND_TESTS.get(rel, [])
    if tests:
        code, out = run([py, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], BACKEND)
        if code:
            problems.append(f"backend tests failed after editing backend/{rel}:\n{out[-4000:]}")
    return problems


def check_frontend(rel: str) -> list[str]:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        return []
    problems = []
    code, out = run([npx, "oxlint", rel], FRONTEND)
    if code:
        problems.append(f"oxlint found problems in frontend/{rel}:\n{out}")
    code, out = run([npx, "vitest", "related", rel, "--run", "--passWithNoTests"], FRONTEND)
    if code:
        problems.append(f"frontend tests failed after editing frontend/{rel}:\n{out[-4000:]}")
    return problems


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0
    path = Path(file_path).resolve()

    problems: list[str] = []
    if path.suffix == ".py" and path.is_relative_to(BACKEND) and ".venv" not in path.parts:
        problems = check_backend(path.relative_to(BACKEND).as_posix())
    elif path.suffix in (".ts", ".tsx") and path.is_relative_to(FRONTEND / "src"):
        problems = check_frontend(path.relative_to(FRONTEND).as_posix())

    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
