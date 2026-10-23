"""Run the project's quality gates: backend lint + tests, frontend lint + types + tests.

    python scripts/check.py              # everything
    python scripts/check.py backend      # ruff + pytest only
    python scripts/check.py frontend     # oxlint + tsc + vitest only

Uses backend/.venv when it exists, so it works from any shell (PowerShell, Git
Bash, CI). Exits non-zero if any step fails, after running all of them, and
prints a one-line summary per step. No step touches the network or the dev
stack; the live diagram eval is separate (python -m evals.run_figure_eval).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def backend_python() -> str:
    for candidate in (BACKEND / ".venv" / "Scripts" / "python.exe", BACKEND / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def npm() -> str:
    found = shutil.which("npm") or shutil.which("npm.cmd")
    if not found:
        sys.exit("npm not found on PATH")
    return found


def steps(target: str) -> list[tuple[str, list[str], Path]]:
    py = backend_python()
    backend = [
        ("backend: ruff", [py, "-m", "ruff", "check", "."], BACKEND),
        ("backend: pytest", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider"], BACKEND),
    ]
    frontend = [
        ("frontend: oxlint", [npm(), "run", "--silent", "lint"], FRONTEND),
        ("frontend: tsc", [npm(), "exec", "--", "tsc", "-b"], FRONTEND),
        ("frontend: vitest", [npm(), "run", "--silent", "test"], FRONTEND),
    ]
    return {"backend": backend, "frontend": frontend, "all": backend + frontend}[target]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target not in ("all", "backend", "frontend"):
        sys.exit(__doc__)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "FORCE_COLOR": "0", "NO_COLOR": "1"}

    results = []
    for name, cmd, cwd in steps(target):
        print(f"\n=== {name}", flush=True)
        started = time.time()
        code = subprocess.call(cmd, cwd=cwd, env=env)
        results.append((name, code, time.time() - started))

    print("\n=== summary")
    for name, code, seconds in results:
        print(f"  {'ok  ' if code == 0 else 'FAIL'}  {name:<20} {seconds:5.1f}s")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
