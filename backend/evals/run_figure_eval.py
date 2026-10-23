"""Live diagram eval: asks the running chat API the questions in figures.yaml and
checks that the chat window shows the expected figures and that every image loads.

A figure counts as shown when the answer draws it inline, or when the answer
cites a figure passage, which the chat UI shows under the answer ("Diagrams
cited in this answer"). The report keeps the two apart.

    cd backend
    python -m evals.run_figure_eval                  # all cases
    python -m evals.run_figure_eval --only lift      # cases whose id contains "lift"
    python -m evals.run_figure_eval --api http://localhost:8000 --min-pass 0.9

Needs the stack running (API, Qdrant, Postgres) and spends OpenAI tokens: about
three LLM calls per question. Log-in comes from IMKA_EVAL_EMAIL and
IMKA_EVAL_PASSWORD; without them a local eval account is registered once and
remembered in evals/.eval_account.json (git-ignored). Writes a Markdown and a
JSON report to evals/reports/ and exits 1 when the pass rate is below --min-pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?[^)]*\)")
FIGURE_FILE_RE = re.compile(r"/fig(\d+)_\d+\.png$")
FENCE = "`" * 3


@dataclass
class Result:
    id: str
    question: str
    passed: bool
    reason: str
    figures_drawn: list[int] = field(default_factory=list)
    figures_cited: list[int] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    cited_figures: list[str] = field(default_factory=list)
    seconds: float = 0.0
    answer: str = ""


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.token: str | None = None

    def request(self, path: str, body: dict | None = None, timeout: float = 240) -> tuple[int, bytes, str]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data else "GET")
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "")

    def login(self) -> None:
        email, password = os.getenv("IMKA_EVAL_EMAIL"), os.getenv("IMKA_EVAL_PASSWORD")
        account_file = HERE / ".eval_account.json"
        if not (email and password) and account_file.exists():
            saved = json.loads(account_file.read_text(encoding="utf-8"))
            email, password = saved["email"], saved["password"]
        if not (email and password):
            email, password = f"imka-eval-{secrets.token_hex(4)}@example.test", secrets.token_urlsafe(18)
            status, body, _ = self.request(
                "/api/auth/register", {"email": email, "password": password, "name": "IMKA eval"}
            )
            if status != 201:
                sys.exit(f"Could not register an eval account ({status}): {body[:200]!r}")
            account_file.write_text(json.dumps({"email": email, "password": password}), encoding="utf-8")
            print(f"Registered eval account {email} (saved in {account_file.name})")
        status, body, _ = self.request("/api/auth/login", {"email": email, "password": password})
        if status != 200:
            sys.exit(f"Eval log-in failed ({status}). Set IMKA_EVAL_EMAIL / IMKA_EVAL_PASSWORD.")
        self.token = json.loads(body)["access_token"]


def run_case(api: Api, case: dict, persona: str) -> Result:
    started = time.time()
    status, body, _ = api.request(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": case["question"]}], "stream": False, "persona": persona},
    )
    if status != 200:
        return Result(case["id"], case["question"], False, f"chat API returned {status}",
                      seconds=round(time.time() - started, 1))
    response = json.loads(body)
    answer = response["choices"][0]["message"]["content"]

    images, drawn = [], []
    for alt, src in IMAGE_RE.findall(answer):
        code: int | str | None = None
        if src.startswith("diagrams/") and ".." not in src:
            code, _, content_type = api.request(f"/data/{src}", timeout=30)
            if code == 200 and not content_type.startswith("image/"):
                code = f"200 but {content_type}"
        images.append({"src": src, "alt": alt, "status": code})
        if match := FIGURE_FILE_RE.search(src):
            drawn.append(int(match.group(1)))
    drawn = sorted(set(drawn))
    cited_images = [c for c in response.get("citations", []) if c.get("image_path")]
    cited = [c.get("figure_caption") or c["image_path"] for c in cited_images]
    cited_numbers = sorted({int(m.group(1)) for c in cited_images if (m := FIGURE_FILE_RE.search(c["image_path"]))})
    for c in cited_images:  # the UI loads these too
        if c["image_path"] not in {i["src"] for i in images}:
            code, _, _ = api.request(f"/data/{c['image_path']}", timeout=30)
            if code != 200:
                images.append({"src": c["image_path"], "alt": "cited figure", "status": code})
    shown = set(drawn) | set(cited_numbers)

    problems = []
    broken = [i["src"] for i in images if i["status"] != 200]
    if broken:
        problems.append(f"images that do not load: {broken}")
    if case.get("expect_none") and (images or shown):
        problems.append(f"expected no diagrams, got {sorted(shown) or [i['src'] for i in images]}")
    expected = case.get("expect_any")
    if expected and not set(expected) & shown:
        problems.append(f"expected one of figures {expected}, showed {sorted(shown) or 'none'}")

    return Result(
        id=case["id"], question=case["question"], passed=not problems,
        reason="; ".join(problems) or "ok", figures_drawn=drawn, figures_cited=cited_numbers, images=images,
        cited_figures=cited, seconds=round(time.time() - started, 1), answer=answer,
    )


def write_reports(results: list[Result], api_url: str, golden: dict) -> Path:
    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    passed = sum(r.passed for r in results)
    lines = [
        f"# Diagram eval {stamp}",
        "",
        f"{passed}/{len(results)} passed against {api_url} ({golden.get('document', '')}).",
        "",
        "| Case | Result | Drawn inline | Shown under the answer | Time | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        under = [n for n in r.figures_cited if n not in r.figures_drawn]
        lines.append(
            f"| {r.id} | {'PASS' if r.passed else 'FAIL'} | {', '.join(map(str, r.figures_drawn)) or '–'} "
            f"| {', '.join(map(str, under)) or '–'} | {r.seconds}s | {r.reason.replace('|', '/')} |"
        )
    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "## Failed answers", ""]
        for r in failed:
            lines += [f"### {r.id}", "", f"**Q:** {r.question}", "", FENCE + "markdown", r.answer.strip(), FENCE, ""]
    report = out_dir / f"figures-{stamp}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"figures-{stamp}.json").write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.getenv("IMKA_API_URL", "http://localhost:8000"))
    parser.add_argument("--golden", default=str(HERE / "figures.yaml"))
    parser.add_argument("--only", help="run only cases whose id contains this text")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-pass", type=float, default=0.85)
    args = parser.parse_args()

    golden = yaml.safe_load(Path(args.golden).read_text(encoding="utf-8"))
    cases = [c for c in golden["cases"] if not args.only or args.only in c["id"]]
    if not cases:
        sys.exit(f"No cases match --only {args.only!r}")

    api = Api(args.api)
    status, _, _ = api.request("/api/health", timeout=10)
    if status != 200:
        sys.exit(f"API at {args.api} is not healthy ({status}). Start the stack first (start.ps1).")
    api.login()

    print(f"Running {len(cases)} diagram cases against {args.api} ...")
    persona = golden.get("persona", "engineer")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda c: run_case(api, c, persona), cases))

    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        under = [n for n in r.figures_cited if n not in r.figures_drawn]
        print(f"  {verdict}  {r.id:<30} inline {r.figures_drawn or '-'}  under {under or '-'}  {r.seconds}s  "
              f"{'' if r.passed else r.reason}")
    report = write_reports(results, args.api, golden)
    passed = sum(r.passed for r in results)
    rate = passed / len(results)
    print(f"\n{passed}/{len(results)} passed ({rate:.0%}). Report: {report}")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
