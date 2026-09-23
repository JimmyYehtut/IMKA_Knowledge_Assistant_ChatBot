"""Retrieval metrics — Hit Rate@K and MRR — for answers-*.json runs. No LLM judge.

Every question in questions.json names the page its answer lives on (`source`:
"Table 11, p.114"), and every chunk a RAG run retrieved is labeled with its page
("[ABB Manual ..., p. 114]"). So retrieval can be graded with plain comparison:
a retrieved chunk is *relevant* if it sits on an expected page. No judge, no
judge bias, no cost, deterministic — the complement to evaluate-ragas.py.

  Hit Rate@K   share of questions with at least one relevant chunk in the top K
  MRR          mean of 1/rank of the first relevant chunk (0 when none is found)

Two relevance tolerances are reported side by side:

  exact          chunk page == a source page
  within_1_page  chunk page within ±1 of a source page. A chunk is labeled with
                 the page it *starts* on, so text that runs over a page break, or
                 a table split across two pages, is attributed to the neighbouring
                 page. On an early IMKA run 17 of 67 nearest chunks sat
                 exactly one page off, so strict matching alone undercounts.

What this does and does not measure:

- The ranks are those of the *final* context — the BM25-reranked top N (5 by
  default) that the answer was generated from. MRR is therefore MRR@N. Measuring
  the reranker itself (Hit@15 before vs Hit@5 after) needs the pipeline to log
  the pre-rerank candidates, which the answers files do not contain.
- A page hit means the right page was retrieved, not that the right *passage*
  was: Q1 retrieves p.114 but that chunk stops before the 100 °C row. Use
  context_recall from evaluate-ragas.py for "was the fact actually in the context".
- Questions whose source names no page (the "deliberately fabricated" absence
  probes) have nothing to retrieve and are skipped, and listed as skipped.
- No-RAG baseline runs have no retrieved chunks
  and are skipped with a message.

Writes test/output/retrieval-<run>.json per RAG answers file: per-question ranks
and hits, aggregates overall / by failure_type / by rag_target, and a flat `rows`
array for charting (same shape as ragas-<run>.json).

Usage (from repo root; standard library only, any Python works):
  python test\\evaluate-retrieval.py                                   # every test/output/answers-*.json
  python test\\evaluate-retrieval.py test\\output\\answers-imka.json
  python test\\evaluate-retrieval.py -k 1,3,5,10
  python test\\evaluate-retrieval.py --document "ABB Manual"          # multi-document collections
"""
import argparse
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# How rag/pipeline.py:_stage4_assemble_context joins chunks and labels each one
# ("[<title>, p. <page>]", or just "[<title>]" when the page is unknown). If that
# format changes, these two must change with it.
CONTEXT_SEPARATOR = "\n\n---\n\n"
CHUNK_LABEL = re.compile(r"^\[(?P<document>.+?)(?:, p\. (?P<page>\d+))?\]\s*\n")

# Every "p.N" in a questions.json source. Some name two pages
# ("§7.3.1, p.75 legend + e.g. p.76 ..."); either one counts as relevant.
SOURCE_PAGE = re.compile(r"\bp\.\s*(\d+)")

DEFAULT_K = (1, 3, 5)
TOLERANCES = {"exact": 0, "within_1_page": 1}


# ── Parsing ───────────────────────────────────────────────────────────────────

def expected_pages(source: str | None) -> list[int]:
    return sorted({int(p) for p in SOURCE_PAGE.findall(source or "")})


def retrieved_chunks(context: str | None) -> list[dict]:
    """The retrieved chunks in rank order (rank 1 = first in the context)."""
    chunks = []
    for part in (context or "").split(CONTEXT_SEPARATOR):
        part = part.strip()
        if not part:
            continue
        match = CHUNK_LABEL.match(part + "\n")
        chunks.append({
            "rank": len(chunks) + 1,
            "document": match.group("document") if match else None,
            "page": int(match.group("page")) if match and match.group("page") else None,
        })
    return chunks


# ── Scoring ───────────────────────────────────────────────────────────────────

def is_relevant(chunk: dict, pages: list[int], tolerance: int, document: str | None) -> bool:
    if chunk["page"] is None:
        return False
    if document and document not in (chunk["document"] or ""):
        return False
    return any(abs(chunk["page"] - page) <= tolerance for page in pages)


def score_record(record: dict, ks: list[int], document: str | None) -> tuple[dict | None, str | None]:
    """Per-question metrics, or (None, reason) when the question cannot be graded."""
    if record.get("error"):
        return None, f"run error: {record['error'][:100]}"
    pages = expected_pages(record.get("source"))
    if not pages:
        return None, "source names no page (fabricated / nothing to retrieve)"
    chunks = retrieved_chunks(record.get("context"))
    if not chunks:
        return None, "no retrieved context"

    metrics = {}
    for name, tolerance in TOLERANCES.items():
        first = next((c["rank"] for c in chunks if is_relevant(c, pages, tolerance, document)), None)
        metrics[name] = {
            "first_relevant_rank": first,
            "reciprocal_rank": round(1 / first, 4) if first else 0.0,
            **{f"hit@{k}": first is not None and first <= k for k in ks},
        }

    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "failure_type": record.get("failure_type"),
        "rag_target": record.get("rag_target"),
        "source": record.get("source"),
        "expected_pages": pages,
        "retrieved_pages": [c["page"] for c in chunks],
        "n_retrieved": len(chunks),
        "metrics": metrics,
    }, None


# ── Aggregation ───────────────────────────────────────────────────────────────

def summarise(results: list[dict], ks: list[int]) -> dict:
    summary = {"questions": len(results)}
    for name in TOLERANCES:
        block = {}
        for k in ks:
            values = [r["metrics"][name][f"hit@{k}"] for r in results]
            block[f"hit_rate@{k}"] = round(statistics.mean(values), 4) if values else None
        rrs = [r["metrics"][name]["reciprocal_rank"] for r in results]
        block["mrr"] = round(statistics.mean(rrs), 4) if rrs else None
        summary[name] = block
    return summary


def aggregate(results: list[dict], ks: list[int]) -> dict:
    def grouped(key):
        groups: dict[str, list[dict]] = {}
        for r in results:
            groups.setdefault(r.get(key) or "(none)", []).append(r)
        return {k: summarise(v, ks) for k, v in sorted(groups.items())}

    return {
        "overall": summarise(results, ks),
        "by_failure_type": grouped("failure_type"),
        "by_rag_target": grouped("rag_target"),
    }


def build_rows(results: list[dict], ks: list[int]) -> list[dict]:
    """Long/tidy format: one row per (question, tolerance, metric) — ready for pandas."""
    rows = []
    for r in results:
        for name in TOLERANCES:
            m = r["metrics"][name]
            values = {f"hit@{k}": float(m[f"hit@{k}"]) for k in ks}
            values["reciprocal_rank"] = m["reciprocal_rank"]
            for metric, value in values.items():
                rows.append({
                    "question_id": r["id"],
                    "tolerance": name,
                    "metric": metric,
                    "value": value,
                    "failure_type": r.get("failure_type"),
                    "rag_target": r.get("rag_target"),
                })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def default_out(input_path: Path, out_dir: Path) -> Path:
    """answers-<run>.json -> <out_dir>/retrieval-<run>.json"""
    stem = input_path.stem
    stem = stem[len("answers-"):] if stem.startswith("answers-") else stem
    return out_dir / f"retrieval-{stem}.json"


def evaluate_file(path: Path, ks: list[int], document: str | None, out_dir: Path) -> dict | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("results", [])
    mode = payload.get("mode")

    if not any(r.get("context") for r in records):
        print(f"\n{path.name}: skipped — no retrieved context (mode={mode}). "
              f"Retrieval metrics need a RAG run that kept its context.")
        return None

    results, skipped = [], []
    for i, record in enumerate(records, 1):
        record = {"id": i, **record}
        scored, reason = score_record(record, ks, document)
        if scored is None:
            skipped.append({"id": record["id"], "question": record.get("question"), "reason": reason})
        else:
            results.append(scored)

    documents = Counter(c["document"] for r in records for c in retrieved_chunks(r.get("context")))
    if len(documents) > 1 and not document:
        print(f"  !! {path.name}: chunks come from {len(documents)} documents but --document is not "
              f"set, so a matching page number in the wrong document counts as a hit")

    max_retrieved = max((r["n_retrieved"] for r in results), default=0)
    beyond = [k for k in ks if k > max_retrieved]

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scored_file": str(path),
        "run": {k: payload.get(k) for k in ("mode", "model", "endpoint_id", "generated_at", "source_file")},
        "method": {
            "relevance": "chunk page matches a page named in the question's source",
            "tolerances": TOLERANCES,
            "k": ks,
            "ranked_list": "final reranked context the answer was generated from",
            "max_retrieved": max_retrieved,
            "document_filter": document,
            "documents_seen": dict(documents),
            "note": (f"hit@{beyond} equals hit@{max_retrieved}: only {max_retrieved} chunks "
                     f"were retrieved" if beyond else None),
        },
        "questions_total": len(records),
        "count": len(results),
        "skipped": skipped,
        "aggregate": aggregate(results, ks),
        "results": results,
        "rows": build_rows(results, ks),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = default_out(path, out_dir)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)

    print(f"\n{path.name}: {len(results)} graded, {len(skipped)} skipped -> {out_path.name}")
    if beyond:
        print(f"  note: k={beyond} exceeds the {max_retrieved} chunks retrieved per question")
    overall = out["aggregate"]["overall"]
    header = "".join(f"{f'hit@{k}':>9}" for k in ks) + f"{'MRR':>9}"
    print(f"  {'':15}{header}")
    for name in TOLERANCES:
        cells = "".join(f"{overall[name][f'hit_rate@{k}']:>9.3f}" for k in ks)
        print(f"  {name:15}{cells}{overall[name]['mrr']:>9.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="answers-*.json files (default: every test/output/answers-*.json)")
    parser.add_argument("-k", default=",".join(map(str, DEFAULT_K)),
                        help=f"comma-separated cut-offs (default {','.join(map(str, DEFAULT_K))})")
    parser.add_argument("--document",
                        help="only count chunks whose document name contains this text "
                             "(needed once the collection holds more than one document)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "output")
    args = parser.parse_args()

    try:
        ks = sorted({int(k) for k in args.k.split(",") if k.strip()})
    except ValueError:
        parser.error(f"-k must be comma-separated integers, got {args.k!r}")
    if not ks or min(ks) < 1:
        parser.error("-k values must be >= 1")

    inputs = args.inputs or sorted((HERE / "output").glob("answers-*.json"))
    if not inputs:
        parser.error("no answers-*.json files found")

    for path in inputs:
        if not path.exists():
            print(f"\n{path}: not found")
            continue
        evaluate_file(path, ks, args.document, args.out_dir)


if __name__ == "__main__":
    main()
