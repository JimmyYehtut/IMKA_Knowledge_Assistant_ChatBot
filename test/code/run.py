"""Command line for the test-A … test-G evaluation (see README.md).

    python run.py questions                         # check questions.json, counts per group and PDF
    python run.py documents test-F                  # which document_ids the run will use (pins them)
    python run.py answer test-F                     # answer all questions, repeats 1-3
    python run.py judge test-F [--check]            # judge them (Claude); --check adds the Sonnet check
    python run.py report                            # summary, comparisons, judge agreement and spend

Every step resumes where it stopped.
"""
import argparse
import asyncio
import sys

import imka_eval as E


def main() -> None:
    # Questions contain Ω, ±, … — a redirected log on Windows would otherwise be cp1252 and crash.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["questions", "documents", "answer", "judge", "report"])
    parser.add_argument("run", nargs="?", choices=list(E.RUNS))
    parser.add_argument("--repeats", type=int, nargs="+", default=list(E.REPEATS))
    parser.add_argument("--concurrent", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="judge: also score context_recall with Sonnet")
    parser.add_argument("--redo", action="store_true", help="ignore saved results and start over")
    args = parser.parse_args()

    if args.step in ("documents", "answer", "judge") and not args.run:
        parser.error(f"{args.step} needs a run: {', '.join(E.RUNS)}")

    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    if args.step == "questions":
        questions = E.load_questions()
        print(f"{len(questions)} questions, {sum(E.expects_refusal(q) for q in questions)} expect a refusal\n")
        print(E.question_counts(questions))
    elif args.step == "documents":
        backend = E.connect()
        for d in E.resolve_documents(backend, args.run, E.load_questions()):
            print(f"{d['document_id']}  {d['document_name']}  chunks={d.get('chunk_count')}  "
                  f"models={d.get('models')}  embedding={(d.get('embedding') or {}).get('model', E.DEFAULT_EMBEDDING)}")
    elif args.step == "answer":
        backend = E.connect()
        questions = E.load_questions()
        for repeat in args.repeats:
            E.run_answers(backend, args.run, repeat, questions, concurrent=args.concurrent or 4, redo=args.redo)
    elif args.step == "judge":
        for repeat in args.repeats:
            asyncio.run(E.run_judge(args.run, repeat, check=args.check,
                                    concurrent=args.concurrent or 3, redo=args.redo))
    elif args.step == "report":
        print("== Summary per run and group ==")
        print(E.summary())
        print("\n== Comparisons (primary metrics, paired bootstrap 95% CI) ==")
        print(E.comparisons().to_string(index=False))
        print("\n== Answering cost per question (Qwen: model working time x GPU $/h) ==")
        print(E.answering_cost().to_string())
        print("\n== Haiku vs Sonnet agreement ==")
        print(E.judge_agreement())
        spend = E.judge_spend()
        if not spend.empty:
            print("\n== Judge spend ==")
            print(spend.groupby(["run", "model"])[["calls", "input_tokens", "output_tokens", "cost_usd"]].sum())
            print(f"total ${spend['cost_usd'].sum():.2f}")


if __name__ == "__main__":
    sys.exit(main())
