"""Writes and executes the notebooks for completed runs.

    python notebooks.py test-A test-B      # output/<run>/evaluate.ipynb for those runs
    python notebooks.py --all              # every run that has judged answers
    python notebooks.py --report           # output/report.ipynb (all runs side by side)

Each notebook only reads saved results; nothing in it calls an LLM. Executed in place, so
the .ipynb files under output/ carry their outputs and open with the results visible.
"""
import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import imka_eval as E

CODE_DIR = Path(__file__).resolve().parent
DESCRIPTIONS = {
    "C1": "standard pipeline: MarkItDown + 1,000-character splitter",
    "C2": "Docling layout-aware, table-atomic chunking",
    "C3": "Docling chunking + VLM figure descriptions",
}
COST_NOTE = ("## Answering cost and time per question\n\nOpenAI models are priced per token; Qwen per second of "
             "model working time (llama-server's prompt + generation time) at `E.GPU_PRICE_PER_HOUR`. RunPod "
             "queueing, cold starts and job overhead are shown (`runpod_exec_s`, `runpod_delay_s`) but not "
             "charged. `wall_s` is the question's end-to-end time as the evaluation saw it.")
SETUP = f"""import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parent.parent / "code"))  # test/code
import imka_eval as E

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)"""


def _execute(nb, path: Path) -> None:
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)


def run_notebook(run: str) -> Path:
    spec = E.RUNS[run]
    stack = f"answering: **{spec['answer_model']}**"
    if spec.get("caption_model"):
        stack += f", figure descriptions: **{spec['caption_model']}**"
    stack += f", embedding: **{spec.get('embedding', E.DEFAULT_EMBEDDING)}**"
    cells = [
        new_markdown_cell(f"# {run} — {spec['config']}: {DESCRIPTIONS[spec['config']]}\n\n{stack}.\n\n"
                          "Means over 3 repeats of 150 in-scope and 20 out-of-scope questions. Reads "
                          "`answers-r*.json` and `judge-r*.json` in this folder; nothing here calls an LLM."),
        new_code_cell(SETUP + f'\n\nRUN = "{run}"\nrows = E.run_rows(RUN)\nin_scope = rows[rows.group != "out_of_scope"]'),
        new_markdown_cell("## Documents\n\nThe five documents the run searched (pinned in `documents.json`)."),
        new_code_cell('docs = E._read_json(E.run_dir(RUN) / "documents.json")["documents"]\n'
                      'pd.DataFrame([{"document": d["document_name"][:45], "document_id": d["document_id"], '
                      '"chunks": d.get("chunk_count"), "captioning": (d.get("models") or {}).get("captioning", "-"), '
                      '"embedding": (d.get("embedding") or {}).get("model", E.DEFAULT_EMBEDDING)} for d in docs])'),
        new_markdown_cell("## Results by question group"),
        new_code_cell("E.summary(runs=(RUN,), by=\"group\")"),
        new_code_cell("E.summary(runs=(RUN,), by=None)"),
        new_markdown_cell("## Stability across the three repeats"),
        new_code_cell('in_scope.groupby("repeat")[["hit@5", "context_recall", "answer_accuracy"]].mean().round(3)'),
        new_markdown_cell("## Out-of-scope questions\n\nShare of the 20 out-of-scope questions declined, per repeat."),
        new_code_cell('rows[rows.group == "out_of_scope"].groupby("repeat")["refused"].mean().round(3)'),
        new_markdown_cell("## Lowest answer accuracy\n\nThe ten in-scope questions with the lowest accuracy, "
                          "averaged over the repeats, with the first repeat's answer."),
        new_code_cell('answers = {r["question"]: r.get("answer") or "" for r in E.load_answers(RUN, 1)["results"]}\n'
                      'worst = (in_scope.groupby(["question", "group", "document_name"])["answer_accuracy"].mean()\n'
                      '         .sort_values().head(10).reset_index())\n'
                      'worst["document_name"] = worst["document_name"].str[:20]\n'
                      'worst["answer (repeat 1)"] = worst["question"].map(lambda q: answers.get(q, "")[:160])\n'
                      'worst'),
        new_markdown_cell(COST_NOTE),
        new_code_cell("E.answering_cost((RUN,)).T"),
        new_markdown_cell("## Judge spend (Anthropic)"),
        new_code_cell('spend = E.judge_spend((RUN,))\n'
                      'spend.groupby("model")[["calls", "input_tokens", "output_tokens", "cost_usd"]].sum()'),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                             "language": "python"}})
    path = E.run_dir(run) / "evaluate.ipynb"
    _execute(nb, path)
    return path


def report_notebook() -> Path:
    cells = [
        new_markdown_cell("# IMKA evaluation — all runs\n\nRuns, comparisons and primary metrics are fixed in "
                          "`code/imka_eval.py` (`RUNS`, `COMPARISONS`). Means over 3 repeats; 95% CIs from a "
                          "paired bootstrap over questions. Nothing here calls an LLM."),
        new_code_cell(SETUP.replace('Path.cwd().parent.parent / "code"', 'Path.cwd().parent / "code"')
                      + '\n\nDONE = [r for r in E.RUNS if (E.run_dir(r) / "judge-r3.json").exists()]\nDONE'),
        new_markdown_cell("## Every run, all questions"),
        new_code_cell("E.summary(runs=DONE, by=None)"),
        new_markdown_cell("## Every run, by question group"),
        new_code_cell('E.summary(runs=DONE, by="group")'),
        new_markdown_cell("## Pre-registered comparisons\n\nB − A on each comparison's primary metric, over the "
                          "questions both runs scored (each averaged over its repeats)."),
        new_code_cell("comparisons = E.comparisons()\ncomparisons"),
        new_code_cell("import matplotlib.pyplot as plt\n\n"
                      'shown = comparisons.dropna(subset=["diff"])\n'
                      'labels = [f"{r.A} → {r.B}  {r.metric} ({r.groups})" for r in shown.itertuples()]\n'
                      "fig, ax = plt.subplots(figsize=(9, 0.55 * len(shown) + 1))\n"
                      'ax.errorbar(shown["diff"], range(len(shown)), xerr=[shown["diff"] - shown["ci_low"], '
                      'shown["ci_high"] - shown["diff"]], fmt="o", capsize=4)\n'
                      'ax.axvline(0, color="grey", linewidth=1)\n'
                      "ax.set_yticks(range(len(shown)), labels)\nax.invert_yaxis()\n"
                      'ax.set_xlabel("B − A (95% CI)")\nplt.tight_layout()'),
        new_markdown_cell(COST_NOTE),
        new_code_cell("E.answering_cost(DONE)"),
        new_markdown_cell("## Judge agreement (Haiku 5.5 vs Sonnet 5.5 on context recall)"),
        new_code_cell("E.judge_agreement(DONE)"),
        new_markdown_cell("## Judge spend (Anthropic)"),
        new_code_cell('spend = E.judge_spend(DONE)\n'
                      'display(spend.groupby(["run", "model"])[["calls", "input_tokens", "output_tokens", "cost_usd"]].sum())\n'
                      'print(f"total ${spend.cost_usd.sum():.2f}")'),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                             "language": "python"}})
    path = E.OUTPUT_DIR / "report.ipynb"
    _execute(nb, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help=f"runs to write: {', '.join(E.RUNS)}")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    runs = [r for r in E.RUNS if (E.run_dir(r) / "judge-r3.json").exists()] if args.all else args.runs
    for run in runs:
        if run not in E.RUNS:
            sys.exit(f"unknown run {run}; runs: {', '.join(E.RUNS)}")
        print("wrote", run_notebook(run))
    if args.report:
        print("wrote", report_notebook())


if __name__ == "__main__":
    main()
