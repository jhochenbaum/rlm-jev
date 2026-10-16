"""E8b: same OOLONG/Pairs questions as E2, run through dspy.RLM (Deno sandbox) with Jev tools."""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import dspy
import pandas as pd

sys.path.insert(0, ".")
from jevlab.dspy_rlm import configure, make_rlm, run
from jevlab.oolong import LABELS, e2_jobs, load_val
from jevlab.oolong_program import extract, score
from jevlab.pairs_program import f1
from jevlab.rlm import ledger_rows

# E8B_API=1: call Claude through dspy.LM and the Anthropic API (ANTHROPIC_API_KEY, e.g. in .env.local)
# instead of `claude -p`.
API = os.environ.get("E8B_API") == "1"
if API and "ANTHROPIC_API_KEY" not in os.environ:
    from dotenv import dotenv_values

    os.environ["ANTHROPIC_API_KEY"] = dotenv_values(".env.local").get("ANTHROPIC_API_KEY") or ""
MAX_USD = float(os.environ.get("E8B_MAX_USD", "12"))  # hard stop on estimated Claude spend for this script (API mode)
configure("sonnet", api=API)
# E8B_NOTE=1: dspy.RLM with its `reasoning` output field replaced by a one-sentence `note` (rlm_jev.dspy_rlm.NoteRLM);
# Claude Sonnet 5.5 declines stock dspy.RLM (refusal category reasoning_extraction).
NOTE = os.environ.get("E8B_NOTE") == "1"
PFX = os.environ.get("E8B_PFX") or ("e8b_api" + dspy.__version__.replace(".", "") if API else "e8b") + (
    "_note" if NOTE else ""
)  # e.g. e8b_api341_note; E8B_PFX names a run on an unreleased dspy build
WORKERS = int(os.environ.get("E8B_WORKERS", "1"))  # per-run costs come from a shared ledger, so >1 mixes runs' costs

# The E2 questions at 131k, 1M and OOLONG-Pairs; dspy.RLM returns an answer string rather than calling FINAL.
SUFFIX = " Return the answer with the pairs formatted as (id1, id2) separated by commas."
jobs = e2_jobs(load_val().query("dataset == 'trec_coarse'"), with_32k=False, pairs_suffix=SUFFIX)

done = ledger_rows(PFX, include_failed=True)  # earlier runs, refusals included; matched to jobs by exact tag below

SPENT = [0.0]


def go(job):
    kind, _, r, q, context = job
    if API and SPENT[0] >= MAX_USD:
        print(f"SKIP (Claude spend cap ${MAX_USD} reached)", kind, flush=True)
        return None
    prev = next((d for d in done if d["tag"] == f"{PFX}_{kind}" and d["query"] == q[:300]), None)
    rlm = make_rlm(note_field=NOTE)
    rec = prev or run(rlm, context, q, f"{PFX}_{kind}")
    if kind == "pairs32k":
        G = {tuple(re.findall(r"\d+", s)) for s in r["answer"]}
        s = f1({tuple(m) for m in re.findall(r"\((\d+),\s*(\d+)\)", rec["answer"] or "")}, G)
        task = "pairs"
    else:
        s = score(extract(rec["answer"], r.answer_type, list(LABELS["trec_coarse"])), r.answer, r.answer_type)
        task = r.task
    SPENT[0] += (rec.get("root_usd_list") or 0) + (rec.get("sub_usd_list") or 0)
    row = {"kind": kind, "task": task, "score": s, **{k: v for k, v in rec.items() if k not in ("query", "answer")}}
    print(json.dumps(row), flush=True)
    return row


with ThreadPoolExecutor(WORKERS) as ex:
    rows = [x for x in ex.map(go, jobs) if x]
R = pd.DataFrame(rows)
R.to_csv(f"results/{PFX.replace('e8b', 'e8b_dspy_rlm')}.csv", index=False)
print(f"estimated Claude spend: ${SPENT[0]:.2f}")
print(
    R.groupby("kind")[
        ["score", "root_calls", "sub_llm_calls", "root_usd_list", "sub_usd_list", "wall_s", "jev_tool_used"]
    ]
    .mean()
    .round(3)
    .to_string()
)
