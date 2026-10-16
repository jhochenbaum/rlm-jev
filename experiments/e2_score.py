"""Score the E2 live RLM runs from rlm_ledger.jsonl into results/e2_scored.csv."""

import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from jevlab.oolong import LABELS, e2_jobs, load_val
from jevlab.oolong_program import extract, score
from jevlab.pairs_program import f1
from jevlab.rlm import ledger_rows

led = ledger_rows("e2_")
jobs = e2_jobs(load_val().query("dataset == 'trec_coarse'"))
e1 = pd.read_csv("results/e1_oolong_val.csv").set_index("id")

rows, used = [], set()
for kind, leaf, r, question, context in jobs:
    pair = kind == "pairs32k"
    tag = f"e2_{kind}_{leaf}"
    hit = next(
        (
            i
            for i, x in enumerate(led)
            if i not in used and x["tag"] == tag and x["query"] == question[:300] and x["ctx_chars"] == len(context)
        ),
        None,
    )
    if hit is None:
        continue
    used.add(hit)
    x = led[hit]
    if pair:
        gold = {tuple(re.findall(r"\d+", s)) for s in r["answer"]}
        s = f1({tuple(m) for m in re.findall(r"\((\d+),\s*(\d+)\)", x["answer"] or "")}, gold)
        fixed, task = None, "pairs"
    else:
        s = score(extract(x["answer"], r.answer_type, list(LABELS["trec_coarse"])), r.answer, r.answer_type)
        fixed, task = e1.loc[r.id, "jev"], r.task
    rows.append(
        {
            "kind": kind,
            "leaf": leaf,
            "task": task,
            "score": s,
            "fixed_program_jev": fixed,
            "turns": x["turns"],
            "jev_items": x["jev_items"],
            "llm_calls": x["llm_calls"],
            "root_usd": x["root_usd_list"],
            "leaf_llm_usd": x["llm_usd_list"],
            "wall_s": x["wall_s"],
            "answer": (x["answer"] or "")[:60],
        }
    )
R = pd.DataFrame(rows)
R.to_csv("results/e2_scored.csv", index=False)
cols = ["score", "fixed_program_jev", "turns", "root_usd", "leaf_llm_usd", "wall_s"]
print(R.groupby(["kind", "leaf"])[cols].mean().round(3).to_string())
print(f"{len(R)}/{len(jobs)} jobs scored")
