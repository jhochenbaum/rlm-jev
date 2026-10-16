"""E2: live RLM runs. A Claude Sonnet root writes REPL code over the raw context; sub-calls go to Jev, or to an LLM
(Claude Haiku) for comparison. Resumable: questions already in rlm_ledger.jsonl are skipped. Score with e2_score.py."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from jevlab.oolong import e2_jobs, load_val
from jevlab.rlm import ledger_rows, run_rlm

jobs = e2_jobs(load_val().query("dataset == 'trec_coarse'"))

done = ledger_rows("e2_")


def run(job):
    kind, leaf, _, question, context = job
    tag = f"e2_{kind}_{leaf}"
    hit = next(
        (x for x in done if x["tag"] == tag and x["query"] == question[:300] and x["ctx_chars"] == len(context)), None
    )
    if hit is not None:
        done.remove(hit)
        return
    try:
        out = run_rlm(context, question, root_model="sonnet", leaf=leaf, tag=tag)
        print(json.dumps({"tag": tag, "turns": out["turns"], "answer": (out["answer"] or "")[:60]}), flush=True)
    except Exception as e:
        print("JOB FAILED", tag, repr(e)[:300], flush=True)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(run, jobs))
