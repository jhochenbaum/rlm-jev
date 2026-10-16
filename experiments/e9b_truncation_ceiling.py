"""E9b: how much does truncation itself cost on OOLONG 1M? For the 20 questions in E9's truncated direct-read arms,
score a perfect reader (the fixed program on gold labels) given only the lines each truncation rule keeps.
No API calls. Rules: keep the most recent lines (what E9 ran), keep the earliest lines, keep a random sample."""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.oolong import LABELS, load_val, parse_window, truncate_keep_recent
from jevlab.oolong_program import answer, score

df = load_val()
trec = df[df.dataset == "trec_coarse"]
sub = trec[trec.context_len == 1048576]
rows = [json.loads(l) for l in Path("rlm_ledger.jsonl").read_text().splitlines() if '"e9_c2_direct1m_trunc"' in l]
out = {"recent": [], "earliest": [], "random": [], "full": []}
per_task = {}
for x in rows:
    r = sub[(sub.question.str[:300] == x["query"]) & (sub.context_window_text.str.len() == x["ctx_chars"])].iloc[0]
    recs = parse_window(r.context_window_text_with_labels)
    _, kept = truncate_keep_recent(r.context_window_text, x["ctx_chars_used"])
    k = round(kept * len(recs))
    rng = random.Random(0)
    keeps = {
        "full": recs,
        "recent": recs[-k:],
        "earliest": recs[:k],
        "random": sorted(rng.sample(recs, k), key=lambda q: q.idx),
    }
    for name, kr in keeps.items():
        s = score(answer(r.question, kr, [q.label for q in kr], list(LABELS["trec_coarse"])), r.answer, r.answer_type)
        out[name].append(s)
        per_task.setdefault((r.task, name), []).append(s)
res = {name: round(float(np.mean(v)), 3) for name, v in out.items()}
res["n"] = len(rows)
res["by_task_recent"] = {t: round(float(np.mean(v)), 3) for (t, n), v in per_task.items() if n == "recent"}
print(json.dumps(res, indent=1))
Path("results/e9b_truncation_ceiling.json").write_text(json.dumps(res, indent=1))
