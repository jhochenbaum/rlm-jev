"""E3: OOLONG-Pairs with the fixed program: Jev labels every line once (both option orders), the join runs in code,
and the returned pairs are scored by F1 against gold, alongside gold-label and regex baselines (labels are cached, so records
labeled by earlier experiments cost nothing)."""

import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.baselines import trec_rule
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window
from jevlab.pairs_program import f1, solve


async def main():
    df = load_val()
    allres = {}
    for L in [32768, 131072]:
        gold_q = json.loads(Path(f"data/oolong-pairs-{L}.json").read_text())
        row = df[(df.dataset == "trec_coarse") & (df.context_len == L)].iloc[0]
        recs = parse_window(row.context_window_text_with_labels)
        texts = [r.text for r in recs]
        pf = await label_texts(texts, LABELS["trec_coarse"], batch=200, tag="e3_label")
        pr = await label_texts(texts, LABELS["trec_coarse"], batch=200, tag="e3_label", order="reversed")
        leaves = {
            "gold": [r.label for r in recs],
            "regex": [trec_rule(t) for t in texts],
            "jev": [max(p, key=p.get) for p in pf],
            "jev_ens": [max(a, key=lambda k: a[k] + b[k]) for a, b in zip(pf, pr, strict=False)],
        }
        res = {k: [] for k in leaves}
        for g in gold_q:
            G = {tuple(re.findall(r"\d+", s)) for s in g["answer"]}
            for k, labs in leaves.items():
                res[k].append(f1(solve(g["question"], recs, labs), G))
        print(f"OOLONG-Pairs L={L} records={len(recs)} users={len({r.user for r in recs})}")
        for k, v in res.items():
            print(f"  {k:8s} mean F1={np.mean(v) * 100:5.1f}  per-q: {[round(x * 100) for x in v]}")
        allres[L] = {k: {"mean_f1": float(np.mean(v)), "per_q": [float(x) for x in v]} for k, v in res.items()}
    Path("results/e3_pairs.json").write_text(json.dumps(allres, indent=1))


asyncio.run(main())
