"""E1: OOLONG-synth val, fixed (oracle-decomposition) program + leaf labels.

Label sources compared: gold labels (program ceiling), regex heuristic, Jev (single order),
Jev 2-order ensemble (fixed + reversed option order, probs averaged).
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.baselines import RULES
from jevlab.client import summary
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window
from jevlab.oolong_program import answer, score


async def main(maxlen):
    df = load_val()
    df = df[df.context_len <= maxlen]
    windows = {}
    for r in df.drop_duplicates(["dataset", "context_len", "context_window_id"]).itertuples():
        windows[(r.dataset, r.context_len, r.context_window_id)] = parse_window(r.context_window_text_with_labels)
    probs = {}
    for ds in ["spam", "trec_coarse"]:
        texts = sorted({x.text for (d, _, _), recs in windows.items() if d == ds for x in recs})
        t0 = time.time()
        pf = await label_texts(texts, LABELS[ds], batch=200, concurrency=24, tag=f"e1_{ds}_fwd")
        pr = await label_texts(texts, LABELS[ds], batch=200, concurrency=24, tag=f"e1_{ds}_rev", order="reversed")
        print(ds, len(texts), "unique texts labeled in", round(time.time() - t0, 1), "s", flush=True)
        for t, a, b in zip(texts, pf, pr, strict=False):
            probs[(ds, t)] = (a, {k: (a[k] + b[k]) / 2 for k in a})
    # per-record accuracy on unique texts
    rec_acc = {}
    for ds in ["spam", "trec_coarse"]:
        gold = {x.text: x.label for (d, _, _), recs in windows.items() if d == ds for x in recs}
        for name, j in [("jev", 0), ("jev_ens", 1)]:
            rec_acc[f"{ds}/{name}"] = np.mean(
                [max(probs[(ds, t)][j], key=probs[(ds, t)][j].get) == g for t, g in gold.items()]
            )
        rec_acc[f"{ds}/regex"] = np.mean([RULES[ds](t) == g for t, g in gold.items()])
    print("record accuracy:", {k: round(v, 4) for k, v in rec_acc.items()})
    # question-level
    rows = []
    for r in df.itertuples():
        recs = windows[(r.dataset, r.context_len, r.context_window_id)]
        ls = list(LABELS[r.dataset])
        leaves = {
            "gold": [x.label for x in recs],
            "regex": [RULES[r.dataset](x.text) for x in recs],
            "jev": [max(probs[(r.dataset, x.text)][0], key=probs[(r.dataset, x.text)][0].get) for x in recs],
            "jev_ens": [max(probs[(r.dataset, x.text)][1], key=probs[(r.dataset, x.text)][1].get) for x in recs],
        }
        row = {"id": r.id, "dataset": r.dataset, "context_len": r.context_len, "task": r.task, "n_records": len(recs)}
        for k, labs in leaves.items():
            row[k] = score(answer(r.question, recs, labs, ls), r.answer, r.answer_type)
        rows.append(row)
    import pandas as pd

    R = pd.DataFrame(rows)
    R.to_csv("results/e1_oolong_val.csv", index=False)
    print(R.groupby(["dataset", "context_len"])[["gold", "regex", "jev", "jev_ens"]].mean().round(3).to_string())
    print(R.groupby("dataset")[["gold", "regex", "jev", "jev_ens"]].mean().round(3).to_string())
    print(R.groupby("task")[["gold", "regex", "jev", "jev_ens"]].mean().round(3).to_string())
    Path("results/e1_record_acc.json").write_text(json.dumps({k: float(v) for k, v in rec_acc.items()}, indent=1))
    print(summary())


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4194304))
