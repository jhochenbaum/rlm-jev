"""Pilot: Jev per-record labeling accuracy/calibration vs batch size and option order."""

import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import itertools

from jevlab.client import summary
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window


def metrics(recs, probs, opts):
    y = [r.label for r in recs]
    pred = [max(p, key=p.get) for p in probs]
    acc = np.mean([a == b for a, b in zip(pred, y, strict=False)])
    conf = np.array([max(p.values()) for p in probs])
    corr = np.array([a == b for a, b in zip(pred, y, strict=False)])
    bins = np.linspace(0, 1, 11)
    ece = 0
    for lo, hi in itertools.pairwise(bins):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
    brier = np.mean([sum((p.get(o, 0) - (o == t)) ** 2 for o in opts) for p, t in zip(probs, y, strict=False)])
    # aggregate count error per label: hard counts vs expected (sum of probs)
    true_c = {o: sum(t == o for t in y) for o in opts}
    hard = {o: sum(pp == o for pp in pred) for o in opts}
    soft = {o: sum(p.get(o, 0) for p in probs) for o in opts}
    rel = lambda c: np.mean([abs(c[o] - true_c[o]) / max(true_c[o], 1) for o in opts])
    return {
        "acc": round(acc, 3),
        "ece": round(ece, 3),
        "brier": round(brier, 3),
        "cnt_err_hard": round(rel(hard), 3),
        "cnt_err_soft": round(rel(soft), 3),
    }


async def main():
    df = load_val()
    random.seed(0)
    res = {}
    for ds in ["trec_coarse", "spam"]:
        row = df[(df.dataset == ds) & (df.context_len == 131072)].iloc[0]
        recs = [r for r in parse_window(row.context_window_text_with_labels) if r.label]
        recs = random.sample(recs, 400)
        opts = LABELS[ds]
        assert all(r.label in opts for r in recs), {r.label for r in recs}
        texts = [r.text for r in recs]
        for name, kw in [
            ("B1", {"batch": 1, "mode": "state"}),
            ("B25", {"batch": 25, "mode": "state"}),
            ("B100", {"batch": 100, "mode": "state"}),
            ("B250", {"batch": 250, "mode": "state"}),
            ("B25_rev", {"batch": 25, "order": "reversed", "mode": "state"}),
            ("in_B1", {"batch": 1}),
            ("in_B25", {"batch": 25}),
            ("in_B100", {"batch": 100}),
            ("in_B250", {"batch": 250}),
            ("in_B100_rev", {"batch": 100, "order": "reversed"}),
        ]:
            probs = await label_texts(texts, opts, tag=f"e0_{ds}_{name}", concurrency=12, labels_in_state=False, **kw)
            m = metrics(recs, probs, opts)
            res[f"{ds}/{name}"] = m
            print(ds, name, m, flush=True)
        print(ds, "gold dist", {o: sum(r.label == o for r in recs) for o in opts})
    Path("results/e0_pilot.json").write_text(json.dumps(res, indent=1))
    print(summary())


asyncio.run(main())
