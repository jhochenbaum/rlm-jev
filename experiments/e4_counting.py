"""E4: counting with calibrated Jev labels — hard vs soft vs confusion-corrected, plus interval coverage.
Reuses E1's cached Jev labels (no new API calls when E1 has run)."""

import asyncio
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window
from jevlab.oolong_program import score, subset


async def main():
    df = load_val()
    df = df[df.question.str.contains("how many data points should be classified as label")]
    windows = {}
    for r in df.drop_duplicates(["dataset", "context_len", "context_window_id"]).itertuples():
        windows[(r.dataset, r.context_len, r.context_window_id)] = parse_window(r.context_window_text_with_labels)
    P = {}
    for ds in LABELS:
        texts = sorted({x.text for (d, _, _), recs in windows.items() if d == ds for x in recs})
        pf = await label_texts(texts, LABELS[ds], batch=200, tag="e4_label")
        pr = await label_texts(texts, LABELS[ds], batch=200, tag="e4_label", order="reversed")
        for t, a, b in zip(texts, pf, pr, strict=False):
            P[(ds, t)] = {k: (a[k] + b[k]) / 2 for k in a}
    # labeled seed for confusion matrix: 300 random unique records per dataset (seeded)
    rng = random.Random(1)
    conf = {}
    seeds = {}
    for ds, opts in LABELS.items():
        gold = {x.text: x.label for (d, _, _), recs in windows.items() if d == ds for x in recs}
        seed = rng.sample(sorted(gold), 300)
        seeds[ds] = set(seed)
        L = list(opts)
        M = np.full((len(L), len(L)), 0.5)  # add-0.5 smoothing; M[true, pred]
        for t in seed:
            p = P[(ds, t)]
            M[L.index(gold[t]), L.index(max(p, key=p.get))] += 1
        conf[ds] = (L, M / M.sum(1, keepdims=True))
    rows = []
    for r in df.itertuples():
        recs = windows[(r.dataset, r.context_len, r.context_window_id)]
        L, C = conf[r.dataset]
        lab = re.findall(r"label '([^']+)'", r.question)[0]
        k = L.index(lab)
        idx = subset(r.question, recs)
        ps = [P[(r.dataset, recs[i].text)] for i in idx]
        hard = np.zeros(len(L))
        for p in ps:
            hard[L.index(max(p, key=p.get))] += 1
        soft = sum(p[lab] for p in ps)
        # adjusted classify-and-count: hard = C^T true  -> true = lstsq, clip >=0
        true_hat = np.clip(np.linalg.lstsq(C.T, hard, rcond=None)[0], 0, None)
        # 90% interval from Poisson-binomial (normal approx) on soft probs
        pk = np.array([p[lab] for p in ps])
        sd = np.sqrt((pk * (1 - pk)).sum())
        gold_n = sum(recs[i].label == lab for i in idx)
        rows.append(
            {
                "ds": r.dataset,
                "L": r.context_len,
                "n": len(idx),
                "gold": gold_n,
                "s_hard": score(str(int(hard[k])), r.answer, r.answer_type),
                "s_soft": score(str(round(soft)), r.answer, r.answer_type),
                "s_acc": score(str(round(true_hat[k])), r.answer, r.answer_type),
                "relerr_hard": abs(hard[k] - gold_n) / max(gold_n, 1),
                "relerr_soft": abs(soft - gold_n) / max(gold_n, 1),
                "relerr_acc": abs(true_hat[k] - gold_n) / max(gold_n, 1),
                "cover90": abs(soft - gold_n) <= 1.645 * sd,
            }
        )
    R = pd.DataFrame(rows)
    R.to_csv("results/e4_counting.csv", index=False)
    print(
        R.groupby("ds")[["s_hard", "s_soft", "s_acc", "relerr_hard", "relerr_soft", "relerr_acc", "cover90"]]
        .mean()
        .round(3)
        .to_string()
    )
    R["big"] = R.n > 500
    print(
        R.groupby(["ds", "big"])[["s_hard", "s_soft", "s_acc", "relerr_hard", "relerr_soft", "relerr_acc", "cover90"]]
        .mean()
        .round(3)
        .to_string()
    )


asyncio.run(main())
