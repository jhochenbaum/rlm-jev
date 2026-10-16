"""E10: calibrated abstention for RLM answers. Confidence = agreement of the point answer with answers recomputed
from labels sampled from Jev's per-record probabilities (Monte Carlo). Risk-coverage on OOLONG TEST @131k
(8 unseen datasets, zero-tuning leaf; reuses E1b's cached labels)."""

import asyncio
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.label import label_texts
from jevlab.oolong import parse_window
from jevlab.oolong_program import answer, score

S = 60
INSTR = "Which label applies to `record`, according to the task described in the state?"


def work(args):
    q, recs, P, labels, gold_raw, atype, seed = args
    rng = np.random.default_rng(seed)
    point = answer(q, recs, [labels[int(np.argmax(p))] for p in P], labels)
    cum = np.cumsum(P, axis=1)
    u = rng.random((S, len(recs)))
    agree = 0
    for s in range(S):
        idx = (u[s][:, None] > cum).sum(1).clip(max=len(labels) - 1)
        agree += answer(q, recs, [labels[i] for i in idx], labels) == point
    return score(point, gold_raw, atype), agree / S


async def main():
    df = pd.read_parquet("data/oolong_synth_test_32k_262k.parquet")
    df = df[df.context_len == 131072]
    jobs = []
    for ds, g in df.groupby("dataset"):
        wins = {w: parse_window(gg.iloc[0].context_window_text_with_labels) for w, gg in g.groupby("context_window_id")}
        labels = sorted({x.label for recs in wins.values() for x in recs})
        header = g.iloc[0].context_window_text.split("\n\n")[0]
        texts = sorted({x.text for recs in wins.values() for x in recs})
        kw = {"batch": 100, "concurrency": 24, "context": header, "inline_instruction": INSTR, "tag": "e10_label"}
        pf = await label_texts(texts, dict.fromkeys(labels), **kw)
        pr = await label_texts(texts, dict.fromkeys(labels), order="reversed", **kw)
        PP = {t: np.array([(a[l] + b[l]) / 2 for l in labels]) for t, a, b in zip(texts, pf, pr, strict=False)}
        for i, r in enumerate(g.itertuples()):
            recs = wins[r.context_window_id]
            P = np.stack([PP[x.text] for x in recs])
            P = P / P.sum(1, keepdims=True)
            jobs.append((r.question, recs, P, labels, r.answer, r.answer_type, i))
        print(ds, "queued", flush=True)
    with Pool(8) as pool:
        res = pool.map(work, list(jobs), chunksize=4)
    R = pd.DataFrame(res, columns=["score", "conf"])
    R["dataset"] = [d for d, g in df.groupby("dataset") for _ in range(len(g))]
    R.to_csv("results/e10_abstain.csv", index=False)
    R = R.sort_values("conf", ascending=False).reset_index(drop=True)
    print(f"n={len(R)}  overall score {R.score.mean():.3f}")
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = round(cov * len(R))
        print(f"  coverage {cov:.0%}: score {R.score[:k].mean():.3f}")
    bins = pd.cut(R.conf, [0, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
    print(R.groupby(bins, observed=True).score.agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    asyncio.run(main())
