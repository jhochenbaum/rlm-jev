"""E8f: does a weight fit transfer across TREC subsets? Fit Choice weights on original TREC records that never
appear in any OOLONG validation window, then evaluate on the OOLONG validation records and questions (all unseen by
the fit)."""

import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.client import summary
from jevlab.oolong import load_val, parse_window
from jevlab.oolong_program import answer, score
from jevlab.trec import L, external_records, external_sample, fit, probs


async def main():
    OUT = {}
    df = load_val()
    df = df[df.dataset == "trec_coarse"]
    wins = {
        k: parse_window(g.iloc[0].context_window_text_with_labels)
        for k, g in df.groupby(["context_len", "context_window_id"])
    }
    val_gold = {x.text: x.label for recs in wins.values() for x in recs}
    ext = external_records(val_gold)
    print(f"original TREC records disjoint from OOLONG val pool: {len(ext)} (val pool {len(val_gold)})")
    fit_pool = external_sample(ext)
    Pext = await probs(fit_pool, tag="e8f_label")
    Pval = await probs(sorted(val_gold), tag="e8f_label")
    ext_acc = np.mean([max(Pext[t], key=Pext[t].get) == ext[t] for t in fit_pool])
    print(f"Jev acc on external fit pool (unweighted): {ext_acc:.4f}")
    OUT["external_pool"] = {"n_disjoint_available": len(ext), "n_sampled": len(fit_pool), "jev_acc": float(ext_acc)}
    gold_counts = collections.Counter(val_gold.values())

    def evaluate(W, name):
        lab = {t: max(p, key=lambda k: p[k] * W[k]) for t, p in Pval.items()}
        c = collections.Counter(lab.values())
        rec_acc = np.mean([lab[t] == g for t, g in val_gold.items()])
        relerr = np.mean([abs(c[k] - gold_counts[k]) / gold_counts[k] for k in L])
        qs = [
            score(
                answer(
                    r.question,
                    wins[(r.context_len, r.context_window_id)],
                    [lab[x.text] for x in wins[(r.context_len, r.context_window_id)]],
                    L,
                ),
                r.answer,
                r.answer_type,
            )
            for r in df.itertuples()
        ]
        R = pd.DataFrame({"task": df.task.values, "L": df.context_len.values, "s": qs})
        OUT[name] = {
            "weights": W,
            "valpool_record_acc": float(rec_acc),
            "count_relerr": float(relerr),
            "oolong_overall": float(R.s.mean()),
            "oolong_131k": float(R[R.L == 131072].s.mean()),
        }
        print(
            f"{name:22s} W={ {k[:6]: v for k, v in W.items()} }\n   val-pool record acc {rec_acc:.4f} | count relerr {relerr:.4f} | OOLONG overall {R.s.mean():.3f} | @131k {R[R.L == 131072].s.mean():.3f} | "
            + " ".join(f"{t[10:]}={v:.3f}" for t, v in R.groupby("task").s.mean().items())
        )

    evaluate(dict.fromkeys(L, 1.0), "unweighted")
    for n in [300, 1000]:
        evaluate(fit(Pext, ext, fit_pool[:n]), f"fit on {n} external")
    Path("results/e8f_disjoint_weights.json").write_text(json.dumps(OUT, indent=1))
    print(summary().splitlines()[0])


asyncio.run(main())
