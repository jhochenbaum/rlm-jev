"""E8g: (1) is OOLONG's TREC pool a selected/easier subset? (2) leak-free in-distribution weight fit:
fit on 300 pool records, delete them from every window, recompute gold with the validated program, re-score."""

import asyncio
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.oolong import load_val, parse_window
from jevlab.trec import COARSE, L, fit, fit_split, leakfree_oolong, norm, probs

# --labels-in-state: the same fit with label descriptions written once in the shared state (E6c's cheaper format)
SHARED = "--labels-in-state" in sys.argv[1:]


async def main():
    import json

    OUT = {}
    df = load_val()
    df = df[df.dataset == "trec_coarse"]
    wins = {
        k: parse_window(g.iloc[0].context_window_text_with_labels)
        for k, g in df.groupby(["context_len", "context_window_id"])
    }
    val_gold = {x.text: x.label for recs in wins.values() for x in recs}
    val_norm = {norm(t) for t in val_gold}
    # (1) TREC_10 official test set: in-pool vs out-of-pool accuracy
    t10 = [
        (q.strip(), COARSE[lab.split(":")[0]])
        for lab, q in (
            l.split(" ", 1) for l in Path("data/trec/TREC_10.label").read_text(encoding="latin-1").splitlines()
        )
    ]
    P10 = await probs([q for q, _ in t10], labels_in_state=SHARED)
    inp = [(q, g) for q, g in t10 if norm(q) in val_norm]
    outp = [(q, g) for q, g in t10 if norm(q) not in val_norm]
    a = lambda xs: np.mean([max(P10[q], key=P10[q].get) == g for q, g in xs])
    OUT["trec10"] = {
        "in_pool_n": len(inp),
        "in_pool_acc": float(a(inp)),
        "out_pool_n": len(outp),
        "out_pool_acc": float(a(outp)),
    }
    print(f"TREC_10 test: in OOLONG pool n={len(inp)} acc={a(inp):.3f} | not in pool n={len(outp)} acc={a(outp):.3f}")
    # does OOLONG's label agree with original TREC label for in-pool items?
    agree = np.mean([val_gold[next(t for t in val_gold if norm(t) == norm(q))] == g for q, g in inp[:200]])
    print(f"OOLONG label == original TREC label on in-pool items: {agree:.3f}")
    # (2) leak-free in-distribution fit
    Pval = await probs(sorted(val_gold), labels_in_state=SHARED)
    fit_items = fit_split(val_gold)
    fitset = set(fit_items)
    W = fit(Pval, val_gold, fit_items)
    OUT["weights"] = W
    print("weights fitted on 300 pool records:", {k[:6]: v for k, v in W.items()})
    ev = [t for t in val_gold if t not in fitset]
    gc = collections.Counter(val_gold[t] for t in ev)
    for name, w in [("unweighted", dict.fromkeys(L, 1.0)), ("weighted", W)]:
        lab = {t: max(Pval[t], key=lambda k: Pval[t][k] * w[k]) for t in ev}
        c = collections.Counter(lab.values())
        R = leakfree_oolong(df, wins, fitset, lab)
        OUT[name] = {
            "heldout_n": len(ev),
            "record_acc": float(np.mean([lab[t] == val_gold[t] for t in ev])),
            "count_relerr": float(np.mean([abs(c[k] - gc[k]) / gc[k] for k in L])),
            "oolong_overall": float(R.s.mean()),
            "oolong_131k": float(R[R.L == 131072].s.mean()),
        }
        print(
            f"{name:11s} held-out records n={len(ev)} acc {np.mean([lab[t] == val_gold[t] for t in ev]):.4f} | count relerr "
            f"{np.mean([abs(c[k] - gc[k]) / gc[k] for k in L]):.4f} | OOLONG(fit records removed) overall {R.s.mean():.3f} @131k {R[R.L == 131072].s.mean():.3f} | "
            + " ".join(f"{t[10:]}={v:.3f}" for t, v in R.groupby("task").s.mean().items())
        )

    Path(f"results/e8g_leakfree{'_shared' if SHARED else ''}.json").write_text(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
