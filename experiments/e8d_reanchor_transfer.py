"""E8d: do weights fitted through DSPy's request format (E8c's ReAnchor Choice weights) transfer to inline Jev
probabilities? Evaluated leak-free: E8c's 300 fit records are excluded from the record accuracy and removed from every
OOLONG window, with gold answers recomputed (as in E8g)."""

import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.oolong import load_val, parse_window
from jevlab.trec import L, fit_split, leakfree_oolong, probs

W = next(
    f["value"]
    for f in json.loads(Path("results/e8c_reanchor.json").read_text())["trec_choice_weights"]["fitted"]
    if f["parameter"] == "weights"
)


async def main():
    df = load_val().query("dataset == 'trec_coarse'")
    wins = {
        k: parse_window(g.iloc[0].context_window_text_with_labels)
        for k, g in df.groupby(["context_len", "context_window_id"])
    }
    gold = {x.text: x.label for recs in wins.values() for x in recs}
    P = await probs(sorted(gold), tag="e8d_label")
    fitset = set(fit_split(gold))  # the records E8c fitted its weights on
    held = [t for t in gold if t not in fitset]
    gc = collections.Counter(gold[t] for t in held)
    out = {"weights_from": "E8c DSPy-native ReAnchor fit", "heldout_n": len(held)}
    for name, w in [("unweighted", dict.fromkeys(L, 1.0)), ("dspy_weights", W)]:
        lab = {t: max(P[t], key=lambda k, t=t, w=w: P[t][k] * w[k]) for t in held}
        c = collections.Counter(lab.values())
        R = leakfree_oolong(df, wins, fitset, lab)
        out[name] = {
            "record_acc": float(np.mean([lab[t] == gold[t] for t in held])),
            "count_relerr": float(np.mean([abs(c[k] - gc[k]) / gc[k] for k in L])),
            "oolong_overall": float(R.s.mean()),
            "oolong_131k": float(R[R.L == 131072].s.mean()),
        }
        print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    Path("results/e8d_reanchor_transfer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
