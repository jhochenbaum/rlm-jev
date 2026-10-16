"""E6: leaf swap (Claude haiku labels vs Jev) + System-1->System-2 cascade on TREC unique records."""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.claude_leaf import claude_label
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window

TASK = "Each record is a general-knowledge question. Label it by the category of its expected ANSWER (TREC coarse question classification)."


async def main(thr=0.9):
    ds = "trec_coarse"
    df = load_val()
    gold = {}
    for r in df[df.dataset == ds].drop_duplicates(["context_len", "context_window_id"]).itertuples():
        for x in parse_window(r.context_window_text_with_labels):
            gold[x.text] = x.label
    texts = sorted(gold)
    pf = await label_texts(texts, LABELS[ds], batch=200, tag="e6_label")
    pr = await label_texts(texts, LABELS[ds], batch=200, tag="e6_label", order="reversed")
    P = {t: {k: (a[k] + b[k]) / 2 for k in a} for t, a, b in zip(texts, pf, pr, strict=False)}
    jev = {t: max(p, key=p.get) for t, p in P.items()}
    # leaf swap on random 500
    random.seed(7)
    sample = random.sample(texts, 500)
    t0 = time.time()
    fresh = await label_texts(sample, LABELS[ds], batch=250, tag="e6_swap_jev", use_cache=False)
    jev_dt = time.time() - t0
    prev = Path("results/e6_labels.json")
    prev = json.loads(prev.read_text())["swap"] if prev.exists() else {}
    if prev.get("texts") == sample:  # Haiku's labels for these 500 don't depend on Jev; reuse the measured run
        cl, haiku_dt = prev["haiku"], prev["haiku_wall_s"]
    else:
        t0 = time.time()
        cl = await claude_label(sample, LABELS[ds], model="haiku", task=TASK, tag="e6_swap_haiku")
        haiku_dt = time.time() - t0
    ok = [c is not None for c in cl]
    print(
        f"LEAF SWAP n=500: haiku acc={np.mean([c == gold[t] for c, t in zip(cl, sample, strict=False)]):.3f} parse_ok={np.mean(ok):.3f} wall={haiku_dt:.1f}s | jev acc={np.mean([jev[t] == gold[t] for t in sample]):.3f}"
    )
    # cascade
    low = [t for t in texts if max(P[t].values()) < thr]
    t0 = time.time()
    esc = await claude_label(low, LABELS[ds], model="haiku", task=TASK, tag="e6_cascade_haiku")
    dt = time.time() - t0
    casc = dict(jev)
    for t, c in zip(low, esc, strict=False):
        if c:
            casc[t] = c
    print(f"CASCADE thr={thr}: escalated {len(low)}/{len(texts)} ({len(low) / len(texts):.1%}) wall={dt:.1f}s")
    print(
        f"  jev acc={np.mean([jev[t] == gold[t] for t in texts]):.4f}  cascade acc={np.mean([casc[t] == gold[t] for t in texts]):.4f}"
    )
    print(
        f"  on escalated set: jev={np.mean([jev[t] == gold[t] for t in low]):.3f} haiku={np.mean([casc[t] == gold[t] for t in low]):.3f}"
    )
    out = {
        "swap": {
            "texts": sample,
            "gold": [gold[t] for t in sample],
            "haiku": cl,
            "jev_fwd": [pf[texts.index(t)] for t in sample],
            "jev_rev": [pr[texts.index(t)] for t in sample],
            "jev_fresh": fresh,
            "haiku_wall_s": round(haiku_dt, 1),
            "jev_wall_s": round(jev_dt, 2),
        },
        "cascade": {
            "threshold": thr,
            "n": len(texts),
            "escalated": len(low),
            "jev_acc": float(np.mean([jev[t] == gold[t] for t in texts])),
            "cascade_acc": float(np.mean([casc[t] == gold[t] for t in texts])),
        },
    }
    Path("results/e6_labels.json").write_text(json.dumps(out))


if __name__ == "__main__":
    asyncio.run(main())
