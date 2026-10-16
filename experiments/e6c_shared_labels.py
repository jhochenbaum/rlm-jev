"""E6c: how much of Jev's per-item cost is the label descriptions repeated in every question?

Same 500 TREC records as E6, one fresh (uncached) Jev pass per request format, 250 records per request:
  per_question : each question carries the six label descriptions (what E6 and the rest of the experiments use)
  in_state     : the descriptions are written once in the shared state; each question lists only the label names
  names_only   : label names only, no descriptions anywhere
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.client import LEDGER
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window
from rlm_jev.calibrate import ece

DEFAULT_STATE = "Classify each record independently."


async def main():
    swap = json.loads(Path("results/e6_labels.json").read_text())["swap"]
    texts, gold = swap["texts"], swap["gold"]
    desc = LABELS["trec_coarse"]
    names = dict.fromkeys(desc)
    in_state = DEFAULT_STATE + " The categories:\n" + "\n".join(f"- {k}: {v}" for k, v in desc.items())
    variants = {
        "per_question": {"options": desc, "context": None, "labels_in_state": False},
        "in_state": {"options": names, "context": in_state, "labels_in_state": True},
        "names_only": {"options": names, "context": None, "labels_in_state": True},
    }
    out = {"n": len(texts)}
    for name, v in variants.items():
        tag = f"e6c_{name}"
        t0 = time.time()
        P = await label_texts(
            texts,
            v["options"],
            batch=250,
            tag=tag,
            use_cache=False,
            context=v["context"],
            labels_in_state=v["labels_in_state"],
        )
        wall = time.time() - t0
        rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if f'"{tag}"' in l]
        rows = [r for r in rows if r["t"] >= t0]
        tokens = sum(r["input_tokens"] for r in rows)
        pred = [max(p, key=p.get) for p in P]
        correct = [a == g for a, g in zip(pred, gold, strict=True)]
        out[name] = {
            "acc": float(np.mean(correct)),
            "ece": round(ece([max(p.values()) for p in P], correct), 4),
            "tokens_per_item": round(tokens / len(texts), 1),
            "usd": round(sum(r["usd"] for r in rows), 5),
            "wall_s": round(wall, 2),
        }
        print(name, out[name], flush=True)
    # The same comparison on all 4,151 OOLONG TREC records, both option orders, as the other experiments label them
    df = load_val().query("dataset == 'trec_coarse'")
    pool = {
        x.text: x.label
        for r in df.drop_duplicates(["context_len", "context_window_id"]).itertuples()
        for x in parse_window(r.context_window_text_with_labels)
    }
    texts = sorted(pool)
    for name, shared in (("full_pool_per_question", False), ("full_pool_in_state", True)):
        kw = {"batch": 200, "tag": "e1_trec_coarse", "concurrency": 24, "labels_in_state": shared}
        pf = await label_texts(texts, desc, **kw)
        pr = await label_texts(texts, desc, order="reversed", **kw)
        res = {}
        for k, P in (
            ("single", pf),
            ("both_orders", [{c: (a[c] + b[c]) / 2 for c in a} for a, b in zip(pf, pr, strict=True)]),
        ):
            correct = [max(p, key=p.get) == pool[t] for p, t in zip(P, texts, strict=True)]
            res[k] = {
                "acc": round(float(np.mean(correct)), 4),
                "ece": round(ece([max(p.values()) for p in P], correct), 4),
            }
        out[name] = {"n": len(texts), **res}
        print(name, out[name], flush=True)
    Path("results/e6c_shared_labels.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
