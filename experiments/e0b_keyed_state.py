"""E0b: is the E0 state-mode collapse positional indexing, or the state itself?
Same 400 TREC records as E0, packed into the state under visible IDs (r0, r1, ...) and asked about by ID."""

import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
from jevlab.label import label_texts
from jevlab.oolong import LABELS, load_val, parse_window


async def main():
    df = load_val()
    random.seed(0)
    res = {}
    ds = "trec_coarse"
    row = df[(df.dataset == ds) & (df.context_len == 131072)].iloc[0]
    recs = random.sample(
        [r for r in parse_window(row.context_window_text_with_labels) if r.label], 400
    )  # as in e0_pilot.py
    texts, opts = [r.text for r in recs], LABELS[ds]
    for mode in ("keyed", "keyed_lines"):
        for b in (1, 25, 100, 250):
            probs = await label_texts(texts, opts, batch=b, mode=mode, tag=f"e0b_{mode}_B{b}", concurrency=12)
            acc = sum(max(p, key=p.get) == r.label for p, r in zip(probs, recs, strict=False)) / len(recs)
            res[f"{mode}_B{b}"] = round(acc, 3)
            print(mode, b, res[f"{mode}_B{b}"], flush=True)
    Path("results/e0b_keyed_state.json").write_text(json.dumps(res, indent=1))


asyncio.run(main())
