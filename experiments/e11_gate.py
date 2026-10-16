"""E11: System-1 -> System-2 gate. Jev labels everything; records with top prob < t escalate to Haiku.
Haiku labels for all records are computed once and cached (so every threshold, incl. Haiku-only, is exact).
Datasets where Jev is weakest: OOLONG test @131k multinli + negation."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.claude_leaf import claude_label
from jevlab.label import label_texts
from jevlab.oolong import parse_window
from jevlab.oolong_program import answer, score

CACHE = Path("results/e11_haiku_labels.json")
INSTR = "Which label applies to `record`, according to the task described in the state?"


async def main():
    df = pd.read_parquet("data/oolong_synth_test_32k_262k.parquet")
    df = df[df.context_len == 131072]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for ds in ["multinli", "negation"]:
        g = df[df.dataset == ds]
        wins = {w: parse_window(gg.iloc[0].context_window_text_with_labels) for w, gg in g.groupby("context_window_id")}
        labels = sorted({x.label for recs in wins.values() for x in recs})
        header = g.iloc[0].context_window_text.split("\n\n")[0]
        texts = sorted({x.text for recs in wins.values() for x in recs})
        gold = {x.text: x.label for recs in wins.values() for x in recs}
        kw = {"batch": 100, "concurrency": 24, "context": header, "inline_instruction": INSTR, "tag": "e11_label"}
        pf = await label_texts(texts, dict.fromkeys(labels), **kw)
        pr = await label_texts(texts, dict.fromkeys(labels), order="reversed", **kw)
        P = {t: {l: (a[l] + b[l]) / 2 for l in labels} for t, a, b in zip(texts, pf, pr, strict=False)}
        todo = [t for t in texts if f"{ds}\x00{t}" not in cache]
        if todo:
            print(ds, "haiku labeling", len(todo), flush=True)
            hl = await claude_label(
                todo, dict.fromkeys(labels, ""), model="haiku", batch=100, concurrency=4, tag="e11_haiku", task=header
            )
            for t, h in zip(todo, hl, strict=False):
                cache[f"{ds}\x00{t}"] = h
            CACHE.write_text(json.dumps(cache))
        H = {t: cache[f"{ds}\x00{t}"] for t in texts}
        jev = {t: max(P[t], key=P[t].get) for t in texts}
        conf = {t: max(P[t].values()) for t in texts}
        for thr in [0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
            lab = {t: (H[t] or jev[t]) if conf[t] < thr else jev[t] for t in texts}
            esc = np.mean([conf[t] < thr for t in texts])
            qs = [
                score(
                    answer(
                        r.question, wins[r.context_window_id], [lab[x.text] for x in wins[r.context_window_id]], labels
                    ),
                    r.answer,
                    r.answer_type,
                )
                for r in g.itertuples()
            ]
            rows.append(
                {
                    "dataset": ds,
                    "thr": thr,
                    "escalated": round(esc, 3),
                    "record_acc": round(np.mean([lab[t] == gold[t] for t in texts]), 4),
                    "question_score": round(np.mean(qs), 3),
                    "haiku_parse_fail": sum(H[t] is None for t in texts),
                }
            )
    R = pd.DataFrame(rows)
    R.to_csv("results/e11_gate.csv", index=False)
    print(R.to_string())


asyncio.run(main())
