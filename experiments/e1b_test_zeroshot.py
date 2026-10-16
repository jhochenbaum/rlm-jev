"""E1b: OOLONG-synth TEST split @131k, 8 unseen datasets, zero-tuning leaf:
options = bare label names, state = the task description paragraph copied from the context header."""

import asyncio
import collections
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.client import summary
from jevlab.label import label_texts
from jevlab.oolong import parse_window
from jevlab.oolong_program import answer, score


async def main(L=131072):
    df = pd.read_parquet("data/oolong_synth_test_32k_262k.parquet")
    df = df[df.context_len == L]
    rows, recacc = [], {}
    for ds, g in df.groupby("dataset"):
        wins = {w: parse_window(gg.iloc[0].context_window_text_with_labels) for w, gg in g.groupby("context_window_id")}
        labels = sorted({x.label for recs in wins.values() for x in recs})
        header = g.iloc[0].context_window_text.split("\n\n")[0]  # task description, verbatim from context
        texts = sorted({x.text for recs in wins.values() for x in recs})
        opts = dict.fromkeys(labels)
        kw = {
            "batch": 100,
            "concurrency": 24,
            "context": header,
            "inline_instruction": "Which label applies to `record`, according to the task described in the state?",
        }
        pf = await label_texts(texts, opts, tag=f"e1b_{ds}_fwd", **kw)
        pr = await label_texts(texts, opts, tag=f"e1b_{ds}_rev", order="reversed", **kw)
        P = {t: (a, {k: (a[k] + b[k]) / 2 for k in a}) for t, a, b in zip(texts, pf, pr, strict=False)}
        gold = {x.text: x.label for recs in wins.values() for x in recs}
        recacc[ds] = {
            n: np.mean([max(P[t][j], key=P[t][j].get) == gold[t] for t in texts])
            for n, j in [("jev", 0), ("jev_ens", 1)]
        }
        maj = collections.Counter(gold.values()).most_common(1)[0][0]
        for r in g.itertuples():
            recs = wins[r.context_window_id]
            leaves = {
                "gold": [x.label for x in recs],
                "majority": [maj] * len(recs),
                "jev": [max(P[x.text][0], key=P[x.text][0].get) for x in recs],
                "jev_ens": [max(P[x.text][1], key=P[x.text][1].get) for x in recs],
            }
            row = {"dataset": ds, "task": r.task}
            for k, labs in leaves.items():
                try:
                    row[k] = score(answer(r.question, recs, labs, labels), r.answer, r.answer_type)
                except NotImplementedError:
                    row[k] = np.nan
            rows.append(row)
        print(ds, len(texts), "records", {k: round(v, 3) for k, v in recacc[ds].items()}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("results/e1b_test_131k.csv", index=False)
    print(R.groupby("dataset")[["gold", "majority", "jev", "jev_ens"]].mean().round(3).to_string())
    print(
        "OVERALL", R[["gold", "majority", "jev", "jev_ens"]].mean().round(3).to_dict(), "unparsed", R.gold.isna().sum()
    )
    print(summary())


asyncio.run(main())
