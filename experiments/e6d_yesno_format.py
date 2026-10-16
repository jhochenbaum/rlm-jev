"""E6d: yes/no questions with the question text in every item's question vs once in the shared state.

Two binary tasks on OOLONG records, one fresh (uncached) Jev pass per format, 250 items per request:
  spam   : "Is this text message spam?" on 1,000 spam/ham records
  person : "Does this question ask for a person, group of people or organization?" on 1,000 TREC records
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from typesafe_sdk import Noul

sys.path.insert(0, ".")
from jevlab.client import LEDGER, Jev
from jevlab.oolong import load_val, parse_window
from rlm_jev.calibrate import ece

STATE = "Answer the question about each item independently."
TASKS = {
    "spam": ("spam", "Is this text message spam (unsolicited promotional, prize, scam or advertising text)?", "spam"),
    "person": (
        "trec_coarse",
        "Does this question ask for a person, group of people or organization?",
        "human being",
    ),
}


def records(ds, n=1000, seed=3):
    df = load_val().query("dataset == @ds")
    g = {
        x.text: x.label
        for r in df.drop_duplicates(["context_len", "context_window_id"]).itertuples()
        for x in parse_window(r.context_window_text_with_labels)
    }
    texts = random.Random(seed).sample(sorted(g), n)
    return texts, [g[t] for t in texts]


async def run(jev, texts, question, fmt, tag):
    if fmt == "per_question":
        state = STATE
        make = lambda t: Noul(instructions={"item": t, "question": question})
    else:
        state = f"{STATE}\nThe question for every item: {question}"
        make = lambda t: Noul(instructions={"item": t})
    out = [0.0] * len(texts)

    async def one(lo):
        qs = {f"q{j}": make(t) for j, t in enumerate(texts[lo : lo + 250])}
        r = await jev.aask(state, qs, tag=tag)
        for j in range(len(qs)):
            out[lo + j] = r.answers[f"q{j}"].noul

    await asyncio.gather(*(one(lo) for lo in range(0, len(texts), 250)))
    return out


async def main():
    res = {}
    async with Jev() as jev:
        for task, (ds, question, positive) in TASKS.items():
            texts, labels = records(ds)
            y = np.array([lab == positive for lab in labels])
            for fmt in ("per_question", "in_state"):
                tag = f"e6d_{task}_{fmt}"
                t0 = time.time()
                p = np.array(await run(jev, texts, question, fmt, tag))
                wall = time.time() - t0
                rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if f'"{tag}"' in l]
                rows = [r for r in rows if r["t"] >= t0]
                pred = p > 0.5
                conf = np.where(pred, p, 1 - p)
                res[f"{task}/{fmt}"] = {
                    "n": len(texts),
                    "positives": int(y.sum()),
                    "acc": round(float(((p > 0.5) == y).mean()), 4),
                    "ece": round(ece(conf, pred == y), 4),
                    "tokens_per_item": round(sum(r["input_tokens"] for r in rows) / len(texts), 1),
                    "usd": round(sum(r["usd"] for r in rows), 5),
                    "wall_s": round(wall, 2),
                }
                print(task, fmt, res[f"{task}/{fmt}"], flush=True)
    Path("results/e6d_yesno_format.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
