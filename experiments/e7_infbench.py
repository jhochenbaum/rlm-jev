"""E7: InfiniteBench En.MC (novel-length multiple choice). Jev as an attention compressor.

full      : Haiku reads the whole book (up to 700k words; books over its 1M-token window are recorded as
            full_too_long and handled in E7b)
jev_ctx   : Jev scores every ~500-word window against the question and each option (all queries share
            the window state -> one pass over the book), keeps the top windows (~8k words), Haiku answers.
"""

import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from typesafe_sdk import Noul

from jevlab.claude_cli import TOO_LONG
from jevlab.claude_leaf import claude_call
from jevlab.client import Jev, summary

WIN = 500
KEEP = 16  # 16 windows x 500 words = ~8k words
SPLIT_WORDS = 110_000  # question sets: A = books that fit a ~150k-token window, B = longer books
MAX_FULL_WORDS = 700_000  # at ~1.86 tokens/word, books over ~540k words exceed Haiku 5.5's 1M-token window


def pick():
    rows = [json.loads(l) for l in Path("data/longbook_choice_eng.jsonl").read_text().splitlines()]
    rng = random.Random(0)
    fit = [r for r in rows if len(r["context"].split()) <= SPLIT_WORDS]
    long = [r for r in rows if len(r["context"].split()) >= 250_000]
    return rng.sample(fit, 20), rng.sample(long, 8)


def mc_prompt(q, opts, ctx):
    o = "\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(opts))
    return f"Read the text, then answer the multiple-choice question.\n\n<text>\n{ctx}\n</text>\n\nQuestion: {q}\n{o}\n\nReply with the single letter of the correct option only."


def parse_letter(s, opts):
    m = re.search(r"\b([A-D])\b", s.strip())
    return opts[ord(m[1]) - 65] if m and ord(m[1]) - 65 < len(opts) else None


async def jev_select(jev, book, q, opts):
    words = book.split()
    wins = [" ".join(words[i : i + WIN]) for i in range(0, len(words), WIN)]
    queries = [f"Does this passage contain information that helps answer the question: {q}"] + [
        f'Does this passage contain evidence about whether the answer to "{q}" is: {o}' for o in opts
    ]
    S = np.zeros((len(queries), len(wins)))
    sem = asyncio.Semaphore(16)

    async def one(wi):
        qs = {f"q{k}": Noul(instructions=t) for k, t in enumerate(queries)}
        for attempt in range(5):  # Jev throttles bursts; the SDK's 10 s timeout surfaces as an error, so retry
            try:
                async with sem:
                    r = await jev.aask(wins[wi], qs, tag="e7_jev")
                break
            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        for k in range(len(queries)):
            S[k, wi] = r.answers[f"q{k}"].noul

    await asyncio.gather(*(one(wi) for wi in range(len(wins))))
    # round-robin: best windows per query until KEEP
    chosen, order = [], [list(np.argsort(-S[k])) for k in range(len(queries))]
    while len(chosen) < min(KEEP, len(wins)):
        for k in range(len(queries)):
            for wi in order[k]:
                if wi not in chosen:
                    chosen.append(wi)
                    break
            if len(chosen) >= KEEP:
                break
    chosen.sort()
    return "\n[...]\n".join(wins[i] for i in chosen), len(wins)


async def main():
    jev = Jev()
    setA, setB = pick()
    sem = asyncio.Semaphore(2)

    async def run(name, r):
        q, opts, gold = r["input"], r["options"], r["answer"][0]
        t0 = time.time()
        ctx, nw = await jev_select(jev, r["context"], q, opts)
        tj = time.time() - t0
        a_jev = parse_letter(await claude_call(mc_prompt(q, opts, ctx), model="haiku", tag=f"e7_{name}_jevctx"), opts)
        words = r["context"].split()
        full = " ".join(words[:MAX_FULL_WORDS])
        raw_full = await claude_call(mc_prompt(q, opts, full), model="haiku", tag=f"e7_{name}_full")
        full_too_long = raw_full.startswith(TOO_LONG)  # book exceeds Haiku's 1M-token window
        a_full = parse_letter(raw_full, opts)
        row = {
            "set": name,
            "id": r["id"],
            "book_words": len(words),
            "windows": nw,
            "jev_select_s": round(tj, 1),
            "ctx_words": len(ctx.split()),
            "jev_ok": a_jev == gold,
            "full_ok": a_full == gold,
            "full_too_long": full_too_long,
            "full_truncated": len(words) > MAX_FULL_WORDS,
        }
        print(json.dumps(row), flush=True)
        return row

    async def guarded(name, r):
        async with sem:
            try:
                return await run(name, r)
            except Exception as e:
                print("FAIL", repr(e)[:200], flush=True)

    out = []
    for name, rows in [("A_fits", setA), ("B_long", setB)]:
        out += [x for x in await asyncio.gather(*(guarded(name, r) for r in rows)) if x]
    await jev.aclose()
    import pandas as pd

    R = pd.DataFrame(out)
    R.to_csv("results/e7_infbench.csv", index=False)
    print(
        R.groupby("set")[["jev_ok", "full_ok", "book_words", "ctx_words", "jev_select_s"]].mean().round(3).to_string()
    )
    print(summary())


if __name__ == "__main__":
    asyncio.run(main())
