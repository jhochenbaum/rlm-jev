"""E7b: Haiku 5.5 on the InfiniteBench books too long for its 1M-token window, truncated to fit by keeping the most
recent text (the end of the book), as in the RLM blog's truncated baseline. Complements E7, where these books were
marked full_too_long."""

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from e7_infbench import mc_prompt, parse_letter

from jevlab.claude_cli import TOO_LONG
from jevlab.claude_leaf import claude_call

TOKENS_PER_WORD = 1.86  # measured in E7: 352k-word books came to ~656k Claude tokens
MAX_TOKENS = 900_000


async def main():
    e7 = pd.read_csv("results/e7_infbench.csv")
    ids = set(e7[e7.full_too_long].id)
    rows = {
        r["id"]: r
        for r in (json.loads(l) for l in Path("data/longbook_choice_eng.jsonl").read_text().splitlines())
        if r["id"] in ids
    }
    out = []
    for i in sorted(ids):
        r = rows[i]
        q, opts, gold = r["input"], r["options"], r["answer"][0]
        words = r["context"].split()
        for attempt in range(4):
            keep = int(MAX_TOKENS / TOKENS_PER_WORD * 0.9**attempt)
            raw = await claude_call(mc_prompt(q, opts, " ".join(words[-keep:])), model="haiku", tag="e7b_trunc")
            if not raw.startswith(TOO_LONG):
                break
        else:
            raise RuntimeError(f"{i}: still too long after keeping {keep:,} words")
        a = parse_letter(raw, opts)
        out.append(
            {
                "id": i,
                "book_words": len(words),
                "kept_words": min(keep, len(words)),
                "kept_share": round(min(keep, len(words)) / len(words), 3),
                "trunc_ok": a == gold,
            }
        )
        print(json.dumps(out[-1]), flush=True)
    pd.DataFrame(out).to_csv("results/e7b_truncated.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
