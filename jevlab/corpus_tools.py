"""Jev search tools over a document corpus (for BrowseComp-Plus style RLM runs).

make_tools(docs, budget_usd) -> (REPL functions by name, spend dict, number of windows):
  jev_search(queries, within=None, k=8) -> {query: [{"doc", "win", "p", "snippet"}]}
      One shared-state pass: each request = one window of one document (state) + one Noul per query,
      so every query costs only ~35 extra tokens per window. `within` restricts the pass to some doc indices.
  jev_scout(queries, k=30) -> {query: [...]}   ranks documents by their first 1,500 characters (cheap first pass)
  read(doc, win=None) -> str   full text of a window (or whole document)
  jev_budget() -> str          Jev dollars spent / remaining for this question
A hard per-question budget is enforced: when it would be exceeded, jev_search raises instead of calling Jev.
"""

from __future__ import annotations

import asyncio

from typesafe_sdk import Noul

from .client import PRICE_PER_TOKEN, Jev

WIN_CHARS = 8000  # ~1,400 words per window


def split_windows(text: str) -> list[str]:
    if len(text) <= WIN_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + WIN_CHARS)
        if j < len(text):  # break on a paragraph or sentence boundary when possible
            cut = max(text.rfind("\n", i + WIN_CHARS // 2, j), text.rfind(". ", i + WIN_CHARS // 2, j))
            j = cut + 1 if cut > 0 else j
        out.append(text[i:j])
        i = j
    return out


def make_tools(docs: list[str], budget_usd: float = 0.60, tag: str = "bcp_leaf"):
    wins = [(d, w, t) for d, doc in enumerate(docs) for w, t in enumerate(split_windows(doc))]
    spent = {"usd": 0.0}

    def jev_search(queries, within=None, k=8):
        queries = [str(q) for q in ([queries] if isinstance(queries, str) else queries)]
        sel = [x for x in wins if within is None or x[0] in set(within)]
        est = sum(len(t) / 4 + 35 * len(queries) + 60 for _, _, t in sel) * PRICE_PER_TOKEN
        if spent["usd"] + est > budget_usd:
            raise RuntimeError(
                f"Jev budget would be exceeded: spent ${spent['usd']:.3f}, this call ~${est:.3f}, "
                f"budget ${budget_usd}. Narrow `within=` to fewer documents."
            )
        scores = [[0.0] * len(queries) for _ in sel]

        async def go():
            async with Jev() as jev:
                sem = asyncio.Semaphore(48)

                async def one(i):
                    qs = {
                        f"q{j}": Noul(instructions=f"Does this passage contain information that helps answer: {q}")
                        for j, q in enumerate(queries)
                    }
                    async with sem:
                        r = await jev.aask(sel[i][2], qs, tag=tag)
                    spent["usd"] += (r.usage.input_tokens or 0) * PRICE_PER_TOKEN
                    for j in range(len(queries)):
                        scores[i][j] = r.answers[f"q{j}"].noul

                await asyncio.gather(*(one(i) for i in range(len(sel))))

        asyncio.run(go())
        out = {}
        for j, q in enumerate(queries):
            top = sorted(range(len(sel)), key=lambda i: -scores[i][j])[:k]
            out[q] = [
                {
                    "doc": sel[i][0],
                    "win": sel[i][1],
                    "p": round(scores[i][j], 3),
                    "snippet": sel[i][2][:400].replace("\n", " "),
                }
                for i in top
            ]
        return out

    def jev_scout(queries, k=30):
        """Cheap first pass: scores only the first 1,500 characters of each document (~$0.02 per call)."""
        queries = [str(q) for q in ([queries] if isinstance(queries, str) else queries)]
        heads = [d[:1500] for d in docs]
        est = sum(len(h) / 4 + 35 * len(queries) + 60 for h in heads) * PRICE_PER_TOKEN
        if spent["usd"] + est > budget_usd:
            raise RuntimeError(f"Jev budget would be exceeded: spent ${spent['usd']:.3f}, this call ~${est:.3f}.")
        scores = [[0.0] * len(queries) for _ in heads]

        async def go():
            async with Jev() as jev:
                sem = asyncio.Semaphore(48)

                async def one(i):
                    qs = {
                        f"q{j}": Noul(
                            instructions=f"Is this document likely to contain information that helps answer: {q}"
                        )
                        for j, q in enumerate(queries)
                    }
                    async with sem:
                        r = await jev.aask(heads[i], qs, tag=tag)
                    spent["usd"] += (r.usage.input_tokens or 0) * PRICE_PER_TOKEN
                    for j in range(len(queries)):
                        scores[i][j] = r.answers[f"q{j}"].noul

                await asyncio.gather(*(one(i) for i in range(len(heads))))

        asyncio.run(go())
        return {
            q: [
                {"doc": i, "p": round(scores[i][j], 3), "head": heads[i][:200].replace("\n", " ")}
                for i in sorted(range(len(heads)), key=lambda i: -scores[i][j])[:k]
            ]
            for j, q in enumerate(queries)
        }

    def read(doc, win=None):
        if win is None:
            return docs[doc]
        return next(t for d, w, t in wins if d == doc and w == win)

    def jev_budget():
        return f"Jev spent ${spent['usd']:.3f} of ${budget_usd} for this question"

    return {"jev_search": jev_search, "jev_scout": jev_scout, "read": read, "jev_budget": jev_budget}, spent, len(wins)


SYSTEM_BCP = """You are the root of a Recursive Language Model answering a hard multi-hop research question.
You operate a persistent Python REPL. The variable `docs` is a list of 1,000 web documents (strings) - far too
much to read. Never print whole documents; print short slices.

Tools (calibrated relevance model Jev; fast and parallel). Budget per question is about $0.60; a full
jev_search pass over ALL documents costs ~$0.30, so prefer jev_scout and `within=` shortlists:
  jev_search(queries: list[str], within: list[int] | None = None, k: int = 8) -> dict
      Scores every ~1,500-word window of every document (or only docs in `within`) against EACH query in one
      pass, returns the top-k windows per query as {"doc", "win", "p", "snippet"}. Batch several sub-questions
      into one call: extra queries are almost free, extra passes are not. Use `within=` for cheap follow-ups.
  jev_scout(queries: list[str], k: int = 30) -> dict
      CHEAP (~$0.02): scores only the first 1,500 characters of each document. Use it first to shortlist
      documents, then run jev_search(..., within=shortlist) on their full text.
  read(doc: int, win: int | None = None) -> str     full text of one window (or the whole document)
  jev_budget() -> str                               Jev spend so far for this question
You can also use Python (re, string search) over `docs` freely.

Strategy hint: decompose the question into its clues, search for several clues at once, read the best
windows, extract candidate entities, then verify candidates against the remaining clues.
Reply with exactly one ```python ...``` block per turn; you will see its output. When confident, call
FINAL("<short exact answer>")."""


# Ablation prompt: identical except that the Jev tools are removed (the RLM paper's "without sub-calls" condition).
SYSTEM_BCP_NOJEV = (
    SYSTEM_BCP.split("Tools (calibrated")[0]
    + """Tools:
  read(doc: int, win: int | None = None) -> str     full text of one ~1,500-word window (or the whole document)
You can also use Python (re, string search) over `docs` freely.

"""
    + "Strategy hint:"
    + SYSTEM_BCP.split("Strategy hint:")[1]
)
