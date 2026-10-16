"""jev — Jev (System One) sub-calls for coding agents (RLM toolkit over files too big to read).

  jev locate  FILE... -q QUERY [-q QUERY ...] [-k 10]      two-level zoom: windows -> lines
  jev filter  FILE... -q QUESTION [--thr 0.5]               semantic grep: per-line yes/no
  jev classify FILE --labels "a,b,c" [--desc a="..." ...]   per-line label + confidence (JSONL)

Design rules baked in (measured in this repo):
  * per-item text goes inside each question (inline); never point into a big state with `records[i]`
  * many queries share one window state (the state is billed once; each extra question adds only its own text)
  * every call is logged to ledger.jsonl and budget-capped (see jevlab/client.py)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from typesafe_sdk import Noul

from .client import Jev
from .label import label_texts, yesno_texts

WIN_LINES = 60
L1Q = "Does this passage contain content relevant to: {q}"
L2Q = "Is `line` relevant to: {q}"


def _lines(paths):
    out = []
    for p in paths:
        for i, t in enumerate(Path(p).read_text(errors="ignore").splitlines(), 1):
            if t.strip():
                out.append((p, i, t))
    return out


async def locate(paths, queries, k=10, top_windows=4, tag="cli_locate"):
    jev, sem = Jev(), asyncio.Semaphore(32)
    L = _lines(paths)
    wins = [list(range(i, min(i + WIN_LINES, len(L)))) for i in range(0, len(L), WIN_LINES)]
    W = np.zeros((len(queries), len(wins)))

    async def l1(wi):
        state = "\n".join(L[i][2] for i in wins[wi])
        qs = {f"q{j}": Noul(instructions=L1Q.format(q=q)) for j, q in enumerate(queries)}
        async with sem:
            r = await jev.aask(state, qs, tag=tag)
        for j in range(len(queries)):
            W[j, wi] = r.answers[f"q{j}"].noul

    await asyncio.gather(*(l1(wi) for wi in range(len(wins))))
    win_of = {i: wi for wi, w in enumerate(wins) for i in w}
    results = {}
    for j, q in enumerate(queries):
        cand = [i for wi in np.argsort(-W[j])[:top_windows] for i in wins[wi]]
        p = await yesno_texts(
            [L[i][2][:2000] for i in cand],
            L2Q.format(q=q),
            tag=tag,
            context="Judge each line independently.",
            field="line",
            jev=jev,
        )
        scored = sorted(((p[n] * W[j, win_of[i]], i) for n, i in enumerate(cand)), reverse=True)[:k]
        results[q] = [
            {"file": L[i][0], "line": L[i][1], "score": round(float(s), 3), "text": L[i][2][:300]} for s, i in scored
        ]
    await jev.aclose()
    return results


async def filt(paths, question, thr=0.5, tag="cli_filter"):
    L = _lines(paths)
    p = await yesno_texts(
        [t[:2000] for _, _, t in L], question, tag=tag, context="Judge each line independently.", field="line"
    )
    return [
        {"file": f, "line": n, "p": round(x, 3), "text": t[:300]}
        for (f, n, t), x in zip(L, p, strict=False)
        if x >= thr
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jev", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("locate")
    a.add_argument("files", nargs="+")
    a.add_argument("-q", action="append", required=True)
    a.add_argument("-k", type=int, default=10)
    f = sub.add_parser("filter")
    f.add_argument("files", nargs="+")
    f.add_argument("-q", required=True)
    f.add_argument("--thr", type=float, default=0.5)
    c = sub.add_parser("classify")
    c.add_argument("file")
    c.add_argument("--labels", required=True)
    c.add_argument("--desc", action="append", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "locate":
        res = asyncio.run(locate(args.files, args.q, args.k))
        for q, hits in res.items():
            print(f"## {q}")
            for h in hits:
                print(f"{h['file']}:{h['line']}\t{h['score']}\t{h['text']}")
    elif args.cmd == "filter":
        for h in asyncio.run(filt(args.files, args.q, args.thr)):
            print(f"{h['file']}:{h['line']}\t{h['p']}\t{h['text']}")
    else:
        labels = {l.strip(): None for l in args.labels.split(",")}
        for d in args.desc:
            k, v = d.split("=", 1)
            labels[k] = v
        L = _lines([args.file])
        probs = asyncio.run(label_texts([t for _, _, t in L], labels, batch=200, tag="cli_classify"))
        for (_fn, n, t), p in zip(L, probs, strict=False):
            top = max(p, key=p.get)
            print(json.dumps({"line": n, "label": top, "p": round(p[top], 3), "text": t[:200]}))


if __name__ == "__main__":
    sys.exit(main())
