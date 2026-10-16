"""E5b: the shipped rlm_jev.locate() on E5's corpus and 30 hidden facts, with its default settings.

E5 measures the two-level zoom as an experiment script; this checks the library function users get. The corpus is
E5's sentences joined into one document; a hit counts if its span overlaps the planted fact's span.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
import jevlab  # noqa: F401  (loads TYPESAFE_API_KEY from .env.local)
from jevlab.client import _check_budget, _record
from jevlab.needles import load_corpus, load_needles, place_needles
from rlm_jev import Spend, locate, stats

K = 20


def main():
    needles = load_needles()
    sents, at = place_needles(load_corpus(750_000), needles)
    starts, pos = [], 0
    for s in sents:
        starts.append(pos)
        pos += len(s) + 1
    text = " ".join(sents)
    spans = [(starts[i], starts[i] + len(sents[i])) for i in at]
    _check_budget()
    t0 = time.time()
    hits = locate(text, [n["query"] for n in needles], k=K, spend=Spend(max_usd=0.5))
    wall = time.time() - t0
    ranks = []
    for n, (a, b) in zip(needles, spans, strict=True):
        r = next(
            (i + 1 for i, h in enumerate(hits[n["query"]]) if h["offset"] < b and h["offset"] + len(h["text"]) > a),
            None,
        )
        ranks.append(r or 10**6)
    rk = np.array(ranks)
    kinds = np.array([n["kind"] for n in needles])
    out = {
        "settings": {"k": K, "window_chars": 8000, "top_windows": 5},
        "R@1": float((rk == 1).mean()),
        "R@5": float((rk <= 5).mean()),
        "R@20": float((rk <= 20).mean()),
        "R@5_by_kind": {k: float((rk[kinds == k] <= 5).mean()) for k in sorted(set(kinds))},
        "ranks": [int(r) for r in rk],
        "wall_s": round(wall, 1),
        "jev": stats(),
    }
    _record("e5b_locate", SimpleNamespace(input_tokens=out["jev"]["input_tokens"]), len(needles), wall)
    print(json.dumps({k: v for k, v in out.items() if k != "ranks"}, indent=1))
    Path("results/e5b_library_locate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
