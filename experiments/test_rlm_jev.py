"""Smoke test of the public rlm_jev library (spend recorded to ledger.jsonl)."""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import jevlab  # noqa: F401  (loads TYPESAFE_API_KEY from .env.local)
from jevlab.oolong import LABELS, load_val, parse_window
from rlm_jev import apply_weights, classify, fit_weights, locate, yesno
from rlm_jev._client import Spend

spend = Spend(max_usd=0.15)
df = load_val()
row = df[(df.dataset == "trec_coarse") & (df.context_len == 131072)].iloc[0]
random.seed(0)
recs = random.sample([r for r in parse_window(row.context_window_text_with_labels) if r.label], 400)
t0 = time.time()
P = classify([r.text for r in recs], LABELS["trec_coarse"], ensemble=True, spend=spend)
acc = sum(max(p, key=p.get) == r.label for p, r in zip(P, recs, strict=False)) / len(recs)
print(f"classify: acc={acc:.3f} on 400 (expect ~0.95) in {time.time() - t0:.1f}s")
w = fit_weights(P[:200], [r.label for r in recs[:200]])
acc_w = sum(a == r.label for a, r in zip(apply_weights(P[200:], w), recs[200:], strict=False)) / 200
print(
    f"fit_weights on 200 -> held-out 200 acc {acc_w:.3f} (unweighted {sum(max(p, key=p.get) == r.label for p, r in zip(P[200:], recs[200:], strict=False)) / 200:.3f})"
)
y = yesno(["Where is the Orange Bowl ?", "What is a tonne ?"], "Is this question asking about a place?", spend=spend)
print("yesno:", [round(v, 2) for v in y])
book = Path("data/books/1661.txt").read_text(errors="ignore")[:200000]
hits = locate(book, ["Who is the woman Holmes always refers to as 'the woman'?"], k=3, spend=spend)
for _q, hs in hits.items():
    for h in hs:
        print("locate:", h["score"], h["text"][:110])
with open("ledger.jsonl", "a") as f:
    f.write(
        json.dumps(
            {
                "t": time.time(),
                "tag": "test_rlm_jev",
                "input_tokens": spend.tokens,
                "output_tokens": 0,
                "questions": 0,
                "latency_s": 0,
                "usd": spend.usd,
            }
        )
        + "\n"
    )
print(f"spent ${spend.usd:.4f}")
