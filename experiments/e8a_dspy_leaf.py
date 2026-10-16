"""E8a: DSPy's native Jev integration (one request per item) vs inline batching (200 items per request), on the same
400 TREC records as E0. Both arms call Jev fresh (no label cache), so tokens, cost and time are comparable."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from jevlab.client import LEDGER, PRICE_PER_TOKEN
from jevlab.dspy_jev import inline_classify, native_classify
from jevlab.oolong import LABELS, load_val, parse_window


def tokens_logged(tag: str) -> int:
    rows = (json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()) if LEDGER.exists() else ()
    return sum(r["input_tokens"] for r in rows if r["tag"] == tag)


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    total = 0.0
    for lo in np.arange(0, 1, 0.1):
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


df = load_val()
row = df[(df.dataset == "trec_coarse") & (df.context_len == 131072)].iloc[0]
random.seed(0)
recs = random.sample([r for r in parse_window(row.context_window_text_with_labels) if r.label], 400)
texts, gold = [r.text for r in recs], [r.label for r in recs]
arms = {
    "dspy_native": lambda tag: native_classify(texts, LABELS["trec_coarse"], tag=tag),
    "inline_b200": lambda tag: inline_classify(texts, LABELS["trec_coarse"], tag=tag, use_cache=False),
}
out = {}
for name, fn in arms.items():
    tag = f"e8a_{name}"
    before = tokens_logged(tag)
    t0 = time.time()
    P = fn(tag)
    dt = time.time() - t0
    tok = tokens_logged(tag) - before
    conf = np.array([max(p.values()) for p in P])
    correct = np.array([max(p, key=p.get) == g for p, g in zip(P, gold, strict=True)])
    out[name] = {
        "acc": round(float(correct.mean()), 4),
        "ece": round(ece(conf, correct), 4),
        "tokens": tok,
        "tok_per_item": round(tok / len(texts), 1),
        "usd": round(tok * PRICE_PER_TOKEN, 5),
        "wall_s": round(dt, 1),
    }
    print(name, out[name], flush=True)
Path("results/e8a_dspy_leaf.json").write_text(json.dumps(out, indent=1))
