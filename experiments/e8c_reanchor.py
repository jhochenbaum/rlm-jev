"""E8c: DSPy ReAnchor — fit Choice weights (TREC) and a Noul threshold (spam) on 300 records, test on 1000 held out."""

import collections
import json
import logging
import random
import sys
from pathlib import Path

import dspy

sys.path.insert(0, ".")
from dspy.experimental import Choice, Noul, ReAnchor, TypeSafe

from jevlab.client import _check_budget
from jevlab.dspy_jev import _KEY, _log_history
from jevlab.oolong import LABELS, load_val, parse_window

logging.basicConfig(level=logging.WARNING)
_check_budget()
df = load_val()
out = {}
lm = TypeSafe("jev-latest", api_key=_KEY, cache=True, timeout=30)


def pool(ds):
    g = {}
    for r in df[df.dataset == ds].drop_duplicates(["context_len", "context_window_id"]).itertuples():
        for x in parse_window(r.context_window_text_with_labels):
            g[x.text] = x.label
    items = sorted(g.items())
    random.Random(5).shuffle(items)
    return items[:300], items[300:1300]


# TREC: Choice with weights
tr, te = pool("trec_coarse")
C = Choice[tuple(LABELS["trec_coarse"].items())]
sig = dspy.Signature(
    {
        "item": (str, dspy.InputField()),
        "label": (
            C,
            dspy.OutputField(desc="Which category best describes the expected answer to the question in `item`?"),
        ),
    },
    "Classify the question by the type of its expected answer.",
)
prog = dspy.Predict(sig)
prog.set_lm(lm)
ex = lambda items: [dspy.Example(item=t, gold=l).with_inputs("item") for t, l in items]
metric = lambda e, p, trace=None: float(p.label.value == e.gold)
start = len(lm.history)
opt = ReAnchor(metric, num_threads=24)
cal = opt.compile(prog, trainset=ex(tr), valset=ex(te))
_log_history(lm, start, "e8c_trec")
out["trec_choice_weights"] = opt.report
print("TREC", json.dumps(dict(opt.report.items()), default=str)[:1500], flush=True)


# count error on held-out before/after (aggregate quality, the OOLONG weak spot)
def counts(p, items):
    return collections.Counter(p(item=t).label.value for t, _ in items)


gold_c = collections.Counter(l for _, l in te)
for name, p in [("before", prog), ("after", cal)]:
    c = counts(p, te)
    out[f"trec_count_relerr_{name}"] = round(sum(abs(c[k] - gold_c[k]) / gold_c[k] for k in gold_c) / len(gold_c), 4)
print({k: v for k, v in out.items() if "count" in k}, flush=True)

# spam: Noul threshold
tr, te = pool("spam")
N = Noul[
    (True, "spam: unsolicited promotional, prize, scam or advertising text"), (False, "ham: a normal personal message")
]
sig2 = dspy.Signature(
    {"message": (str, dspy.InputField()), "is_spam": (N, dspy.OutputField(desc="Is `message` spam?"))},
    "Decide whether the text message is spam.",
)
prog2 = dspy.Predict(sig2)
prog2.set_lm(lm)
ex2 = lambda items: [dspy.Example(message=t, gold=(l == "spam")).with_inputs("message") for t, l in items]
metric2 = lambda e, p, trace=None: float(bool(p.is_spam.value) == e.gold)
start = len(lm.history)
opt2 = ReAnchor(metric2, num_threads=24)
opt2.compile(prog2, trainset=ex2(tr), valset=ex2(te))
_log_history(lm, start, "e8c_spam")
out["spam_noul_threshold"] = opt2.report
print("SPAM", json.dumps(opt2.report, default=str)[:1500], flush=True)
Path("results/e8c_reanchor.json").write_text(json.dumps(out, indent=1, default=str))
