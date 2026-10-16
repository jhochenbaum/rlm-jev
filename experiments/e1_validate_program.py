"""Validate the deterministic root program with gold labels (no Jev calls)."""

import collections
import sys

sys.path.insert(0, ".")
from jevlab.oolong import LABELS, load_val, parse_window
from jevlab.oolong_program import answer, score

df = load_val()
max_len = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9  # optional: only contexts up to this length
df = df[df.context_len <= max_len]
cache = {}
tot = collections.defaultdict(list)
fails = collections.Counter()
bad = []
for r in df.itertuples():
    k = (r.dataset, r.context_len, r.context_window_id)
    if k not in cache:
        cache[k] = parse_window(r.context_window_text_with_labels)
    recs = cache[k]
    labels = [x.label for x in recs]
    try:
        p = answer(r.question, recs, labels, list(LABELS[r.dataset]))
        s = score(p, r.answer, r.answer_type)
    except NotImplementedError as e:
        s = 0
        fails[str(e)[:120]] += 1
        p = "NI"
    tot[r.task].append(s)
    if s < 1:
        bad.append((r.task, r.question[:250], p, r.answer))
for t, v in tot.items():
    print(f"{t:35s} n={len(v):4d} mean={sum(v) / len(v):.3f}")
print("overall", sum(sum(v) for v in tot.values()) / sum(len(v) for v in tot.values()))
print(fails.most_common(10))
for b in bad[:15]:
    print(b)
