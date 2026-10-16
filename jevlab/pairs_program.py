"""Root program for OOLONG-Pairs: per-user label profiles -> exact pair enumeration."""

import collections
import itertools
import re

from .oolong import LABELS
from .oolong_program import pdate

LABS = list(LABELS["trec_coarse"])
NUM = {"one": 1, "two": 2}


def _lab(s):
    s = s.strip().removeprefix("an ").removeprefix("a ").strip()
    if s not in LABS:
        raise ValueError(f"unknown label in question: {s!r}")
    return s


def parse_conds(text):
    text = text.replace("and one with", "and at least one instance with")
    conds = []
    alt = "|".join(LABS)
    for m in re.finditer(rf"(at least|exactly) (one|two) instances? with ({alt})", text):
        conds.append((m[1], NUM[m[2]], _lab(m[3])))
    return conds


def sat(prof, conds):
    return all((prof[l] >= n) if kind == "at least" else (prof[l] == n) for kind, n, l in conds)


def solve(q, recs, labels):
    prof = collections.defaultdict(collections.Counter)
    dates = collections.defaultdict(list)
    for r, l in zip(recs, labels, strict=False):
        prof[r.user][l] += 1
        dates[(r.user, l)].append(pdate(r.date))
    users = sorted(prof, key=int)
    m = re.search(
        r"where both users have at least one instance with (.+?) or (.+?)(?:, and all instances that are (.+?) for both users must be (after|before) (\w+ \d+, \d+))?\. Each",
        q,
    )
    if m:
        a, b = _lab(m[1]), _lab(m[2])
        ok = {u for u in users if prof[u][a] + prof[u][b] > 0}
        if m[3]:
            z, d = _lab(m[3]), pdate(m[5])
            cmp = (lambda x: x > d) if m[4] == "after" else (lambda x: x < d)
            ok = {u for u in ok if all(cmp(x) for x in dates[(u, z)])}
        ok = sorted(ok, key=int)
        return {(u, v) for u, v in itertools.combinations(ok, 2)}
    m = re.search(r"such that one user has (.+?), and the other user has (.+?)\. Each", q)
    if not m:
        raise ValueError(f"unrecognized OOLONG-Pairs question: {q[:120]!r}")
    c1, c2 = parse_conds(m[1]), parse_conds(m[2])
    s1 = {u for u in users if sat(prof[u], c1)}
    s2 = {u for u in users if sat(prof[u], c2)}
    out = set()
    for u in s1:
        for v in s2:
            if u != v:
                out.add(tuple(sorted((u, v), key=int)))
    return out


def f1(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)
