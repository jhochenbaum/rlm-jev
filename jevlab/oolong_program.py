"""Deterministic 'root program' for OOLONG-synth questions.

answer(q, recs, labels, label_set) -> answer string. `labels[i]` is the label for records[i] (gold or Jev).
experiments/e1_validate_program.py checks that it scores 1.0 on gold labels. extract() and score() turn a
free-form answer into the benchmark's score.
"""

from __future__ import annotations

import collections
import contextlib
import re
from datetime import datetime

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def pdate(s: str) -> datetime:
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(s)


def subset(q: str, recs):
    """Indices of the records that the question's filter clause (user IDs, dates) selects."""
    idx = list(range(len(recs)))
    m = re.search(r"associated with user IDs ([\d, and]+)\.", q)
    if m:
        users = set(re.findall(r"\d+", m[1]))
        idx = [i for i in idx if recs[i].user in users]
    m = re.search(r"occur between (\w+ \d+, \d+) and (\w+ \d+, \d+), inclusive", q)
    if m:
        a, b = pdate(m[1]), pdate(m[2])
        idx = [i for i in idx if a <= pdate(recs[i].date) <= b]
    m = re.search(r"occur in (\w+) of any year", q)
    if m:
        mi = MONTHS.index(m[1]) + 1
        idx = [i for i in idx if pdate(recs[i].date).month == mi]
    return idx


def _cmp(a: float, b: float) -> str:
    return "more common than" if a > b else "less common than" if a < b else "same frequency as"


def answer(q: str, recs, labels, label_set: list[str]) -> str:
    idx = subset(q, recs)
    cnt = collections.Counter(labels[i] for i in idx)
    quoted = re.findall(r"label '([^']+)'", q)

    if "how many data points should be classified as label" in q:
        return str(cnt[quoted[0]])
    m = re.search(
        r"was label '([^']+)' more common, less common, or the same frequency before (\S+), as compared to after (\S+)\?",
        q,
    )
    if m:
        d = pdate(m[2].rstrip("?,"))
        b = [i for i in idx if pdate(recs[i].date) < d]
        a = [i for i in idx if pdate(recs[i].date) > d]
        before = sum(labels[i] == m[1] for i in b) / max(len(b), 1)  # OOLONG compares proportions
        after = sum(labels[i] == m[1] for i in a) / max(len(a), 1)
        return _cmp(before, after)
    if "more common, less common, or the same frequency as label" in q:
        return _cmp(cnt[quoted[0]], cnt[quoted[1]])
    if re.search(r"which of the labels is the (most|least|second most) common", q):
        allowed = re.search(r"one of the labels: (.*?)\.?$", q.strip())
        opts = [s.strip() for s in allowed[1].split(",")] if allowed else label_set
        ranked = sorted(opts, key=lambda o: cnt[o])
        if "least" in q:
            return ranked[0]
        if "second most" in q:
            return sorted(opts, key=lambda o: -cnt[o])[1]
        return ranked[-1]
    m = re.search(r"which user has (?:the most|more) instances with the label ([^:?]+?)(?::|\?)", q)
    if m:
        lab = m[1].strip()
        users = re.findall(r"User (\d+)", q)
        uc = collections.Counter(recs[i].user for i in idx if labels[i] == lab)
        if users:
            return max(users, key=lambda u: uc[u])
        return uc.most_common(1)[0][0] if uc else ""
    if "which user is represented" in q:
        uc = collections.Counter(recs[i].user for i in idx).most_common()
        return uc[1][0] if "second most" in q else uc[0][0]
    if re.search(r"which date is represented (the second )?most often", q):
        dc = collections.Counter(pdate(recs[i].date).strftime("%m/%d/%Y") for i in idx).most_common()
        return dc[1][0] if "second most" in q else dc[0][0]
    m = re.search(r"how many dates are represented exactly (\d+) times", q)
    if m:
        dc = collections.Counter(pdate(recs[i].date) for i in idx)
        return str(sum(1 for v in dc.values() if v == int(m[1])))
    m = re.search(r"For how many months does the label '([^']+)' occur more frequently than the label '([^']+)'", q)
    if m:
        by = collections.defaultdict(collections.Counter)
        for i in idx:
            d = pdate(recs[i].date)
            by[(d.year, d.month)][labels[i]] += 1
        return str(sum(1 for c in by.values() if c[m[1]] > c[m[2]]))
    m = re.search(r"For how many months is the label '([^']+)' the single most frequently occuring label", q)
    if m:
        by = collections.defaultdict(collections.Counter)
        for i in idx:
            d = pdate(recs[i].date)
            by[(d.year, d.month)][labels[i]] += 1
        n = 0
        for c in by.values():
            top = c.most_common()
            if top and top[0][0] == m[1] and (len(top) == 1 or top[0][1] > top[1][1]):
                n += 1
        return str(n)
    m = re.search(r"In which month did the label '([^']+?)'? first occur more often than the label '([^']+?)'?\?", q)
    if m:
        by = collections.defaultdict(collections.Counter)
        for i in idx:
            d = pdate(recs[i].date)
            by[(d.year, d.month)][labels[i]] += 1
        for y, mo in sorted(by):
            if by[(y, mo)][m[1]] > by[(y, mo)][m[2]]:
                return f"{MONTHS[mo - 1]} {y}"
        return ""
    raise NotImplementedError(q[:200])


def score(pred: str, gold_raw: str, answer_type: str) -> float:
    import datetime as _dt

    gold = eval(gold_raw, {"__builtins__": {}}, {"datetime": _dt})  # trusted dataset field
    gold = gold[0] if isinstance(gold, list) else gold
    if isinstance(gold, _dt.date):
        gold = gold.strftime("%m/%d/%Y")
    if answer_type == "ANSWER_TYPE.COMPARISON":
        norm = lambda x: str(x).strip().lower().replace(" than", "").replace(" as", "")
        return float(norm(pred) == norm(gold))
    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(float(pred) - float(gold))
        except ValueError:
            return 0.0
    return float(str(pred).strip().lower() == str(gold).strip().lower())


_MONTHS = [m.lower() for m in MONTHS]
_MON_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def _month(name: str) -> int:
    return next(i for i, m in enumerate(_MONTHS, 1) if m.startswith(name.lower()[:3]))


def _dates(s: str) -> list[str]:
    """Every date in s, in order of appearance, as MM/DD/YYYY."""
    found = []
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s):
        found.append((m.start(), int(m[1]), int(m[2]), int(m[3])))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        found.append((m.start(), int(m[2]), int(m[3]), int(m[1])))
    for m in re.finditer(rf"\b{_MON_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", s, re.I):
        found.append((m.start(), _month(m[1]), int(m[2]), int(m[3])))
    out = []
    for _, mo, d, y in sorted(found):
        with contextlib.suppress(ValueError):
            out.append(datetime(y, mo, d).strftime("%m/%d/%Y"))
    return out


def extract(ans: str | None, answer_type: str, label_set: list[str]) -> str:
    """Pull the scorable value out of a free-form root answer (dates and month names in any common format)."""
    if not ans:
        return ""
    s = ans.strip()
    tail = s.split(":")[-1].strip() if ":" in s else s
    if answer_type == "ANSWER_TYPE.COMPARISON":
        m = re.findall(r"(more common|less common|same frequency)", s.lower())
        return m[-1] if m else ""
    if answer_type == "ANSWER_TYPE.LABEL":

        def hits(text):
            t = text.lower()
            return [
                (m.start(), l) for l in label_set for m in re.finditer(rf"(?<![a-z]){re.escape(l.lower())}(?![a-z])", t)
            ]

        h = hits(tail) or hits(s)
        return max(h)[1] if h else ""
    if answer_type in ("ANSWER_TYPE.NUMERIC", "ANSWER_TYPE.USER"):
        m = re.findall(r"\d+(?:\.\d+)?", tail) or re.findall(r"\d+(?:\.\d+)?", s)
        return m[-1] if m else ""
    if answer_type == "ANSWER_TYPE.DATE":
        d = _dates(s)
        return d[-1] if d else ""
    if answer_type == "ANSWER_TYPE.MONTH_YEAR":
        m = list(re.finditer(rf"\b{_MON_RE}\.?\s+(\d{{4}})\b", s, re.I))
        return f"{_MONTHS[_month(m[-1][1]) - 1].capitalize()} {m[-1][2]}" if m else ""
    return tail
