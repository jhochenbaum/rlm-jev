"""locate(): two-level semantic zoom over text far larger than any context window.

Level 1: split into windows of about 8,000 characters (~1,400 words); one request per window with the window as the
state and one yes/no question per query, so every query shares a single read of the corpus.
Level 2: in the top windows for each query, score each line or sentence with inline yes/no questions.
"""

from __future__ import annotations

import re

from typesafe_sdk import Noul

from ._client import (
    MAX_REQUEST_TOKENS,
    TOKENS_PER_QUESTION,
    TOKENS_PER_REQUEST,
    Spend,
    ask_many,
    estimate_tokens,
    pack,
    run,
)

L1 = "Does this passage contain a sentence that answers the question: {q}"
L2 = "Does `{unit}` directly answer this question: {q}"
L2_STATE = "Answer each question about its {unit} independently."


def _windows(docs: list[str], window_chars: int):
    out = []
    for d, text in enumerate(docs):
        i = 0
        while i < len(text):
            j = min(len(text), i + window_chars)
            if j < len(text):
                cut = max(text.rfind("\n", i + window_chars // 2, j), text.rfind(". ", i + window_chars // 2, j))
                j = cut + 1 if cut > 0 else j
            out.append((d, i, text[i:j]))
            i = j
    return out


# A sentence ends at . ! or ? (plus closing quotes) before whitespace and a capital, except after a title or initial.
_SENTENCE = re.compile(
    r"\S.*?(?:(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\b[A-Z])[.!?][\"'”’)]*"
    r"(?=\s+[A-Z\"'“‘(])|$)",
    re.S,
)


def _units(text: str) -> tuple[str, list[tuple[int, str]]]:
    """("line" or "sentence", units with their offsets in `text`): lines, or sentences for prose without breaks."""
    unit = "line"
    lines = [m for m in re.finditer(r"[^\n]+", text) if m.group().strip()]
    if len(lines) < 3:
        unit = "sentence"
        lines = [m for m in _SENTENCE.finditer(text) if m.group().strip()]
    out = []
    for m in lines:
        g = m.group()
        out.append((m.start() + len(g) - len(g.lstrip()), g.strip()))
    return unit, out


def _answer(r, key):
    if key not in r.answers:
        raise RuntimeError(f"Jev returned no answer for question {key}")
    return r.answers[key].noul


def _query_groups(queries: list[str], wins) -> list[list[int]]:
    """Split the queries into groups small enough that one window plus one group fits in a single Jev request."""
    q_tok = [estimate_tokens(L1.format(q=q)) + TOKENS_PER_QUESTION for q in queries]
    room = MAX_REQUEST_TOKENS - TOKENS_PER_REQUEST - max(estimate_tokens(t) for _, _, t in wins)
    groups, cur, size = [], [], 0
    for j, t in enumerate(q_tok):
        if t > room:
            raise ValueError(f"a window plus query {j} exceeds one Jev request; lower window_chars or shorten it")
        if cur and size + t > room:
            groups.append(cur)
            cur, size = [], 0
        cur.append(j)
        size += t
    groups.append(cur)
    return groups


def _window_scores(wins, queries: list[str], concurrency: int, spend: Spend | None) -> list[list[float]]:
    """Level 1: P(window helps answer query) for every window and query. Each window is sent once as the state, with
    a group of queries attached as yes/no questions."""
    groups = _query_groups(queries, wins)
    q_tok = [estimate_tokens(L1.format(q=q)) + TOKENS_PER_QUESTION for q in queries]
    reqs, owners = [], []
    for i, (_, _, text) in enumerate(wins):
        for g in groups:
            qs = {f"q{n}": Noul(instructions=L1.format(q=queries[j])) for n, j in enumerate(g)}
            reqs.append((text, qs, estimate_tokens(text) + sum(q_tok[j] for j in g) + TOKENS_PER_REQUEST))
            owners.append((i, g))
    W = [[0.0] * len(queries) for _ in wins]
    for r, (i, g) in zip(run(ask_many(reqs, concurrency, spend=spend)), owners, strict=True):
        for n, j in enumerate(g):
            W[i][j] = _answer(r, f"q{n}")
    return W


def _line_candidates(wins, orders, top_windows: int, queries: list[str]):
    """For each query, the lines (or sentences) of its best windows: (window, doc, offset, text, unit)."""
    split, cands = {}, {}
    for j in range(len(queries)):
        cands[j] = []
        for i in orders[j][:top_windows]:
            if i not in split:
                split[i] = _units(wins[i][2])
            unit, units = split[i]
            cands[j] += [(i, wins[i][0], wins[i][1] + off, u, unit) for off, u in units]
    return cands


def _line_scores(cands, queries: list[str], concurrency: int, spend: Spend | None) -> dict[int, list[float]]:
    """Level 2: P(line answers query) for every candidate line, every query's lines in one batch of requests."""
    reqs, owners = [], []
    for j, q in enumerate(queries):
        for unit in {c[4] for c in cands[j]}:
            sel = [n for n, c in enumerate(cands[j]) if c[4] == unit]
            items = [cands[j][n][3][:2000] for n in sel]
            question, state = L2.format(unit=unit, q=q), L2_STATE.format(unit=unit)
            per_q = estimate_tokens(question) + TOKENS_PER_QUESTION
            for idx in pack(items, estimate_tokens(state), per_q, 200):
                qs = {f"q{n}": Noul(instructions={unit: items[i], "question": question}) for n, i in enumerate(idx)}
                est = estimate_tokens(state) + TOKENS_PER_REQUEST + sum(estimate_tokens(items[i]) + per_q for i in idx)
                reqs.append((state, qs, est))
                owners.append((j, [sel[i] for i in idx]))
    p = {j: [0.0] * len(cands[j]) for j in cands}
    for r, (j, idx) in zip(run(ask_many(reqs, concurrency, spend=spend)), owners, strict=True):
        for n, i in enumerate(idx):
            p[j][i] = _answer(r, f"q{n}")
    return p


def locate(
    docs,
    queries,
    k: int = 8,
    top_windows: int = 5,
    window_chars: int = 8000,
    zoom: bool = True,
    spend: Spend | None = None,
    concurrency: int = 48,
) -> dict[str, list[dict]]:
    """docs: a string or a list of strings. queries: a string or a list of strings.
    Returns {query: [{"doc", "offset", "score", "text"}]} ranked best first. With zoom=False, returns windows."""
    if window_chars < 200:
        raise ValueError("window_chars must be at least 200")
    docs = [docs] if isinstance(docs, str) else [str(d) for d in docs]
    queries = [queries] if isinstance(queries, str) else [str(q) for q in queries]
    wins = _windows(docs, window_chars)
    if not wins or not queries:
        return {q: [] for q in queries}

    W = _window_scores(wins, queries, concurrency, spend)
    orders = {j: sorted(range(len(wins)), key=lambda i: -W[i][j]) for j in range(len(queries))}
    if not zoom:
        return {
            q: [
                {"doc": wins[i][0], "offset": wins[i][1], "score": round(W[i][j], 3), "text": wins[i][2]}
                for i in orders[j][:k]
            ]
            for j, q in enumerate(queries)
        }

    cands = _line_candidates(wins, orders, top_windows, queries)
    p = _line_scores(cands, queries, concurrency, spend)
    out = {}
    for j, q in enumerate(queries):  # a line's score is P(line answers) x P(its window helps)
        scored = sorted(((p[j][n] * W[c[0]][j], c) for n, c in enumerate(cands[j])), key=lambda x: -x[0])[:k]
        out[q] = [{"doc": c[1], "offset": c[2], "score": round(s, 3), "text": c[3]} for s, c in scored]
    return out
