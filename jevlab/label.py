"""Batched per-record labeling with Jev (the RLM sub-call map step).

label_texts(texts, options, batch=B) sends B records per request, each inside its own Choice question (mode="inline"),
and returns one {option: probability} dict per text. Results are cached on disk (data/label_cache.sqlite), keyed by
the full request configuration and the text, so re-runs and overlapping context windows cost nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from typesafe_sdk import Choice, Noul

from rlm_jev._client import TOKENS_PER_QUESTION, estimate_tokens, pack

from .client import Jev

# The harness packs requests close to Jev's ~64k-token limit so a batch of 250 short records stays one request (as in
# the published runs); the library keeps a wider safety margin.
HARNESS_MAX_TOKENS = 62_000
ROOT = Path(__file__).resolve().parent.parent
_DB_LOCK = threading.Lock()  # one connection shared by every thread; sqlite needs the access serialized
_DB: list[sqlite3.Connection] = []


def _db() -> sqlite3.Connection:
    if not _DB:
        (ROOT / "data").mkdir(exist_ok=True)
        con = sqlite3.connect(ROOT / "data/label_cache.sqlite", check_same_thread=False)
        con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        _DB.append(con)
    return _DB[0]


def _key(cfg: str, text: str) -> str:
    return hashlib.sha1(f"{cfg}\x00{text}".encode()).hexdigest()


def _cfg(*parts) -> str:
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]


async def label_texts(
    texts: list[str],
    options: dict[str, str],
    instruction: str = "Which category best describes `records[{i}]`?",
    batch: int = 50,
    order: str = "fixed",  # fixed | reversed
    # inline (use this): each record inside its own question | state: positional records[i] pointers, kept only to
    # reproduce E0's collapse | keyed / keyed_lines: records under visible IDs (E0b)
    mode: str = "inline",
    concurrency: int = 16,
    tag: str = "label",
    jev: Jev | None = None,
    use_cache: bool = True,
    context: str | None = None,
    inline_instruction: str = "Which category best describes `record`?",
    labels_in_state: bool = False,
    model: str = "jev-latest",
) -> list[dict[str, float]]:
    """Per-record label distributions from Jev, cached in data/label_cache.sqlite by text and request settings."""
    # inline mode puts the label descriptions in every question; labels_in_state=True writes them once in the shared
    # state and gives each question only the label names (E6c)
    shared = mode == "inline" and labels_in_state and any(options.values())
    cfg = _cfg(
        options, instruction, mode, context, inline_instruction, batch, order, model, *(["shared"] if shared else [])
    )
    out: list[dict | None] = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        row = None
        if use_cache:
            with _DB_LOCK:
                row = _db().execute("SELECT v FROM c WHERE k=?", (_key(cfg, t),)).fetchone()
        if row:
            out[i] = json.loads(row[0])
        else:
            todo.append(i)

    opts = dict(reversed(list(options.items()))) if order == "reversed" else options
    state_text = context or "Classify each record independently."
    if shared:
        state_text += "\nThe labels:\n" + "\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in options.items())
        opts = dict.fromkeys(opts)
    sem = asyncio.Semaphore(concurrency)

    async def run(chunk: list[int]):
        if mode == "state":
            state = {"records": [texts[i] for i in chunk]}
            qs = {f"r{j}": Choice(instructions=instruction.format(i=j), criteria=opts) for j in range(len(chunk))}
        elif mode in ("keyed", "keyed_lines"):
            # each record carries a visible ID in the state; the question names the ID (no positional counting)
            if mode == "keyed":
                state = {"records": {f"r{j}": texts[i] for j, i in enumerate(chunk)}}
            else:
                state = "\n".join(f"r{j}: {texts[i]}" for j, i in enumerate(chunk))
            qs = {
                f"r{j}": Choice(instructions=f"Which category best describes record r{j}?", criteria=opts)
                for j in range(len(chunk))
            }
        else:
            state = state_text
            qs = {
                f"r{j}": Choice(instructions={"record": texts[i], "question": inline_instruction}, criteria=opts)
                for j, i in enumerate(chunk)
            }
        async with sem:
            r = await jev.aask(state, qs, tag=tag)
        rows = []
        for j, i in enumerate(chunk):
            out[i] = dict(_answer(r, f"r{j}").probabilities)
            rows.append((_key(cfg, texts[i]), json.dumps(out[i])))
        if use_cache:
            with _DB_LOCK:
                _db().executemany("INSERT OR REPLACE INTO c VALUES (?,?)", rows)
                _db().commit()

    if mode == "inline":  # at most `batch` records per request, and under Jev's request-size limit
        per_q = estimate_tokens(inline_instruction, *opts, *(v or "" for v in opts.values())) + TOKENS_PER_QUESTION
        state_tok = estimate_tokens(state_text)
        chunks = [
            [todo[k] for k in idx]
            for idx in pack([texts[i] for i in todo], state_tok, per_q, batch, HARNESS_MAX_TOKENS)
        ]
    else:
        chunks = [todo[k : k + batch] for k in range(0, len(todo), batch)]
    own = bool(todo) and jev is None
    if todo:
        jev = jev or Jev(model)
    try:
        await asyncio.gather(*(run(c) for c in chunks))
    finally:
        if own:
            await jev.aclose()
    return out  # type: ignore[return-value]


async def yesno_texts(
    texts: list[str],
    question: str,
    tag: str = "yesno",
    batch: int = 200,
    concurrency: int = 24,
    context: str = "Answer each question about its item independently.",
    field: str = "item",
    jev: Jev | None = None,
) -> list[float]:
    """P(yes) for `question` about each text, each text inside its own Noul question (not cached)."""
    own = jev is None
    jev = jev or Jev()
    sem = asyncio.Semaphore(concurrency)
    out = [0.0] * len(texts)
    per_q = estimate_tokens(question) + TOKENS_PER_QUESTION

    async def run(idx: list[int]):
        qs = {f"q{j}": Noul(instructions={field: texts[i], "question": question}) for j, i in enumerate(idx)}
        async with sem:
            r = await jev.aask(context, qs, tag=tag)
        for j, i in enumerate(idx):
            out[i] = _answer(r, f"q{j}").noul

    try:
        await asyncio.gather(
            *(run(idx) for idx in pack(texts, estimate_tokens(context), per_q, batch, HARNESS_MAX_TOKENS))
        )
    finally:
        if own:
            await jev.aclose()
    return out


def _answer(r, key: str):
    if key not in r.answers:
        raise RuntimeError(f"Jev returned no answer for question {key}")
    return r.answers[key]
