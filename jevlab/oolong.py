"""OOLONG helpers: parse context windows into records with gold labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LINE_RE = re.compile(
    r"^Date: (?P<date>[^|]+?) \|\| User: (?P<user>\d+) \|\| Instance: (?P<text>.*?)(?: \|\| Label: (?P<label>.+))?$"
)

# Option descriptions handed to Jev: short and generic, like what a root LM would write from the task header.
LABELS = {
    "trec_coarse": {
        "description and abstract concept": "asks for a definition, description, explanation, reason or manner (what is X, why, how)",
        "entity": "asks for a thing: an object, animal, product, color, food, event, term, sport, etc.",
        "human being": "asks for a person, group of people or organization",
        "numeric value": "asks for a number, date, count, amount, distance, duration, percentage, etc.",
        "location": "asks for a place: city, country, state, mountain, address, etc.",
        "abbreviation": "asks what an abbreviation stands for or for the abbreviation of something",
    },
    "spam": {
        "spam": "unsolicited promotional, prize, scam or advertising text message",
        "ham": "a normal personal text message (not spam)",
    },
}


@dataclass
class Rec:
    idx: int
    date: str
    user: str
    text: str
    label: str | None


def parse_window(text_with_labels: str) -> list[Rec]:
    recs = []
    for line in text_with_labels.split("\n"):
        if not line.startswith("Date:"):
            continue
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        recs.append(Rec(len(recs), m["date"].strip(), m["user"], m["text"].strip(), (m["label"] or "").strip() or None))
    return recs


def load_val() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/oolong_synth_val.parquet")


def truncate_keep_recent(text: str, max_chars: int) -> tuple[str, float]:
    """Fit a context window into max_chars by dropping its earliest data lines (lines are in date order), keeping the
    task header and the label-set footer. Mirrors the RLM blog's truncated baseline ("truncate by most recent tokens").
    Returns (text, share of data lines kept)."""
    if len(text) <= max_chars:
        return text, 1.0
    lines = text.split("\n")
    data = [i for i, l in enumerate(lines) if l.startswith("Date:")]
    if not data:
        raise ValueError("no 'Date:' data lines to truncate")
    head, body, foot = lines[: data[0]], lines[data[0] : data[-1] + 1], lines[data[-1] + 1 :]
    budget = max_chars - len("\n".join(head + foot)) - 2
    kept, size = [], 0
    for l in reversed(body):
        if size + len(l) + 1 > budget:
            break
        kept.append(l)
        size += len(l) + 1
    kept.reverse()
    n_data = sum(l.startswith("Date:") for l in body)
    return "\n".join(head + kept + foot), sum(l.startswith("Date:") for l in kept) / n_data


PAIRS_QIDS = ["1", "4", "11", "19"]  # the OOLONG-Pairs questions used in the live runs


PAIRS_SUFFIX = " Return FINAL with the pairs formatted as (id1, id2) separated by commas."


def e2_jobs(trec: pd.DataFrame, with_32k: bool = True, pairs_suffix: str = PAIRS_SUFFIX) -> list[tuple]:
    """The live RLM jobs as (kind, leaf, item, question, context), where item is the OOLONG row or, for
    OOLONG-Pairs, the question dict. with_32k adds the 32k questions, once per leaf (Jev and an LLM)."""
    q = e2_questions(trec)
    jobs = [
        (k, "jev", r, r.question, r.context_window_text) for k in ("synth131k", "synth1M") for r in q[k].itertuples()
    ]
    if with_32k:
        jobs += [
            ("abl32k", leaf, r, r.question, r.context_window_text)
            for r in q["abl32k"].itertuples()
            for leaf in ("jev", "llm")
        ]
    ctx32 = trec[trec.context_len == 32768].iloc[0].context_window_text
    pairs = {x["id"]: x for x in json.loads((ROOT / "data/oolong-pairs-32768.json").read_text())}
    jobs += [("pairs32k", "jev", pairs[i], pairs[i]["question"] + pairs_suffix, ctx32) for i in PAIRS_QIDS]
    return jobs


def e2_questions(trec: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The questions the live RLM runs use (E2, and the start of E9's and E8b's question lists): up to 3 per task at
    131k (12), 1 per task at 1M (4), and 3 at 32k."""
    r131 = trec[trec.context_len == 131072]
    r1m = trec[trec.context_len == 1048576]
    per_task = lambda df, n: df.loc[
        df.groupby("task", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=0)).index
    ]
    return {
        "synth131k": per_task(r131, 3).head(12),
        "synth1M": per_task(r1m, 1).head(4),
        "abl32k": trec[trec.context_len == 32768].sample(3, random_state=1),
    }
