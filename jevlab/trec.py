"""TREC question-classification helpers for the weight-calibration experiments (E8f, E8g)."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from rlm_jev.calibrate import fit_weights

from .label import label_texts
from .oolong import LABELS

ROOT = Path(__file__).resolve().parent.parent

COARSE = {
    "ABBR": "abbreviation",
    "DESC": "description and abstract concept",
    "ENTY": "entity",
    "HUM": "human being",
    "LOC": "location",
    "NUM": "numeric value",
}
L = list(LABELS["trec_coarse"])


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


async def probs(
    texts: list[str], tag: str = "trec_label", labels_in_state: bool = False
) -> dict[str, dict[str, float]]:
    """Jev label distributions, averaged over both option orders."""
    kw = {"batch": 200, "tag": tag, "labels_in_state": labels_in_state}
    pf = await label_texts(texts, LABELS["trec_coarse"], **kw)
    pr = await label_texts(texts, LABELS["trec_coarse"], order="reversed", **kw)
    return {t: {k: (a[k] + b[k]) / 2 for k in a} for t, a, b in zip(texts, pf, pr, strict=False)}


def fit(P: dict, gold: dict, ts: list[str]) -> dict[str, float]:
    """Per-label weights fitted on the records `ts` (rlm_jev.fit_weights: coordinate ascent on accuracy)."""
    return fit_weights([P[t] for t in ts], [gold[t] for t in ts])


def fit_split(texts, n: int = 300, seed: int = 5) -> list[str]:
    """The records a weight fit uses: the first `n` after a seeded shuffle of the sorted texts. E8d and E8g use it;
    E8c's pool() reproduces the same split, so all three fit on the same records."""
    items = sorted(texts)
    random.Random(seed).shuffle(items)
    return items[:n]


def leakfree_oolong(df, wins: dict, fitset: set, labels: dict[str, str]) -> pd.DataFrame:
    """Score OOLONG questions with `fitset` records removed from every window and the gold answers recomputed by the
    validated program on what remains, so weights fitted on those records are never scored on them."""
    from .oolong_program import answer, score

    rows = []
    for r in df.itertuples():
        recs = [x for x in wins[(r.context_len, r.context_window_id)] if x.text not in fitset]
        gold_ans = answer(r.question, recs, [x.label for x in recs], L)
        pred_ans = answer(r.question, recs, [labels[x.text] for x in recs], L)
        gold_raw = repr([int(gold_ans)]) if r.answer_type == "ANSWER_TYPE.NUMERIC" else repr([gold_ans])
        answer_type = r.answer_type if r.answer_type != "ANSWER_TYPE.DATE" else "X"  # recomputed dates compare as text
        rows.append({"task": r.task, "L": r.context_len, "s": score(pred_ans, gold_raw, answer_type)})
    return pd.DataFrame(rows)


def external_records(val_texts) -> dict[str, str]:
    """Original TREC questions (train + test) that appear in no OOLONG validation window, with coarse labels."""
    val_norm = {norm(t) for t in val_texts}
    ext = {}
    for f in ("train_5500.label", "TREC_10.label"):
        for line in (ROOT / "data/trec" / f).read_text(encoding="latin-1").splitlines():
            lab, q = line.split(" ", 1)
            if norm(q) not in val_norm:
                ext[q.strip()] = COARSE[lab.split(":")[0]]
    return ext


def external_sample(ext: dict, n: int = 1000, seed: int = 11) -> list[str]:
    """The out-of-slice records E8f fits on (and E8h labels with Haiku)."""
    return random.Random(seed).sample(sorted(ext), n)
