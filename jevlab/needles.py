"""The E5 retrieval corpus: 19th-century novels split into sentences, with the hidden facts ("needles") planted at
seeded random positions. Shared by e5_locate.py (Jev) and e5c_modern_baselines.py (embedding + reranker baselines)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_corpus(target_words: int = 750_000) -> list[str]:
    """Sentences (4+ words) from data/books, in file order, until about `target_words` words."""
    sents = []
    for p in sorted((ROOT / "data/books").glob("*.txt")):
        t = p.read_text(errors="ignore")
        s = t.find("*** START")
        e = t.find("*** END")
        t = t[t.find("\n", s) + 1 : e] if s >= 0 and e > s else t
        t = re.sub(r"\s+", " ", t)
        for x in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"‘'])", t):
            if len(x.split()) >= 4:
                sents.append(x.strip())
        if sum(len(x.split()) for x in sents) >= target_words:
            break
    return sents


def load_needles() -> list[dict]:
    return json.load(open(ROOT / "experiments/assets/needles.json"))


def place_needles(sents: list[str], needles: list[dict], seed: int = 42) -> tuple[list[str], list[int]]:
    """Insert each needle at a seeded random position. Returns (sentences, index of each needle, in needles order)."""
    sents = list(sents)
    pos = sorted(random.Random(seed).sample(range(len(sents)), len(needles)))
    for p, n in sorted(zip(pos, needles, strict=False), key=lambda z: z[0], reverse=True):
        sents.insert(p, n["needle"])
    at = {s: i for i, s in enumerate(sents)}
    return sents, [at[n["needle"]] for n in needles]
