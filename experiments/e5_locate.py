"""E5: locate() — semantic needle retrieval over ~1M tokens of novels.

Methods (sentence-level ranking of the planted needle):
  bm25        : BM25 over sentences
  emb         : MiniLM cosine over sentences
  jev_zoom    : L1 = one request per window (state = window text), one Noul per query (all queries share
                the state, so the corpus is read once for every query); L2 = inline Nouls on sentences of
                the top-k windows per query
  emb+jev     : embeddings top-200 sentences -> Jev inline-Noul rerank

Usage: uv run python experiments/e5_locate.py [corpus_words=750000] [window_words=1500]
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from typesafe_sdk import Noul

from jevlab.client import Jev, summary
from jevlab.needles import load_corpus, load_needles, place_needles

TARGET_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 750_000
WIN_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
TOPK_WIN = 5
L1Q = "Does this passage contain a sentence that answers the question: {q}"
L2Q = "Does `sentence` directly answer this question: {q}"
L2_STATE = "Answer each question about its sentence independently."


def rank_of(scores, target: int) -> int:
    """1-based rank of `target` among `scores` (ties count in the target's favor)."""
    return int((scores > scores[target]).sum()) + 1


def bm25_ranks(sents, queries, targets):
    from rank_bm25 import BM25Okapi

    tok = lambda s: re.findall(r"[a-z]+", s.lower())
    bm = BM25Okapi([tok(s) for s in sents])
    return [rank_of(bm.get_scores(tok(q)), t) for q, t in zip(queries, targets, strict=True)]


def embedding_scores(sents, queries):
    """Cosine similarity of every query to every sentence (MiniLM)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = model.encode(sents, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    Q = model.encode(queries, normalize_embeddings=True)
    return Q @ E.T


def windows(sents) -> list[list[int]]:
    """Consecutive sentences grouped into windows of about WIN_WORDS words."""
    wins, cur, cw = [], [], 0
    for i, s in enumerate(sents):
        cur.append(i)
        cw += len(s.split())
        if cw >= WIN_WORDS:
            wins.append(cur)
            cur, cw = [], 0
    if cur:
        wins.append(cur)
    return wins


async def noul_items(jev, sem, items, question, tag):
    """P(yes) for `question` about each item, 200 items per request."""
    out = [0.0] * len(items)

    async def chunk(lo):
        qs = {
            f"q{j}": Noul(instructions={"sentence": items[lo + j], "question": question})
            for j in range(min(200, len(items) - lo))
        }
        async with sem:
            r = await jev.aask(L2_STATE, qs, tag=tag)
        for j in range(len(qs)):
            out[lo + j] = r.answers[f"q{j}"].noul

    await asyncio.gather(*(chunk(lo) for lo in range(0, len(items), 200)))
    return out


async def window_scores(jev, sem, sents, wins, queries):
    """Level 1: W[query, window], each window sent once with every query attached."""
    W = np.zeros((len(queries), len(wins)))

    async def one(wi):
        state = " ".join(sents[i] for i in wins[wi])
        qs = {f"q{qi}": Noul(instructions=L1Q.format(q=q)) for qi, q in enumerate(queries)}
        async with sem:
            r = await jev.aask(state, qs, tag="e5_l1")
        for qi in range(len(queries)):
            W[qi, wi] = r.answers[f"q{qi}"].noul

    await asyncio.gather(*(one(wi) for wi in range(len(wins))))
    return W


async def zoom_ranks(jev, sem, sents, wins, W, queries, targets):
    """Level 2: score the sentences of each query's top windows; a sentence's score is P(sentence) x P(window)."""
    win_of = {i: wi for wi, w in enumerate(wins) for i in w}

    async def one(qi):
        top = np.argsort(-W[qi])[:TOPK_WIN]
        cand = [i for wi in top for i in wins[wi]]
        p = await noul_items(jev, sem, [sents[i] for i in cand], L2Q.format(q=queries[qi]), "e5_l2")
        score = {i: p[k] * W[qi, win_of[i]] for k, i in enumerate(cand)}
        tgt = targets[qi]
        return int(sum(v > score[tgt] for v in score.values())) + 1 if tgt in score else 10**6

    window_rank = [rank_of(W[qi], win_of[t]) for qi, t in enumerate(targets)]
    return list(await asyncio.gather(*(one(qi) for qi in range(len(queries))))), window_rank


async def rerank_ranks(jev, sem, sents, S, queries, targets):
    """Embeddings' top 200 sentences per query, re-scored by Jev."""

    async def one(qi):
        top = np.argsort(-S[qi])[:200]
        p = await noul_items(jev, sem, [sents[i] for i in top], L2Q.format(q=queries[qi]), "e5_rerank")
        if targets[qi] not in set(top.tolist()):
            return 10**6
        pt = p[top.tolist().index(targets[qi])]
        return int(sum(x > pt for x in p)) + 1

    return list(await asyncio.gather(*(one(qi) for qi in range(len(queries)))))


def recall_table(ranks, kinds) -> pd.DataFrame:
    rows = {}
    for m, rk in ranks.items():
        rk = np.array(rk)
        rows[m] = {"R@1": (rk == 1).mean(), "R@5": (rk <= 5).mean(), "R@20": (rk <= 20).mean(), "MRR": (1 / rk).mean()}
        for kd in sorted(set(kinds)):
            sel = np.array([k == kd for k in kinds])
            rows[m][f"R@5_{kd}"] = (rk[sel] <= 5).mean()
    return pd.DataFrame(rows).T.round(3)


async def main():
    needles = load_needles()
    sents, targets = place_needles(load_corpus(TARGET_WORDS), needles)
    queries = [n["query"] for n in needles]
    nwords = sum(len(s.split()) for s in sents)
    print(f"corpus: {len(sents)} sentences, {nwords:,} words, {len(needles)} needles", flush=True)
    ranks, timing = {}, {}

    t0 = time.time()
    ranks["bm25"] = bm25_ranks(sents, queries, targets)
    timing["bm25"] = time.time() - t0

    t0 = time.time()
    S = embedding_scores(sents, queries)
    ranks["emb"] = [rank_of(S[qi], t) for qi, t in enumerate(targets)]
    timing["emb"] = time.time() - t0

    jev, sem = Jev(), asyncio.Semaphore(32)
    t0 = time.time()
    wins = windows(sents)
    W = await window_scores(jev, sem, sents, wins, queries)
    t_l1 = time.time() - t0
    ranks["jev_zoom"], win_rank = await zoom_ranks(jev, sem, sents, wins, W, queries, targets)
    timing["jev_zoom"] = time.time() - t0

    t0 = time.time()
    ranks["emb+jev"] = await rerank_ranks(jev, sem, sents, S, queries, targets)
    timing["emb+jev"] = time.time() - t0
    await jev.aclose()

    res = {
        "corpus_sentences": len(sents),
        "words": nwords,
        "windows": len(wins),
        "win_words": WIN_WORDS,
        "l1_seconds": round(t_l1, 1),
        "window_recall@1": np.mean([r == 1 for r in win_rank]),
        "window_recall@5": np.mean([r <= 5 for r in win_rank]),
        "timing_s": {k: round(v, 1) for k, v in timing.items()},
    }
    print(json.dumps(res, indent=1))
    T = recall_table(ranks, [n["kind"] for n in needles])
    print(T.to_string())
    out = {
        "meta": res,
        "table": T.to_dict(),
        "ranks": {k: [int(x) for x in v] for k, v in ranks.items()},
        "win_rank": win_rank,
    }
    Path(f"results/e5_locate_{TARGET_WORDS}_{WIN_WORDS}.json").write_text(json.dumps(out, indent=1))
    print(summary())


if __name__ == "__main__":
    asyncio.run(main())
