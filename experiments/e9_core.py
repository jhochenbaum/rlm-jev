"""E9 core set (resumable: each batch skips questions already in rlm_ledger.jsonl).

C1 paired @32k (n=20): RLM with Jev sub-calls | RLM with Haiku sub-calls | Sonnet reads everything |
  Haiku reads everything
C2 scaling: Jev RLM @131k (n=20), @1M (n=20), @4M (n=3); Haiku-sub-call RLM @131k (n=20) and @1M; Sonnet and Haiku reading
everything @131k, and @1M truncated to fit their 1M-token windows (most recent ~80% of lines kept)
Usage: uv run python experiments/e9_core.py <batch> [max_jobs], then: uv run python experiments/e9_core.py score
  batches: c1_direct c1_haiku_direct c1_jev c1_llm c2_jev131 c2_direct131 c2_haiku_direct131 c2_llm131 c2_jev1m
           c2_llm1m c2_jev4m c2_direct1m_trunc c2_haiku_direct1m_trunc
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, ".")
from jevlab.oolong import LABELS, e2_questions, load_val, truncate_keep_recent
from jevlab.oolong_program import extract, score
from jevlab.rlm import ledger_rows, run_direct, run_rlm

df = load_val()
trec = df[df.dataset == "trec_coarse"]


def pick(e2: pd.DataFrame, L: int, n: int, seed: int) -> pd.DataFrame:
    """E2's questions first, then more at random from the same context length."""
    sub = trec[trec.context_len == L]
    return pd.concat([e2, sub.drop(e2.index).sample(n - len(e2), random_state=seed)])


E2 = e2_questions(trec)
Q32 = trec[trec.context_len == 32768].sample(20, random_state=1)
Q131 = pick(E2["synth131k"], 131072, 20, 1)
Q1M = pick(E2["synth1M"], 1048576, 20, 1)
Q4M = trec[trec.context_len == 4194304].sample(3, random_state=1)

# (tag, questions, arm). Tags reuse E2's tags where the arm/length match, so finished E2 runs are skipped.
BATCHES = {
    "c1_direct": ("e9_c1_direct", Q32, "direct"),
    "c1_jev": ("e2_abl32k_jev", Q32, "jev"),
    "c1_llm": ("e2_abl32k_llm", Q32, "llm"),
    "c2_jev131": ("e2_synth131k_jev", Q131, "jev"),
    "c2_direct131": ("e9_c2_direct131", Q131, "direct"),
    "c2_llm131": ("e9_c2_llm131", Q131, "llm"),
    "c2_jev1m": ("e2_synth1M_jev", Q1M, "jev"),
    "c2_llm1m": ("e9_c2_llm1m", Q1M, "llm"),
    "c2_jev4m": ("e9_c2_jev4m", Q4M, "jev"),
    # Haiku 5.5 reading everything, and both models truncated to fit their 1M windows (keep the most recent lines)
    "c1_haiku_direct": ("e9_c1_haiku_direct", Q32, "direct_haiku"),
    "c2_haiku_direct131": ("e9_c2_haiku_direct131", Q131, "direct_haiku"),
    "c2_direct1m_trunc": ("e9_c2_direct1m_trunc", Q1M, "direct_trunc"),
    "c2_haiku_direct1m_trunc": ("e9_c2_haiku_direct1m_trunc", Q1M, "direct_haiku_trunc"),
}


def done(tag):
    return [r for r in ledger_rows(tag) if r["tag"] == tag]


def matches(p, r):
    """A ledger entry belongs to question r if the text matches AND it ran on r's window (the same question text
    can appear in several windows)."""
    return p["query"] == r.question[:300] and p.get("ctx_chars") == len(r.context_window_text)


def run_batch(name, max_jobs):
    tag, Q, arm = BATCHES[name]
    prev = done(tag)
    todo = []
    for r in Q.itertuples():
        hit = next((p for p in prev if matches(p, r)), None)
        if hit:
            prev.remove(hit)
        else:
            todo.append(r)
    todo = todo[:max_jobs]
    print(f"{name}: {len(Q)} questions, {len(todo)} to run", flush=True)

    def go(r):
        try:
            if arm.startswith("direct"):
                out = run_direct(
                    r.context_window_text,
                    r.question,
                    tag=tag,
                    model="haiku" if "haiku" in arm else "sonnet",
                    truncate=truncate_keep_recent if "trunc" in arm else None,
                )
            else:
                out = run_rlm(r.context_window_text, r.question, root_model="sonnet", leaf=arm, tag=tag)
            print(
                json.dumps(
                    {
                        "batch": name,
                        "id": int(r.id),
                        "answer": (out.get("answer") or "")[:60],
                        "wall_s": out.get("wall_s"),
                    }
                ),
                flush=True,
            )
        except Exception as e:
            print("FAIL", name, int(r.id), repr(e)[:200], flush=True)

    with ThreadPoolExecutor(3 if arm == "llm" else 4) as ex:
        list(ex.map(go, todo))


def score_all():
    rows = []
    for name, (tag, Q, arm) in BATCHES.items():
        prev = done(tag)
        for r in Q.itertuples():
            hit = next((p for p in prev if matches(p, r)), None)
            if not hit:
                continue
            prev.remove(hit)
            s = score(extract(hit.get("answer"), r.answer_type, list(LABELS["trec_coarse"])), r.answer, r.answer_type)
            rows.append(
                {
                    "batch": name,
                    "arm": arm,
                    "L": r.context_len,
                    "id": r.id,
                    "task": r.task,
                    "score": s,
                    "root_usd": hit.get("root_usd_list", 0),
                    "leaf_usd": hit.get("llm_usd_list", 0),
                    "jev_items": hit.get("jev_items", 0),
                    "llm_calls": hit.get("llm_calls", 0),
                    "turns": hit.get("turns"),
                    "wall_s": hit.get("wall_s"),
                }
            )
    R = pd.DataFrame(rows)
    R.to_csv("results/e9_core.csv", index=False)
    print(
        R.groupby(["batch"])
        .agg(
            n=("score", "size"),
            score=("score", "mean"),
            root_usd=("root_usd", "mean"),
            leaf_usd=("leaf_usd", "mean"),
            jev_items=("jev_items", "mean"),
            wall_s=("wall_s", "mean"),
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] != "score" and sys.argv[1] not in BATCHES):
        sys.exit(__doc__)
    if sys.argv[1] == "score":
        score_all()
    else:
        run_batch(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 999)
