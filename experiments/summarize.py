"""Build results/summary.json: every number in the write-up, recomputed from the raw files in results/.

No API calls. Run: uv run python experiments/summarize.py

Spend and token figures come from the local ledgers (ledger.jsonl, claude_ledger.jsonl, rlm_ledger.jsonl), which are
not committed. When a ledger, or the rows a figure needs, is missing (e.g. on a fresh clone), that figure is carried
over from the existing summary.json instead of being dropped or zeroed.
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from jevlab.claude_cli import TOO_LONG
from rlm_jev._client import PRICE_PER_TOKEN

R = Path("results")
OLD = json.loads((R / "summary.json").read_text()) if (R / "summary.json").exists() else {}
S = {}


def read_json(name: str):
    return json.loads((R / name).read_text())


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def merge(key: str, new: dict) -> dict:
    """New values over the previous summary's, so ledger-only fields survive a run without ledgers."""
    return {**OLD.get(key, {}), **new}


JEV = read_jsonl("ledger.jsonl")
CLAUDE = read_jsonl("claude_ledger.jsonl")
RLM = read_jsonl("rlm_ledger.jsonl")


def last_run(rows: list[dict], gap_s: float = 180) -> list[dict]:
    """The most recent run in a tag's ledger rows: rows after the last pause longer than gap_s (reruns of an
    experiment log under the same tag, and only the latest one produced the committed results)."""
    rows = sorted(rows, key=lambda r: r["t"])
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["t"] - rows[i - 1]["t"] > gap_s:
            start = i
    return rows[start:]


def jev_rows(*tags: str) -> list[dict]:
    return [r for r in JEV if r["tag"] in tags]


# ---------------------------------------------------------------- spend
spend = {}
if JEV:
    spend["jev_total_usd"] = round(sum(r["usd"] for r in JEV), 3)
if CLAUDE or RLM:
    # Claude at list price: direct CLI calls (claude_ledger) plus RLM root and sub-call costs (rlm_ledger). dspy.RLM
    # runs through the CLI log to claude_ledger themselves, and API-path runs are counted separately.
    cli = sum(r["usd_list"] or 0 for r in CLAUDE) + sum(
        (r.get("root_usd_list") or 0) + (r.get("llm_usd_list") or 0)
        for r in RLM
        if not r["tag"].startswith("e8b") and r.get("path") != "anthropic_api"
    )
    api = sum(
        (r.get("root_usd_list") or 0) + (r.get("sub_usd_list") or 0) for r in RLM if r.get("path") == "anthropic_api"
    )
    spend.update(
        claude_list_usd_total=round(cli, 2),
        anthropic_api_usd_dspy_rlm=round(api, 2),
        claude_list_usd_including_api=round(cli + api, 2),
    )
spend["claude_models"] = {"root_direct_judge": "claude-sonnet-5-5", "llm_subcall_baseline": "claude-haiku-5-5"}
spend["claude_pricing"] = "list price; cache writes at the 5-minute rate (jevlab/models.py)"
S["spend"] = merge("spend", spend)

# ---------------------------------------------------------------- E0, E0b: batching layout
e0 = read_json("e0_pilot.json")
S["E0_inline_vs_state_trec_acc"] = {k.split("/")[1]: v["acc"] for k, v in e0.items() if k.startswith("trec_coarse")}
S["E0_inline_ece_at_250"] = e0["trec_coarse/in_B250"]["ece"]
S["E0b_keyed_state_trec_acc"] = read_json("e0b_keyed_state.json")

# ---------------------------------------------------------------- E1, E1b, E3: OOLONG with the fixed program
e1 = pd.read_csv(R / "e1_oolong_val.csv")
S["E1_record_acc"] = read_json("e1_record_acc.json")
t131 = e1[(e1.dataset == "trec_coarse") & (e1.context_len == 131072)]
S["E1_trec_131k"] = t131[["gold", "regex", "jev", "jev_ens"]].mean().round(3).to_dict()
e1b = pd.read_csv(R / "e1b_test_131k.csv")
S["E1b_test_131k"] = e1b[["gold", "majority", "jev", "jev_ens"]].mean().round(3).to_dict()
e3 = read_json("e3_pairs.json")
S["E3_pairs_f1"] = {L: {k: round(v["mean_f1"], 3) for k, v in d.items()} for L, d in e3.items()}

# ---------------------------------------------------------------- E5: locate() and retrieval baselines
e5 = read_json("e5_locate_750000_1500.json")
S["E5_locate"] = {"table": e5["table"], "meta": e5["meta"]}
S["E5c_modern"] = {}
for f in sorted(glob.glob(str(R / "e5c_*.json"))):
    d = json.loads(Path(f).read_text())
    rr = d["reranker"].split("/")[-1].replace("-seq-cls", "")
    keys = ("R@1", "R@5", "R@20", "MRR", "R@5_by_kind")
    S["E5c_modern"][d["embedding_model"] + ("" if rr == "Qwen3-Reranker-0.6B" else f" + {rr}")] = {
        "emb": {k: d["emb"][k] for k in keys},
        "emb+rerank": {k: d["emb+rerank"][k] for k in keys},
        "reranker": d["reranker"],
        "rerank_top_k": d["rerank_top_k"],
    }

# ---------------------------------------------------------------- E6: Jev vs Haiku on the same 500 records
e6 = read_json("e6_labels.json")
sw, gold6 = e6["swap"], e6["swap"]["gold"]
argmax = lambda p: max(p, key=p.get)
tok_per_rec = OLD.get("provenance", {}).get("inline_tokens_per_record_at_250", 228.4)
b250 = jev_rows("e0_trec_coarse_in_B250")
if b250:
    tok_per_rec = round(sum(r["input_tokens"] for r in b250) / sum(r["questions"] for r in b250), 1)
e6s = {
    "n": len(gold6),
    "haiku_model": "claude-haiku-5-5",
    "haiku_acc": round(float(np.mean([h == g for h, g in zip(sw["haiku"], gold6, strict=True)])), 4),
    "jev_single_acc": round(float(np.mean([argmax(p) == g for p, g in zip(sw["jev_fwd"], gold6, strict=True)])), 4),
    "jev_ensemble_acc": round(
        float(
            np.mean(
                [
                    max(a, key=lambda k, a=a, b=b: a[k] + b[k]) == g
                    for a, b, g in zip(sw["jev_fwd"], sw["jev_rev"], gold6, strict=True)
                ]
            )
        ),
        4,
    ),
    "jev_repeat_label_changes": sum(
        argmax(a) != argmax(b) for a, b in zip(sw["jev_fwd"], sw["jev_fresh"], strict=True)
    ),
    "jev_single_usd": round(len(gold6) * tok_per_rec * PRICE_PER_TOKEN, 4),
    "haiku_wall_s": sw["haiku_wall_s"],
    "jev_wall_s": sw["jev_wall_s"],
    "speedup": round(sw["haiku_wall_s"] / sw["jev_wall_s"], 1),
    "cascade": e6["cascade"],
}
haiku_rows = [r for r in CLAUDE if r["tag"] == "e6_swap_haiku"]
if haiku_rows:
    e6s["haiku_list_usd"] = round(sum(r["usd_list"] for r in haiku_rows), 4)
e6s = {k: v for k, v in merge("E6_leaf_swap", e6s).items() if "haiku45" not in k}
if "haiku_list_usd" in e6s:
    e6s["cost_ratio_single"] = round(e6s["haiku_list_usd"] / e6s["jev_single_usd"], 1)
S["E6_leaf_swap"] = e6s
S["E6c_jev_request_format"] = read_json("e6c_shared_labels.json")
S["E6d_yesno_format"] = read_json("e6d_yesno_format.json")
S["E8h_haiku_offslice"] = read_json("e8h_haiku_offslice.json")
S["E5b_library_locate"] = {k: v for k, v in read_json("e5b_library_locate.json").items() if k != "ranks"}
e6b = read_json("e6b_haiku_nothink.json")
S["E6b_haiku_thinking_off"] = {k: e6b[k] for k in ("n", "model", "path", "acc", "parse_ok", "usd_list", "wall_s")}

# ---------------------------------------------------------------- E2: live RLM runs
e2 = pd.read_csv(R / "e2_scored.csv")
S["E2_live"] = {f"{k}/{l}": round(v, 3) for (k, l), v in e2.groupby(["kind", "leaf"]).score.mean().items()}
S["E2_live_n"] = {f"{k}/{l}": int(v) for (k, l), v in e2.groupby(["kind", "leaf"]).score.size().items()}

# ---------------------------------------------------------------- E7, E7b: InfiniteBench
e7 = pd.read_csv(R / "e7_infbench.csv")
# A: books that fit a ~150k-token window; B: 265k-560k-word books, split by whether they fit Haiku 5.5's 1M window.
e7["group"] = np.where(e7.set == "A_fits", "A_fits", np.where(e7.full_too_long, "B_too_long", "B_fits_1M"))
e7g = (
    e7.groupby("group")
    .agg(
        n=("id", "size"),
        jev_ok=("jev_ok", "mean"),
        full_ok=("full_ok", "mean"),
        book_words_min=("book_words", "min"),
        book_words_max=("book_words", "max"),
        ctx_words=("ctx_words", "mean"),
        windows=("windows", "mean"),
        jev_select_s=("jev_select_s", "mean"),
    )
    .round(3)
    .to_dict(orient="index")
)
e7_req = [r for r in JEV if r["tag"] == "e7_jev"]
tok_per_window = (
    float(np.mean([r["input_tokens"] for r in e7_req]))
    if e7_req
    else OLD.get("E7_infbench", {}).get("jev_tokens_per_window_request")
)
if tok_per_window:
    for g in e7g.values():  # selection cost = windows scored x tokens per window request x price
        g["jev_select_usd"] = round(g["windows"] * tok_per_window * PRICE_PER_TOKEN, 4)
answer_cost = defaultdict(list)  # Haiku cost per answered question (a prompt over the window is rejected at $0)
for r in CLAUDE:
    if r["tag"].startswith("e7_") and (r["usd_list"] or 0) > 0:
        answer_cost[r["tag"]].append(r["usd_list"])
old_e7 = OLD.get("E7_infbench", {})
S["E7_infbench"] = {
    **e7g,
    "haiku_model": "claude-haiku-5-5",
    "jev_tokens_per_window_request": round(tok_per_window, 1) if tok_per_window else None,
    "haiku_usd_per_question": {k: round(float(np.mean(v)), 4) for k, v in answer_cost.items()}
    or old_e7.get("haiku_usd_per_question", {}),
}
e7b = pd.read_csv(R / "e7b_truncated.csv")
S["E7b_truncated"] = {
    "n": len(e7b),
    "trunc_ok": round(float(e7b.trunc_ok.mean()), 3),
    "kept_share_mean": round(float(e7b.kept_share.mean()), 3),
}

# ---------------------------------------------------------------- E8: DSPy
S["E8a_dspy_leaf"] = read_json("e8a_dspy_leaf.json")
stock = pd.read_csv(R / "e8b_dspy_rlm.csv")  # stock dspy.RLM through the CLI: refused by Sonnet 5.5
S["E8b_stock_dspy_rlm"] = {
    "refused_runs": int(stock.error.fillna("").str.contains("can't help with this").sum()),
    "n": len(stock),
}
E8B_NOTE_RUN = ("e8b_dspy_rlm_api341_note.csv", "3.4.1")  # the reported dspy.RLM run (note field, Anthropic API)
if (R / E8B_NOTE_RUN[0]).exists():
    f, ver = R / E8B_NOTE_RUN[0], E8B_NOTE_RUN[1]
    n = pd.read_csv(f)
    S["E8b_dspy_rlm"] = {
        "dspy_version": ver,
        "variant": "note field instead of reasoning field",
        "path": "anthropic_api",
        "n": len(n),
        "errors": int(n.error.notna().sum()),
        "refusals": int(n.refusals.sum()),
        "score": n.groupby("kind").score.mean().round(3).to_dict(),
        "n_by_kind": n.groupby("kind").size().to_dict(),
        "jev_used_share": round(float(n.jev_tool_used.mean()), 2),
        "root_calls_mean": round(float(n.root_calls.mean()), 1),
        "usd_per_q": round(float((n.root_usd_list + n.sub_usd_list).mean()), 3),
    }
S["E8g_leakfree_calibration"] = read_json("e8g_leakfree.json")
S["E8g_leakfree_calibration_labels_in_state"] = read_json("e8g_leakfree_shared.json")

# ---------------------------------------------------------------- E9, E9b: OOLONG scaling
e9 = pd.read_csv(R / "e9_core.csv")
e9s = (
    e9.groupby("batch")
    .agg(
        n=("score", "size"),
        score=("score", "mean"),
        root_usd=("root_usd", "mean"),
        leaf_usd=("leaf_usd", "mean"),
        jev_items=("jev_items", "mean"),
        wall_s=("wall_s", "mean"),
    )
    .round(3)
    .to_dict(orient="index")
)
for b in e9s:
    e9s[b] = {**OLD.get("E9_core", {}).get(b, {}), **e9s[b]}
for b in e9s:
    rows = [r for r in RLM if r["tag"] == f"e9_{b}"]
    if rows and "direct" in b:  # read-everything baselines: share of data lines kept, and prompts over the window
        e9s[b]["prompt_too_long"] = sum(str(r.get("answer", "")).startswith(TOO_LONG) for r in rows)
        if all("kept_share" in r for r in rows):
            e9s[b]["kept_share_mean"] = round(float(np.mean([r["kept_share"] for r in rows])), 3)
S["E9_core"] = e9s
S["E9b_truncation_ceiling"] = read_json("e9b_truncation_ceiling.json")

# ---------------------------------------------------------------- E10: abstention (risk-coverage at real thresholds)
e10 = pd.read_csv(R / "e10_abstain.csv")
curve = []
for t in sorted(e10.conf.unique(), reverse=True):  # whole tie groups only: answer every question with conf >= t
    sel = e10[e10.conf >= t]
    curve.append(
        {"min_conf": round(float(t), 4), "coverage": round(len(sel) / len(e10), 3), "score": round(sel.score.mean(), 3)}
    )
at80 = max((c for c in curve if c["coverage"] <= 0.8), key=lambda c: c["coverage"])
S["E10_abstention"] = {
    "n": len(e10),
    "score_all": round(e10.score.mean(), 3),
    "near_80pct": at80,
    "top_tie_group": curve[0],
    "curve": curve,
    "by_conf_bin": e10.groupby(pd.cut(e10.conf, [0, 0.5, 0.8, 0.95, 1.0], include_lowest=True), observed=True)
    .score.agg(["size", "mean"])
    .round(3)
    .reset_index()
    .astype(str)
    .to_dict(orient="records"),
    "share_conf_above_0.95": round(float((e10.conf > 0.95).mean()), 3),
}

# ---------------------------------------------------------------- E11: escalation gate
S["E11_gate"] = pd.read_csv(R / "e11_gate.csv").to_dict(orient="records")

# ---------------------------------------------------------------- E12: BrowseComp-Plus (aggregates only)
settings = (
    ("E12_bcp", "e12_bcp.jsonl", 14),
    ("E12_bcp_t30", "e12_bcp_t30.jsonl", 30),
    ("E12_bcp_t30_nojev", "e12_bcp_t30_nojev.jsonl", 30),
)
closed = pd.read_json(R / "e12_bcp_closedbook.jsonl", lines=True) if (R / "e12_bcp_closedbook.jsonl").exists() else None
for key, name, turns in settings:
    if not (R / name).exists():  # decrypted answers are local only; keep the committed aggregates
        if key in OLD:
            S[key] = OLD[key]
        continue
    b = pd.read_json(R / name, lines=True)
    out = {
        "n": len(b),
        "turn_limit": turns,
        "accuracy": round(b.correct.mean(), 3),
        "jev_usd_per_q": round(b.jev_usd.mean(), 3),
        "root_list_usd_per_q": round(b.root_usd.mean(), 3),
        "wall_s": round(b.wall_s.mean()),
        "turns_mean": round(b.turns.mean(), 1),
        "hit_turn_limit": int((b.turns >= turns).sum()),
        "exact_match_lower_bound": round(float((b.judged_by == "exact").mean()), 3),
    }
    if closed is not None:
        m = b.merge(closed[["qid", "correct"]], on="qid", suffixes=("", "_cb"))
        out["correct_and_closedbook_wrong"] = int((m.correct & ~m.correct_cb).sum())
    tag = "e12_bcp" + name[len("e12_bcp") : -len(".jsonl")]
    tx = [Path(f).read_text() for f in glob.glob(str(R / "rlm_transcripts" / f"{tag}_[0-9]*.txt"))]
    if tx:  # how the root worked: how often each tool or pattern appears in its code (counts only, no content)
        code = ["\n".join(re.findall(r"```python\n(.*?)```", t, re.S)) for t in tx]
        out["code_mentions"] = {
            k: sum(len(re.findall(p, c)) for c in code)
            for k, p in {
                "jev_scout": r"\bjev_scout\(",
                "jev_search": r"\bjev_search\(",
                "read": r"(?<![\w.])read\(",
                "re.search": r"\bre\.search\(",
                "enumerate(docs)": r"enumerate\(docs\)",
            }.items()
        }
        toks = [int(m[1].replace(",", "")) for t in tx if (m := re.search(r"~([\d,]+) tokens", t[:2000]))]
        if toks:
            out["haystack_tokens_mean_est"] = int(np.mean(toks))
            out["haystack_tokens_range_est"] = [min(toks), max(toks)]
    S[key] = out
if closed is not None:
    S["E12_bcp_closedbook"] = {"n": len(closed), "accuracy": round(closed.correct.mean(), 3)}
elif "E12_bcp_closedbook" in OLD:
    S["E12_bcp_closedbook"] = OLD["E12_bcp_closedbook"]

# ---------------------------------------------------------------- provenance for other numbers in the write-up
OLD_P = OLD.get("provenance", {})
P = {}
P["jev_price_usd_per_mtok"] = round(PRICE_PER_TOKEN * 1e6, 4)  # https://docs.typesafe.ai/models
P["jev_rate_limit_rps_documented"] = 80  # https://docs.typesafe.ai/models


# Ledger-derived values; carried over from the previous summary when the ledger rows aren't available.
def per_record(rows):
    return round(sum(r["input_tokens"] for r in rows) / sum(r["questions"] for r in rows), 1) if rows else None


def locate_single_query_usd(rows):
    """What one query would cost alone: the window pass minus the other queries' question text (estimated with the
    library's token estimate, without its per-question safety margin), plus that query's share of the line pass."""
    from rlm_jev._client import estimate_tokens
    from rlm_jev.locate import L1

    qs = [n["query"] for n in json.loads(Path("experiments/assets/needles.json").read_text())]
    q_tok = [estimate_tokens(L1.format(q=q)) for q in qs]
    l1 = [r for r in rows if r["tag"] == "e5_l1"]
    l1_one = sum(r["input_tokens"] for r in l1) - len(l1) * (sum(q_tok) - np.mean(q_tok))
    l2_usd = sum(r["usd"] for r in rows if r["tag"] == "e5_l2")
    return round(float(l1_one * PRICE_PER_TOKEN + l2_usd / len(qs)), 3)


b1 = jev_rows("e0_trec_coarse_in_B1")
locate_rows = jev_rows("e5_l1", "e5_l2")
n_queries = len(json.loads(Path("experiments/assets/needles.json").read_text()))
from_ledger = {
    "inline_tokens_per_record_at_250": tok_per_rec if b250 else None,
    "inline_latency_s_per_250_record_request": round(float(np.mean([r["latency_s"] for r in b250])), 2)
    if b250
    else None,
    "tokens_per_record_one_per_request": round(sum(r["input_tokens"] for r in b1) / len(b1), 1) if b1 else None,
    "e0b_keyed_B250_tokens_per_record": per_record(jev_rows("e0b_keyed_B250")),
    "e0b_keyed_lines_B250_tokens_per_record": per_record(jev_rows("e0b_keyed_lines_B250")),
    "locate_usd_total": round(sum(r["usd"] for r in locate_rows), 4) if locate_rows else None,
    "locate_usd_single_query_est": locate_single_query_usd(locate_rows) if locate_rows else None,
}
for k, v in from_ledger.items():
    if v is None:
        v = OLD_P.get(k)
    if v is not None:
        P[k] = v
P["locate_queries_sharing_one_read"] = n_queries
if "locate_usd_total" in P:
    P["locate_usd_per_query_amortized"] = round(P["locate_usd_total"] / n_queries, 4)
P["locate_timing_s"] = e5["meta"]["timing_s"]
P["locate_l1_seconds"] = e5["meta"]["l1_seconds"]
# Jev sub-call cost a user would pay without a label cache: items labeled per question x tokens per record x price.
P["jev_leaf_usd_per_question_uncached"] = {
    b: round(float(g.jev_items.mean()) * tok_per_rec * PRICE_PER_TOKEN, 4)
    for b, g in e9[e9.arm == "jev"].groupby("batch")
}
for f in sorted(glob.glob(str(R / "e5c_*.json"))):
    d = json.loads(Path(f).read_text())
    P.setdefault("needles_not_in_embedding_top100", {})[d["embedding_model"]] = int(
        sum(r >= 10**6 for r in d["rr_ranks"])
    )
e4 = pd.read_csv(R / "e4_counting.csv")
P["poisson_binomial_90pct_coverage_by_dataset"] = e4.groupby("ds").cover90.mean().round(3).to_dict()
P["e1b_datasets"] = sorted(e1b.dataset.unique().tolist())
P["e8d_cross_format"] = read_json("e8d_reanchor_transfer.json")
P["e8f_cross_distribution"] = read_json("e8f_disjoint_weights.json")
S["provenance"] = P

# ---------------------------------------------------------------- root turns lost to tool-call markup (OOLONG RLM runs)
# A turn is lost when the root wraps its code in tool-call markup: no ```python block at all, or stray tags that
# make the block a syntax error.
turns_by_arm, markup_by_arm = defaultdict(int), defaultdict(int)
for f in glob.glob(str(R / "rlm_transcripts" / "e[29]_*.txt")):
    arm = "llm" if "_llm" in Path(f).name else "jev"
    for t in re.split(r"\n--- turn \d+ ---\n", Path(f).read_text())[1:]:
        turns_by_arm[arm] += 1
        no_block = "no python block found" in t and bool(re.search(r"<(invoke|parameter|python|function_calls)", t))
        syntax = "SyntaxError" in t and bool(re.search(r"^\s*</?[a-zA-Z_:]+", t, re.M))
        markup_by_arm[arm] += no_block or syntax
if turns_by_arm:
    P["root_turns_lost_to_markup"] = {
        arm: {
            "turns": turns_by_arm[arm],
            "lost": markup_by_arm[arm],
            "share": round(markup_by_arm[arm] / turns_by_arm[arm], 3),
        }
        for arm in sorted(turns_by_arm)
    }

# ---------------------------------------------------------------- spend on the reported runs
# An estimate of what the reported runs cost, each counted once; smoke tests, superseded runs and side tests are left
# out where their tags allow. RLM roots and Claude sub-calls use the costs recorded per run; Jev labeling inside RLM
# runs uses labeled items x measured tokens per item (later runs reused cached labels); dspy.RLM's Jev calls are
# attributed by the reported run's time window; labeling experiments and Claude sub-call runs count their latest run.
# A few side tests shared production tags, so the labeling total can be off by a few cents.
REPORTED_JEV_TAGS = (
    "e0_",
    "e0b_",
    "e1_",
    "e1b_",
    "e3_",
    "e4_",
    "e5_",
    "e6_",
    "e6c_",
    "e8a_",
    "e8c_",
    "e8d_",
    "e8f_",
    "e10_",
    "e11_",
    "trec_label",
)
REPORTED_CLAUDE_TAGS = ("e6_swap_haiku", "e6_cascade_haiku", "e11_haiku", "e7_", "e7b_", "e8h_haiku")
E8B_REPORTED = "e8b_api341_note"


def e8b_jev_usd():
    fs = glob.glob(str(R / "rlm_transcripts" / f"{E8B_REPORTED}_*"))
    rows = [r for r in RLM if r["tag"].startswith(E8B_REPORTED)]
    if not fs or not rows:
        return None
    lo = min(map(os.path.getmtime, fs)) - max(r["wall_s"] for r in rows) - 5
    hi = max(map(os.path.getmtime, fs)) + 5
    return sum(r["usd"] for r in JEV if r["tag"] == "e8b_leaf" and lo <= r["t"] <= hi)


if JEV and CLAUDE and RLM:
    e9c = pd.read_csv(R / "e9_core.csv")
    e2c = pd.read_csv(R / "e2_scored.csv").query("kind == 'pairs32k'")
    e12 = [r for f in ("e12_bcp.jsonl", "e12_bcp_t30.jsonl", "e12_bcp_t30_nojev.jsonl") for r in read_jsonl(R / f)]
    e12cb = read_jsonl(R / "e12_bcp_closedbook.jsonl")
    e7g = S["E7_infbench"]
    jev = {
        "labeling_experiments": sum(
            r["usd"]
            for tag in {r["tag"] for r in JEV if r["tag"].startswith(REPORTED_JEV_TAGS)}
            for r in last_run([x for x in JEV if x["tag"] == tag])
        ),
        "rlm_leaf_oolong": (e9c.jev_items.sum() + e2c.jev_items.sum()) * tok_per_rec * PRICE_PER_TOKEN,
        "infinitebench_selection": sum(
            e7g[g]["jev_select_usd"] * e7g[g]["n"] for g in ("A_fits", "B_fits_1M", "B_too_long")
        ),
        "browsecomp_plus": sum(r.get("jev_usd", 0) for r in e12),
        "dspy_rlm": e8b_jev_usd() or 0.0,
    }
    claude = {
        "oolong_rlm_and_direct": e9c.root_usd.sum() + e9c.leaf_usd.sum() + e2c.root_usd.sum(),
        "browsecomp_plus_roots": sum(r["root_usd"] for r in e12 + e12cb),
        "subcalls_and_infinitebench": sum(
            r["usd_list"]
            for tag in {r["tag"] for r in CLAUDE if r["tag"].startswith(REPORTED_CLAUDE_TAGS)}
            for r in last_run([x for x in CLAUDE if x["tag"] == tag])
        ),
    }
    api = {
        "dspy_rlm": sum(
            (r.get("root_usd_list") or 0) + (r.get("sub_usd_list") or 0)
            for r in RLM
            if r["tag"].startswith(E8B_REPORTED)
        ),
        "haiku_thinking_off": S["E6b_haiku_thinking_off"]["usd_list"]
        + read_json("e8h_haiku_offslice.json")["haiku_api_no_thinking"]["usd_list"],
    }
    S["reported_run_spend"] = {
        "jev_usd": round(sum(jev.values()), 2),
        "claude_cli_list_usd": round(sum(claude.values()), 2),
        "anthropic_api_usd": round(sum(api.values()), 2),
        "jev_by_part": {k: round(v, 3) for k, v in jev.items()},
        "claude_cli_by_part": {k: round(v, 2) for k, v in claude.items()},
        "anthropic_api_by_part": {k: round(v, 3) for k, v in api.items()},
        "not_logged": "BrowseComp-Plus judge calls (Sonnet, 3 votes per non-exact answer)",
    }
elif "reported_run_spend" in OLD:
    S["reported_run_spend"] = OLD["reported_run_spend"]

(R / "summary.json").write_text(json.dumps(S, indent=1, default=float))
print(f"wrote {R / 'summary.json'}: {len(S)} sections")
