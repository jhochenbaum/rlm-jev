"""Render the write-up's figures (static PNG, light surface, 1600x900).

Palette: blue #2a78d6 for our results, orange #eb6834 for comparisons we ran ourselves, hatched gray for numbers
reported by others (plain gray marks a baseline). All numbers come from results/summary.json.
"""

import json as _json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parent  # figures/ (PNG files live next to this script)
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, REF = "#2a78d6", "#eb6834", "#b9b8b1"
FONT = "Helvetica Neue" if any("Helvetica Neue" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 15,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "axes.edgecolor": AXIS,
        "text.parse_math": False,
    }
)
W, H, DPI = 8, 4.5, 200  # 1600x900

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary.json"


def S():
    """All numbers come from results/summary.json (built by experiments/summarize.py)."""
    return _json.loads(SUMMARY.read_text())


PAPER = {  # RLM paper (Zhang, Kraska, Khattab), OOLONG-Pairs F1, GPT-5; external numbers, cited
    "GPT-5, no RLM (paper)": 0.1,
    "RLM, GPT-5, depth 1 (paper)": 58,
    "RLM, GPT-5, depth 3 (paper)": 76,
}


def fig(title, subtitle=None):
    f = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fit_width(f.text(0.04, 0.94, title, fontsize=19, weight="bold", va="top", color=INK), 0.92)
    if subtitle:
        fit_width(f.text(0.04, 0.865, subtitle, fontsize=12.5, va="top", color=INK2), 0.92)
    return f


def foot(f, text):
    import textwrap

    f.text(0.04, 0.025, textwrap.fill(text, 122), fontsize=9.5, color=MUTED, va="bottom", linespacing=1.35)


def style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbar(ax, y, w, h, color, hatch=None):
    """Horizontal bar; hatched when the value is quoted from another paper."""
    if w > 0:
        ax.barh(y, w, height=h, color=color, hatch=hatch, edgecolor=SURFACE if hatch else color, linewidth=0)


def fit_width(t, width, min_size=7):
    """Shrink text artist `t` until it is at most `width` wide (in its axes' data or axes units)."""
    f = t.get_figure()
    r = f.canvas.get_renderer()
    inv = t.get_transform().inverted()
    while t.get_fontsize() > min_size:
        bb = t.get_window_extent(r)
        (x0, _), (x1, _) = inv.transform([(bb.x0, 0), (bb.x1, 0)])
        if x1 - x0 <= width:
            break
        t.set_fontsize(t.get_fontsize() - 0.5)
    return t


def save(f, name):
    f.savefig(OUT / name, facecolor=SURFACE)
    plt.close(f)
    print("wrote", OUT / name)


# 0 ── architecture diagram
def diagram():
    P = S()["provenance"]
    f = fig(
        "Jev answers the sub-calls of a Recursive Language Model",
        "The LLM writes the program. Jev makes the per-item semantic decisions. Code does the math.",
    )
    ax = f.add_axes([0.03, 0.27, 0.94, 0.55])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge, fill="#ffffff", mono=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill, edgecolor=edge, linewidth=1.6
            )
        )
        fit_width(ax.text(x + w / 2, y + h - 4, title, ha="center", va="top", fontsize=14, weight="bold"), w * 0.9)
        body = ax.text(
            x + w / 2,
            y + h - 10,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=10,
            color=INK2,
            linespacing=1.35,
            family="Menlo" if mono else None,
        )
        fit_width(body, w * 0.9)

    def arrow(x1, y1, x2, y2, label=None, dy=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.5})
        if label:
            ax.text(
                (x1 + x2) / 2, y1 + dy, label, ha="center", va="bottom" if dy > 0 else "top", fontsize=9.5, color=MUTED
            )

    box(
        0.5,
        2,
        19,
        36,
        "Huge context",
        ["1M+ tokens of", "books, code, logs,", "held in a REPL", "variable, never", "in the prompt"],
        AXIS,
    )
    box(
        26.5,
        2,
        20,
        36,
        "Root LLM",
        ["System 2 (Claude)", "writes Python to", "parse, filter, join,", "count and compare", "exactly, in code"],
        INK2,
    )
    box(
        56,
        2,
        26,
        36,
        "Jev sub-calls",
        [
            "jev_classify(items, labels)",
            "jev_yesno(items, q)",
            "jev_locate(docs, queries)",
            "System 1, calibrated",
            f"~{P['inline_tokens_per_record_at_250']:.0f} tokens per item,",
            f"~{P['inline_latency_s_per_250_record_request']:.1f} s per 250 items",
        ],
        BLUE,
        "#eef4fc",
    )
    box(87.5, 10, 12, 20, "Answer", ["with", "confidence"], AXIS)
    arrow(19.5, 20, 26.5, 20)
    arrow(46.5, 24, 56, 24, "items")
    arrow(56, 15, 46.5, 15, "scores", -1.8)
    arrow(82, 20, 87.5, 20)
    f.text(
        0.5,
        0.17,
        "Sub-calls come back as calibrated probabilities: nothing to parse, 250 items per request",
        ha="center",
        fontsize=12,
        color=BLUE,
        weight="bold",
    )
    e6, nt = S()["E6_leaf_swap"], S()["E6b_haiku_thinking_off"]
    foot(
        f,
        f"Same 500 TREC records: Jev {e6['jev_single_acc'] * 100:.1f}% for ${e6['jev_single_usd']:.4f} in "
        f"{e6['jev_wall_s']:.1f} s; Claude Haiku 5.5 {nt['acc'] * 100:.1f}% for ${nt['usd_list']:.4f} in {nt['wall_s']:.1f} s "
        f"with thinking off, {e6['haiku_acc'] * 100:.1f}% for ${e6['haiku_list_usd']:.3f} in {e6['haiku_wall_s']:.0f} s with it on.",
    )
    save(f, "0_architecture.png")


# 1 ── stat tiles
def tiles():
    f = fig(
        "Jev vs Claude Haiku 5.5, with and without thinking",
        "Same 500 records, same labels; Haiku thinking off / on",
    )
    e6, nt = S()["E6_leaf_swap"], S()["E6b_haiku_thinking_off"]
    stats = [
        (
            f"{e6['jev_single_acc'] * 100:.1f}%",
            f"accuracy\nHaiku {nt['acc'] * 100:.1f}% / {e6['haiku_acc'] * 100:.1f}%",
        ),
        (
            f"${e6['jev_single_usd']:.4f}",
            f"cost, list price\nHaiku ${nt['usd_list']:.4f} / ${e6['haiku_list_usd']:.4f}",
        ),
        (f"{e6['jev_wall_s']:.1f} s", f"wall time\nHaiku {nt['wall_s']:.1f} s / {e6['haiku_wall_s']:.0f} s"),
        (f"{S()['E0_inline_ece_at_250']:.3f}", "calibration error (ECE), 400\nTREC records, 250 per request"),
    ]
    for i, (v, lab) in enumerate(stats):
        ax = f.add_axes([0.04 + i * 0.235, 0.2, 0.215, 0.5])
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0,rounding_size=0.06",
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor=GRID,
                linewidth=1.2,
            )
        )
        val = ax.text(
            0.5,
            0.66,
            v,
            ha="center",
            va="center",
            fontsize=26 if len(v) <= 6 else 20 if len(v) <= 11 else 16,
            weight="bold",
            color=INK if i == 0 else BLUE,
            transform=ax.transAxes,
        )
        fit_width(val, 0.88)
        lab_t = ax.text(
            0.5,
            0.28,
            lab,
            ha="center",
            va="center",
            fontsize=10.5,
            color=INK2,
            transform=ax.transAxes,
            linespacing=1.45,
        )
        fit_width(lab_t, 0.9)
    foot(
        f,
        "Jev: one pass, input tokens at $0.042/Mtok. Haiku 5.5: tokens x list "
        "price ($0.10 in / $0.50 out per Mtok); thinking on is the Claude Code CLI default (time includes CLI startup), "
        "thinking off is the Anthropic API. Task: TREC coarse question classification.",
    )
    save(f, "1_leaf_swap_tiles.png")


# 2 ── inline vs state pointers
def inline_vs_state():
    f = fig(
        "Don't make Jev find items by position",
        "Per-record accuracy on 400 TREC records vs records packed into one Jev request",
    )
    ax = f.add_axes([0.08, 0.2, 0.66, 0.58])
    style(ax, "y")
    xs = [1, 25, 100, 250]
    e0 = S()["E0_inline_vs_state_trec_acc"]
    inline = [e0[f"in_B{b}"] * 100 for b in xs]
    state = [e0[f"B{b}"] * 100 for b in xs]
    keyed = [S()["E0b_keyed_state_trec_acc"][f"keyed_B{b}"] * 100 for b in xs]
    for ys, c, ls in [(inline, BLUE, "-"), (keyed, INK2, "--"), (state, ORANGE, "-")]:
        ax.plot(xs, ys, color=c, lw=2, ls=ls, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("records per request (log scale)", fontsize=11, color=MUTED, labelpad=6)
    ok = [y for ys in (inline, keyed) for y in ys[1:]]
    ax.text(290, 95.5, f"Visible ID or inline:\n{min(ok):.0f}–{max(ok):.0f}%", color=INK, fontsize=11.5, va="center")
    ax.text(
        290,
        state[-1],
        f"Counting positions,\n“label records[i]”: {state[-1]:.0f}%",
        color=INK,
        fontsize=11.5,
        va="center",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, marker="o", label="item inside its own question"),
            Line2D([], [], color=INK2, lw=2, ls="--", marker="o", label="items in the state under visible IDs (r17)"),
            Line2D([], [], color=ORANGE, lw=2, marker="o", label="items in the state, found by position (records[17])"),
        ],
        loc="lower left",
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=1,
        fontsize=10.5,
        labelcolor=INK2,
    )
    P = S()["provenance"]
    lo = min(
        P["e0b_keyed_lines_B250_tokens_per_record"],
        P["e0b_keyed_B250_tokens_per_record"],
        P["inline_tokens_per_record_at_250"],
    )
    hi = max(
        P["e0b_keyed_lines_B250_tokens_per_record"],
        P["e0b_keyed_B250_tokens_per_record"],
        P["inline_tokens_per_record_at_250"],
    )
    foot(
        f,
        f"Both working layouts: ~{lo:.0f}–{hi:.0f} tokens per record at 250 per request, "
        f"vs ~{P['tokens_per_record_one_per_request']:.0f} for one request per item.",
    )
    save(f, "2_inline_vs_state.png")


# 3 ── OOLONG-Pairs
def pairs():
    f = fig(
        "OOLONG-Pairs: O(n) Jev labels + an exact join",
        "F1 on the pairwise-aggregation task. Hatched gray = reported in the RLM paper (different setup).",
    )
    ax = f.add_axes([0.38, 0.19, 0.55, 0.6])
    style(ax, "x")
    s_ = S()
    f3 = s_["E3_pairs_f1"]
    rows = [(k, v, REF, "///") for k, v in PAPER.items()] + [
        (
            f"Sonnet root + Jev, live (32k, n={s_['E2_live_n']['pairs32k/jev']})",
            round(s_["E2_live"]["pairs32k/jev"] * 100, 1),
            BLUE,
            None,
        ),
        ("Fixed program + Jev (32k)", round(f3["32768"]["jev_ens"] * 100, 1), BLUE, None),
        ("Fixed program + Jev (131k)", round(f3["131072"]["jev_ens"] * 100, 1), BLUE, None),
    ]
    for i, (_lab, v, c, h) in enumerate(reversed(rows)):
        hbar(ax, i, v, 0.55, c, h)
        ax.text(v + 1.2, i, f"{v:g}", va="center", fontsize=12, color=INK, weight="bold" if c == BLUE else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=11.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    foot(
        f,
        "Not head-to-head: the paper uses GPT-5 for the root and GPT-5-mini for sub-calls; ours uses Claude Sonnet 5.5 for the root and Jev for sub-calls. The fixed program scores 100 with gold labels, so the gap is leaf error.",
    )
    save(f, "3_oolong_pairs.png")


# 4a ── retrieval
def retrieval():
    f = fig(
        "locate(): 30 hidden facts in 1.2M tokens of novels",
        "Recall@5. Queries share no content words with the facts they target.",
    )
    ax = f.add_axes([0.37, 0.24, 0.55, 0.54])
    style(ax, "x")
    s_ = S()
    P = s_["provenance"]
    t = s_["E5_locate"]["table"]["R@5"]
    m = s_["E5c_modern"]
    tiers = [
        ("0.6B", "Qwen/Qwen3-Embedding-0.6B"),
        ("4B", "Qwen/Qwen3-Embedding-4B + Qwen3-Reranker-4B"),
        ("8B", "Qwen/Qwen3-Embedding-8B + Qwen3-Reranker-8B"),
    ]
    rows = [(f"Qwen3 {sz} embed + {sz} rerank", m[k]["emb+rerank"]["R@5"] * 100, ORANGE) for sz, k in tiers if k in m]
    rows.append(("Jev locate (two passes)", t["jev_zoom"] * 100, BLUE))
    for i, (_lab, v, c) in enumerate(reversed(rows)):
        hbar(ax, i, v, 0.55, c)
        ax.text(
            v + 1.2, i, f"{v:.0f}%", va="center", fontsize=12.5, color=INK, weight="bold" if c == BLUE else "normal"
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=12)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    foot(
        f,
        f"Each reranker scores its embedder's top 100, after the corpus is embedded and indexed. Jev needs no index: "
        f"{P['locate_l1_seconds']}–{P['locate_timing_s']['jev_zoom']} s for all {P['locate_queries_sharing_one_read']} "
        f"queries, ~${P['locate_usd_per_query_amortized']:.3f} each when they share one read (~${P['locate_usd_single_query_est']:.2f} "
        "for a query alone). The facts were written independently of the retrieval methods.",
    )
    save(f, "4a_locate_retrieval.png")


def grouped(ax, groups, series, ymax, fmt, bold_idx, skip_zero=False):
    n = len(series)
    bw = 0.32
    for si, (vals, c, _) in enumerate(series):
        xs = [g + (si - (n - 1) / 2) * (bw + 0.02) for g in range(len(groups))]
        ax.bar(xs, vals, width=bw, color=c, linewidth=0)
        for x, v in zip(xs, vals, strict=False):
            if skip_zero and v == 0:
                continue
            ax.text(
                x,
                v + ymax * 0.02,
                fmt.format(v),
                ha="center",
                fontsize=12.5,
                weight="bold" if si == bold_idx else "normal",
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, ymax * 1.08)


# 4b ── InfiniteBench
def infbench():
    f = fig(
        "Jev as an attention compressor for long books",
        "InfiniteBench En.MC: Haiku 5.5 answers from the whole book, or from ~8k words Jev picked",
    )
    ax = f.add_axes([0.1, 0.2, 0.86, 0.5])
    style(ax, "y")
    s_ = S()
    e7 = s_["E7_infbench"]
    g = [e7["A_fits"], e7["B_fits_1M"], e7["B_too_long"]]
    trunc = s_.get("E7b_truncated")  # the 560k-word book, truncated to fit (most recent text kept)
    grouped(
        ax,
        [
            f"{round(x['book_words_min'] / 1000)}k–{round(x['book_words_max'] / 1000)}k words (n={x['n']})"
            if x["book_words_min"] != x["book_words_max"]
            else f"{round(x['book_words_min'] / 1000)}k words (n={x['n']})"
            for x in g
        ],
        [
            ([x["full_ok"] * 100 for x in g[:2]] + [trunc["trunc_ok"] * 100 if trunc else 0], ORANGE, ""),
            ([x["jev_ok"] * 100 for x in g], BLUE, ""),
        ],
        100,
        "{:.0f}%",
        1,
        skip_zero=True,
    )
    if trunc:
        ax.text(2 - 0.17, 4, "truncated\nto fit", ha="center", va="bottom", fontsize=10, color=SURFACE, weight="bold")
    else:
        ax.text(2 - 0.17, 3, "doesn't\nfit", ha="center", va="bottom", fontsize=10.5, color=MUTED)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    f.legend(
        handles=[
            Patch(color=ORANGE, label="Haiku 5.5 reads the whole book (1M-token window)"),
            Patch(color=BLUE, label="Jev selects ~8k words, then Haiku answers"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.04, 0.83),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=INK2,
    )
    c = e7["haiku_usd_per_question"]
    wr = lambda x: f"{round(x['book_words_min'] / 1000)}k–{round(x['book_words_max'] / 1000)}k"
    jev_a = g[0]["jev_select_usd"] + c["e7_A_fits_jevctx"]
    jev_b = [x["jev_select_usd"] + c["e7_B_long_jevctx"] for x in g[1:]]
    kept = f" is cut to its last {trunc['kept_share_mean']:.0%}" if trunc else " doesn't fit"
    foot(
        f,
        f"Per question at list price, selection included. With Jev: ${jev_a:.2f} ({wr(g[0])} words), "
        f"${jev_b[0]:.2f} ({wr(g[1])}), ${jev_b[1]:.2f} ({round(g[2]['book_words_min'] / 1000)}k). "
        f"Whole book: ${c['e7_A_fits_full']:.2f} ({wr(g[0])}) and ${c['e7_B_long_full']:.2f} ({wr(g[1])}). "
        f"The {round(g[2]['book_words_min'] / 1000)}k-word book (~1.04M tokens){kept}.",
    )
    save(f, "4b_infinitebench.png")


# 5 ── calibration: small multiples, one axis each
def calibration():
    f = fig(
        "Six label weights cut the count error (leak-free fit)",
        "Fit on 300 labeled records, which were then deleted from every context before scoring",
    )
    c = S()["E8g_leakfree_calibration"]
    panels = [
        (
            "Per-label count error, lower is better",
            [c["unweighted"]["count_relerr"] * 100, c["weighted"]["count_relerr"] * 100],
            "{:.1f}%",
            10,
            ["0%", "2%", "4%", "6%", "8%", "10%"],
        ),
        (
            "OOLONG TREC score, higher is better",
            [c["unweighted"]["oolong_overall"], c["weighted"]["oolong_overall"]],
            "{:.3f}",
            1.0,
            None,
        ),
    ]
    for k, (title, vals, fmt, ymax, ticks) in enumerate(panels):
        ax = f.add_axes([0.08 + k * 0.48, 0.21, 0.4, 0.5])
        style(ax, "y")
        ax.bar([0, 1], vals, width=0.45, color=[REF, BLUE], linewidth=0)
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.03, fmt.format(v), ha="center", fontsize=13.5, weight="bold" if i else "normal")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["uncalibrated", "calibrated"], fontsize=11.5)
        if ticks:
            ax.set_yticks([0, 2, 4, 6, 8, 10])
            ax.set_yticklabels(ticks)
        ax.set_title(title, fontsize=12, color=INK2, loc="left", pad=10)
    foot(
        f,
        f"At 131k tokens the score went from {c['unweighted']['oolong_131k']:.2f} to {c['weighted']['oolong_131k']:.2f}; "
        "the gain is in shorter contexts. Weights fit through another request format or on another TREC subset "
        "lowered the 131k score.",
    )
    save(f, "5_calibration.png")


# 6 ── dspy.RLM vs harness
def dspy_vs_harness():
    s_ = S()
    d, h = s_["E8b_dspy_rlm"], s_["E2_live"]
    f = fig(
        "dspy.RLM + Jev tools vs a minimal custom harness",
        f"Same questions, Sonnet 5.5 root, Jev sub-calls; dspy {d['dspy_version']} with a one-line note field",
    )
    ax = f.add_axes([0.08, 0.2, 0.88, 0.5])
    style(ax, "y")
    sc, n = d["score"], d["n_by_kind"]
    grouped(
        ax,
        [f"OOLONG 131k (n={n['synth131k']})", f"OOLONG 1M (n={n['synth1M']})", f"OOLONG-Pairs F1 (n={n['pairs32k']})"],
        [
            ([sc["synth131k"], sc["synth1M"], sc["pairs32k"]], BLUE, ""),
            ([h["synth131k/jev"], h["synth1M/jev"], h["pairs32k/jev"]], ORANGE, ""),
        ],
        1.0,
        "{:.2f}",
        -1,
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    f.legend(
        handles=[
            Patch(color=BLUE, label="dspy.RLM (Deno sandbox) + Jev tools"),
            Patch(color=ORANGE, label="minimal custom harness + Jev"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.04, 0.83),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=INK2,
    )
    foot(
        f,
        f"Tied within noise. The dspy.RLM root used Jev in {d['jev_used_share'] * 100:.0f}% of runs; ${d['usd_per_q']:.2f} per question via the Anthropic API. "
        "Stock dspy.RLM is refused by Sonnet 5.5 (reasoning_extraction), so its step-by-step `reasoning` field was swapped for a one-line note.",
    )
    save(f, "6_dspy_rlm.png")


ROOT = Path(__file__).resolve().parent.parent


# 7 ── calibrated abstention
def abstention():
    e = S()["E10_abstention"]
    cv = sorted(e["curve"], key=lambda r: -r["coverage"])
    f = fig(
        "Jev's probabilities show which answers are fragile",
        f"Fixed OOLONG program, {e['n']} test questions at 131k tokens, answering only the most confident",
    )
    ax = f.add_axes([0.09, 0.27, 0.84, 0.52])
    style(ax, "y")
    # One point per distinct confidence threshold; questions tied at a threshold are kept or skipped together.
    ax.plot(
        [r["coverage"] * 100 for r in cv],
        [r["score"] * 100 for r in cv],
        color=BLUE,
        lw=2,
        zorder=3,
    )
    top, near = e["top_tie_group"], e["near_80pct"]
    for r, bold in [({"coverage": 1.0, "score": e["score_all"]}, False), (near, True), (top, True)]:
        ax.plot(r["coverage"] * 100, r["score"] * 100, "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.text(
            r["coverage"] * 100,
            r["score"] * 100 + 2.2,
            f"{r['score']:.2f}",
            ha="center",
            va="bottom",
            fontsize=12.5,
            weight="bold" if bold else "normal",
        )
    ax.text(
        top["coverage"] * 100,
        top["score"] * 100 - 2.5,
        f"top {top['coverage']:.0%} all at\nconfidence {top['min_conf']:.1f}",
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
        linespacing=1.3,
    )
    ax.set_xlim(102, 50)
    ax.set_ylim(70, 106)
    xs = [100, 90, 80, 70, 60]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.set_yticks([70, 80, 90, 100])
    ax.set_yticklabels(["0.70", "0.80", "0.90", "1.00"])
    ax.set_xlabel(
        "share of questions answered (the rest are skipped as low-confidence)", fontsize=11, color=MUTED, labelpad=4
    )
    foot(
        f,
        "Confidence = how often the answer survives when each line's label is resampled from Jev's probabilities "
        "(60 draws). No extra model calls. The curve stops where the most confident answers all tie.",
    )
    save(f, "7_abstention.png")


# 8 ── System-1 -> System-2 gate (per-threshold curve from results/e11_gate.csv)
def gate():
    import pandas as pd

    G = pd.read_csv(ROOT / "results/e11_gate.csv")
    f = fig(
        "Send only the uncertain items to the bigger model",
        "Jev labels everything; items it is unsure about go to Claude Haiku 5.5",
    )
    for k, (ds, name) in enumerate([("multinli", "MultiNLI"), ("negation", "Negation")]):
        g = G[G.dataset == ds].sort_values("escalated")
        ax = f.add_axes([0.08 + k * 0.5, 0.27, 0.38, 0.47])
        style(ax, "y")
        x, y = g.escalated * 100, g.record_acc * 100
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.scatter([x.iloc[0], x.iloc[-1]], [y.iloc[0], y.iloc[-1]], s=70, color=INK2, zorder=4)
        ax.text(x.iloc[0] + 2, y.iloc[0] - 1.3, f"Jev only\n{y.iloc[0]:.1f}%", fontsize=10.5, color=INK2, va="top")
        ax.text(
            x.iloc[-1] - 2,
            y.iloc[-1] - 1.3,
            f"Haiku only\n{y.iloc[-1]:.1f}%",
            fontsize=10.5,
            color=INK2,
            va="top",
            ha="right",
        )
        best = g.iloc[g.record_acc.values.argmax()]
        ax.text(
            best.escalated * 100,
            best.record_acc * 100 + 1.0,
            f"{best.record_acc * 100:.1f}%",
            ha="center",
            fontsize=12,
            weight="bold",
        )
        ax.set_xlim(-5, 105)
        ax.set_ylim(80, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_yticks([80, 85, 90, 95, 100])
        ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"])
        ax.set_title(name, fontsize=12.5, color=INK2, loc="left", pad=10)
        ax.set_xlabel("items sent to Haiku", fontsize=10.5, color=MUTED, labelpad=2)
    _b = {ds: G[G.dataset == ds].sort_values("record_acc").iloc[-1] for ds in ("multinli", "negation")}
    foot(
        f,
        f"Best gate: MultiNLI {_b['multinli'].record_acc * 100:.1f}% sending {_b['multinli'].escalated * 100:.0f}% of items; "
        f"negation {_b['negation'].record_acc * 100:.1f}% sending {_b['negation'].escalated * 100:.0f}%. Thresholds chosen on the same items.",
    )
    save(f, "8_gate.png")


# 9 ── scaling: score and cost per question vs context length (reads results/summary.json)
def scaling():
    s_ = S()
    e9 = s_["E9_core"]
    unc = s_["provenance"]["jev_leaf_usd_per_question_uncached"]
    L = {"c1": 32768, "c2_131": 131072, "c2_1m": 1048576, "c2_4m": 4194304}
    series = [
        (
            "RLM + Jev sub-calls",
            BLUE,
            [(32768, "c1_jev"), (131072, "c2_jev131"), (1048576, "c2_jev1m"), (4194304, "c2_jev4m")],
        ),
        ("RLM + Haiku sub-calls", ORANGE, [(32768, "c1_llm"), (131072, "c2_llm131"), (1048576, "c2_llm1m")]),
        (
            "Sonnet 5.5 reads all",
            INK2,
            [(32768, "c1_direct"), (131072, "c2_direct131"), (1048576, "c2_direct1m_trunc")],
        ),
        (
            "Haiku 5.5 reads all",
            REF,
            [(32768, "c1_haiku_direct"), (131072, "c2_haiku_direct131"), (1048576, "c2_haiku_direct1m_trunc")],
        ),
    ]
    series = [(n, c, [(x, b) for x, b in pts if b in e9]) for n, c, pts in series]
    f = fig(
        "Score and cost as the context grows",
        "OOLONG, Claude Sonnet 5.5 root. Hollow circles: 3 questions; others: 20.",
    )
    for k, (title, key) in enumerate([("Score", "score"), ("Cost per question (list price, log scale)", "cost")]):
        ax = f.add_axes([0.08 + k * 0.48, 0.25, 0.4, 0.42])
        style(ax, "y")
        for _name, c, pts in series:
            xs, ys, ns = [], [], []
            for x, b in pts:
                r = e9[b]
                leaf = unc.get(b, r["leaf_usd"]) if "jev" in b else r["leaf_usd"]
                xs.append(x)
                ys.append(r["score"] if key == "score" else r["root_usd"] + leaf)
                ns.append(r["n"])
            ax.plot(xs, ys, color=c, lw=2, zorder=2)
            for (x, b), y, n in zip(pts, ys, ns, strict=False):
                ax.scatter(
                    [x],
                    [y],
                    s=70 if "trunc" in b else 60,
                    marker="D" if "trunc" in b else "o",
                    zorder=3,
                    color=SURFACE if n < 10 or "trunc" in b else c,
                    edgecolors=c,
                    linewidths=2,
                )
        ax.set_xscale("log")
        ax.set_xticks(list(L.values()))
        ax.set_xticklabels(["32k", "131k", "1M", "4M"])
        ax.minorticks_off()
        ax.set_xlim(22000, 6e6)
        if key == "score":
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.003, 4)
            ax.minorticks_off()
            ax.set_yticks([0.01, 0.1, 1.0])
            ax.set_yticklabels(["$0.01", "$0.10", "$1.00"])
        ax.set_title(title, fontsize=12, color=INK2, loc="left", pad=10)
        ax.set_xlabel("context length (tokens)", fontsize=10.5, color=MUTED, labelpad=2)
    f.legend(
        handles=[Line2D([], [], color=c, lw=2, marker="o", label=n) for n, c, _ in series],
        loc="upper left",
        bbox_to_anchor=(0.04, 0.84),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
    )
    foot(
        f,
        f"Diamonds: OOLONG's 1M contexts (~1.12M tokens) truncated to fit the 1M window, keeping the most recent "
        f"~{e9['c2_direct1m_trunc']['kept_share_mean']:.0%} of lines. Different question samples per length. Jev cost uncached.",
    )
    save(f, "9_scaling.png")


FIGS = [
    diagram,
    tiles,
    inline_vs_state,
    pairs,
    retrieval,
    infbench,
    calibration,
    dspy_vs_harness,
    abstention,
    gate,
    scaling,
]

if __name__ == "__main__":
    import sys

    for fn in FIGS:
        if len(sys.argv) == 1 or fn.__name__ in sys.argv[1:]:
            fn()
