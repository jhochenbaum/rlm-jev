# Reproducing the results

Every number in the write-up is in [`results/summary.json`](results/summary.json), built from the raw files in
`results/` by `experiments/summarize.py`. You can rebuild the summary and figures from the committed results for free,
or re-run any experiment.

## 1. Setup

```bash
uv sync --all-extras --all-groups               # Python 3.11 + locked dependencies (uv-managed)
echo "TYPESAFE_API_KEY=..." > .env.local          # Jev key; add ANTHROPIC_API_KEY=... for the API-path dspy.RLM run (E8b)
uv run python experiments/setup_data.py          # all datasets into data/ (about 6 GB); or name steps, e.g. `oolong pairs`
```

You also need:

- The Claude Code CLI (`claude`), logged in. Claude Sonnet 5.5 (roots, direct-read baselines, BrowseComp-Plus judge)
  and Claude Haiku 5.5 (LLM sub-call baseline) are called through `claude -p`, pinned in `jevlab/models.py` with the
  1M-token window enabled. Every call goes through `jevlab/claude_cli.py::run_claude`; swap that function for an
  API client if you prefer.
- [Deno](https://deno.land), for `dspy.RLM`'s sandbox (E8b).

Spend is logged per call to local ledgers (`ledger.jsonl` for Jev; `claude_ledger.jsonl` and `rlm_ledger.jsonl` for
Claude at list price, computed from token counts). Jev spend is hard-capped at $20 by default (`JEV_BUDGET_USD`).
Jev labels are cached in `data/label_cache.sqlite`, keyed by the text and every request setting (labels, option
order, batch size, context, model), so later experiments that label the same texts the same way cost nothing.

## 2. Rebuild the summary and figures (free)

```bash
uv run python experiments/summarize.py
uv run python figures/make_figures.py
```

## 3. Re-run the experiments

Run them in this order: later steps reuse earlier labels and ledger entries. Costs are rough single-run estimates,
with Claude at list price.

```bash
# Batching layout (E0, E0b). Jev ~$0.15
uv run python experiments/e0_pilot.py
uv run python experiments/e0b_keyed_state.py

# OOLONG with the fixed program, and checks that reuse its labels (E1, E1b, E3, E4, E10). Jev ~$0.35
uv run python experiments/e1_validate_program.py        # scores 1.0 on gold labels (no API calls)
uv run python experiments/e1_oolong.py
uv run python experiments/e1b_test_zeroshot.py
uv run python experiments/e3_pairs.py
uv run python experiments/e4_counting.py
uv run python experiments/e10_abstain.py

# Finding passages: locate() on 30 hidden facts (E5). Jev ~$0.30
uv run python experiments/e5_locate.py
uv run python experiments/e5b_library_locate.py         # the shipped rlm_jev.locate() on the same facts
uv run python experiments/e5c_modern_baselines.py 0.6b  # Qwen3 baselines, local GPU/MPS; 4b and 8b below
uv run python experiments/e5c_modern_baselines.py Qwen/Qwen3-Embedding-0.6B tomaarsen/Qwen3-Reranker-4B-seq-cls

# Sub-call model comparison and escalation (E6, E11). Jev <$0.05, Claude ~$0.35
uv run python experiments/e6_cascade.py
uv run python experiments/e6b_haiku_nothink.py          # Haiku with thinking off, via the Anthropic API
uv run python experiments/e6c_shared_labels.py          # three Jev request formats on the same items
uv run python experiments/e6d_yesno_format.py           # the same comparison for yes/no questions
uv run python experiments/e11_gate.py                   # reuses results/e11_haiku_labels.json

# InfiniteBench (E7, then E7b for the books too long for the window). Jev ~$0.45, Claude ~$5
uv run python experiments/e7_infbench.py
uv run python experiments/e7b_truncated.py

# Live RLM runs on OOLONG (E2, then E9, which reuses E2's runs). Jev ~$6, Claude ~$90
uv run python experiments/e2_rlm_root.py
uv run python experiments/e2_score.py
for b in c1_direct c1_haiku_direct c1_jev c1_llm c2_jev131 c2_direct131 c2_haiku_direct131 c2_llm131 c2_llm1m \
         c2_jev1m c2_jev4m c2_direct1m_trunc c2_haiku_direct1m_trunc; do
  uv run python experiments/e9_core.py $b
done
uv run python experiments/e9_core.py score
uv run python experiments/e9b_truncation_ceiling.py     # reads E9's truncated runs (no API calls)

# DSPy and label weights (E8a-E8h). Jev ~$1.5, Claude ~$2 via the API
uv run python experiments/e8a_dspy_leaf.py
uv run python experiments/e8b_dspy_rlm.py               # stock dspy.RLM via the CLI (refused)
E8B_API=1 E8B_NOTE=1 E8B_WORKERS=1 uv run python experiments/e8b_dspy_rlm.py
uv run python experiments/e8c_reanchor.py
uv run python experiments/e8d_reanchor_transfer.py
uv run python experiments/e8f_disjoint_weights.py
uv run python experiments/e8g_leakfree.py
uv run python experiments/e8g_leakfree.py --labels-in-state  # the same fit with the cheaper label format
uv run python experiments/e8h_haiku_offslice.py         # Haiku on E8f's out-of-slice records

# BrowseComp-Plus: 14 and 30 root turns, no-Jev ablation, closed-book check (E12). Jev ~$1.5, Claude ~$5 a setting
uv run python experiments/e12_bcp.py run 20 3 && uv run python experiments/e12_bcp.py score
BCP_TURNS=30 uv run python experiments/e12_bcp.py run 20 3 && BCP_TURNS=30 uv run python experiments/e12_bcp.py score
BCP_TURNS=30 BCP_NOJEV=1 uv run python experiments/e12_bcp.py run 20 3 && BCP_TURNS=30 BCP_NOJEV=1 uv run python experiments/e12_bcp.py score
uv run python experiments/e12_bcp.py closedbook

# Then rebuild the summary and figures (step 2).
```

Re-runs that call Jev or Claude won't match the published numbers exactly. Jev changes a few labels in a thousand
between runs, so accuracies on hundreds of items should land within about half a point; scores on 20 RLM questions,
and the fitted label weights (E8d-E8g), move more.

`E8B_NOTE=1` runs `dspy.RLM` with its step-by-step `reasoning` field replaced by a one-sentence note
(`rlm_jev.dspy_rlm.NoteRLM`): Claude Sonnet 5.5 refuses stock `dspy.RLM`'s prompt (`reasoning_extraction`, see
[stanfordnlp/dspy#10641](https://github.com/stanfordnlp/dspy/issues/10641)). Without `E8B_NOTE`, the script runs
stock `dspy.RLM`. DSPy's fix for the refusal ([#10643](https://github.com/stanfordnlp/dspy/pull/10643)) was tested
before it was released:

```bash
E8B_API=1 E8B_PFX=e8b_api_pr10643 uv run --with "dspy @ git+https://github.com/stanfordnlp/dspy@9e3e3e0c59fe207a96b3321e1a46ad0969ab793e" \
  python experiments/e8b_dspy_rlm.py     # stock dspy.RLM on the fix: results/e8b_dspy_rlm_api_pr10643.csv
```

BrowseComp-Plus asks that its questions never appear online in plain text. They are decrypted locally at run time,
and `results/e12_bcp*.jsonl` and the e12 transcripts are gitignored.

Library checks: `uv run pytest` (offline, with a fake Jev client; CI runs it on every push),
`uv run python experiments/test_rlm_jev.py` (against the real Jev, a few cents) and
`uv run python experiments/check_examples.py dspy|rlm` (each example end to end).

### Larger retrieval baselines

`e5c_modern_baselines.py` needs only public models, the novels and `experiments/assets/needles.json`. The 8B models
need about 16 GB of memory each in fp16:

```bash
uv run python experiments/setup_data.py books
uv run python experiments/e5c_modern_baselines.py 8b    # Qwen3-Embedding-8B + Qwen3-Reranker-8B
uv run python experiments/e5c_modern_baselines.py 4b    # Qwen3-Embedding-4B + Qwen3-Reranker-4B
```

Each run writes `results/e5c_<model>[_<reranker>].json`; then rebuild the summary.
