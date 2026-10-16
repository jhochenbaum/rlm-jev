"""E8h: Claude Haiku 5.5 on the 1,000 original TREC questions outside OOLONG's slice (the records E8f fits on).

Jev scores much lower on these than on OOLONG's own records; this checks whether the drop is the data or the model.
Haiku runs as in E6: through the Claude Code CLI (default extended thinking) and through the Anthropic API with
thinking off (ANTHROPIC_API_KEY, e.g. in .env.local).
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from e6_cascade import TASK
from e6b_haiku_nothink import label_api

from jevlab.claude_leaf import claude_label
from jevlab.oolong import LABELS, load_val, parse_window
from jevlab.trec import external_records, external_sample, probs


async def main():
    df = load_val().query("dataset == 'trec_coarse'")
    val = {
        x.text
        for g in df.drop_duplicates(["context_len", "context_window_id"]).itertuples()
        for x in parse_window(g.context_window_text_with_labels)
    }
    ext = external_records(val)
    texts = external_sample(ext)
    gold = [ext[t] for t in texts]
    P = await probs(texts, tag="e8f_label")  # cached by E8f
    acc = lambda labs: float(np.mean([a == g for a, g in zip(labs, gold, strict=True)]))
    out = {"n": len(texts), "jev_ensemble": acc([max(P[t], key=P[t].get) for t in texts])}
    prev = Path("results/e8h_haiku_offslice.json")
    if prev.exists() and "--rerun-haiku" not in sys.argv:  # Haiku's labels don't depend on Jev; reuse the measured runs
        old = json.loads(prev.read_text())
        out.update({k: old[k] for k in ("haiku_cli_thinking", "haiku_api_no_thinking")})
        Path("results/e8h_haiku_offslice.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    t0 = time.time()
    cli = await claude_label(texts, LABELS["trec_coarse"], model="haiku", task=TASK, tag="e8h_haiku")
    out["haiku_cli_thinking"] = {
        "acc": acc(cli),
        "parse_ok": float(np.mean([c is not None for c in cli])),
        "wall_s": round(time.time() - t0, 1),
    }
    api, usd, wall = label_api(texts, LABELS["trec_coarse"])
    out["haiku_api_no_thinking"] = {
        "acc": acc(api),
        "parse_ok": float(np.mean([c is not None for c in api])),
        "usd_list": usd,
        "wall_s": wall,
    }
    print(json.dumps(out, indent=1))
    Path("results/e8h_haiku_offslice.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
