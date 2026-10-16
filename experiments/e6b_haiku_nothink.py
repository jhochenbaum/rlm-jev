"""E6b: the E6 sub-call swap with Claude Haiku 5.5's extended thinking turned off.

Same 500 TREC records, labels, prompt and batching as E6's Haiku arm (read from results/e6_labels.json). The Claude
Code CLI can't turn Haiku 5.5's thinking off, so this arm calls the Anthropic API (ANTHROPIC_API_KEY, e.g. in
.env.local) with thinking disabled. Answers whether Jev's cost advantage over Haiku comes from Haiku's thinking tokens.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
from dotenv import dotenv_values

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from e6_cascade import TASK

from jevlab.claude_leaf import SYSTEM, label_prompt, parse_labels, read_labels
from jevlab.models import resolve
from jevlab.oolong import LABELS

BATCH, CONCURRENCY = 100, 4  # as in E6's Haiku arm
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6  # Haiku 5.5 list price per token, prompts under 100k tokens


def label_api(texts: list[str], options: dict[str, str]) -> tuple[list[str | None], float, float]:
    """Haiku 5.5 labels through the Anthropic API with thinking off: (labels, list-price USD, wall seconds)."""
    key = os.environ.get("ANTHROPIC_API_KEY") or dotenv_values(".env.local").get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    model = resolve("haiku").removesuffix("[1m]")

    def one(lo):
        chunk = texts[lo : lo + BATCH]
        r = client.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": label_prompt(chunk, options, TASK)}],
        )
        reply = "".join(b.text for b in r.content if b.type == "text")
        return read_labels(parse_labels(reply) or {}, len(chunk), options), r.usage

    t0 = time.time()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        res = list(ex.map(one, range(0, len(texts), BATCH)))
    tin = sum(u.input_tokens + (u.cache_creation_input_tokens or 0) for _, u in res)
    tout = sum(u.output_tokens for _, u in res)
    return (
        [lab for labs, _ in res for lab in labs],
        round(tin * PRICE_IN + tout * PRICE_OUT, 5),
        round(time.time() - t0, 1),
    )


def main():
    swap = json.loads(Path("results/e6_labels.json").read_text())["swap"]
    texts, gold = swap["texts"], swap["gold"]
    labels, usd, wall = label_api(texts, LABELS["trec_coarse"])
    out = {
        "n": len(texts),
        "model": resolve("haiku").removesuffix("[1m]"),
        "path": "anthropic_api, thinking disabled",
        "acc": float(np.mean([a == g for a, g in zip(labels, gold, strict=True)])),
        "parse_ok": float(np.mean([a is not None for a in labels])),
        "usd_list": usd,
        "wall_s": wall,
        "labels": labels,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "labels"}, indent=1))
    Path("results/e6b_haiku_nothink.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
