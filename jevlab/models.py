"""Pinned Claude models and list prices for every `claude -p` call in the experiments.

The CLI aliases ("sonnet", "haiku") drift, and the CLI does not know the prices of the newest models, so
experiments name an alias, it is resolved here, and list cost is computed from the token counts the CLI reports.
Prices: https://platform.claude.com/docs/en/about-claude/pricing (USD per million tokens, read 2026-10-12).
Cache writes are priced at the 5-minute rate, i.e. what the same calls would cost through the API with 5-minute
prompt caching. (The CLI itself writes 1-hour cache; the token split is kept in the ledgers' `usage` fields.)
"""

from __future__ import annotations

# "[1m]" asks Claude Code for the model's full 1M-token window; without it the CLI rejects prompts over ~200k tokens.
MODELS = {"sonnet": "claude-sonnet-5-5[1m]", "haiku": "claude-haiku-5-5[1m]"}

# model id -> list of (max prompt tokens, input, output, cache write 5m, cache read)
PRICES = {
    "claude-sonnet-5-5": [(None, 2.00, 10.00, 2.50, 0.10)],
    "claude-haiku-5-5": [(100_000, 0.10, 0.50, 0.125, 0.01), (None, 0.50, 2.50, 0.625, 0.05)],
}


def resolve(model: str) -> str:
    return MODELS.get(model, model)


def list_cost(cli_json: dict) -> float:
    """List-price USD for one `claude -p --output-format json` result, from its per-model token counts."""
    total = 0.0
    for mid, u in (cli_json.get("modelUsage") or {}).items():
        tiers = PRICES.get(mid.removesuffix("[1m]"))
        if tiers is None:
            raise ValueError(f"no list price for model {mid!r}; add it to PRICES in jevlab/models.py")
        inp, out = u.get("inputTokens", 0), u.get("outputTokens", 0)
        cw, cr = u.get("cacheCreationInputTokens", 0), u.get("cacheReadInputTokens", 0)
        prompt = inp + cw + cr
        _, pi, po, pw, pr = next(t for t in tiers if t[0] is None or prompt <= t[0])
        total += (inp * pi + out * po + cw * pw + cr * pr) / 1e6
    return total
