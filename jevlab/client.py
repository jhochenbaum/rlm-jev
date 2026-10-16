"""Shared Jev client with a spend ledger and a hard budget cap.

Every call is logged to ledger.jsonl (tokens + $), and calls are refused once
the cumulative spend would exceed BUDGET_USD. The API key is TYPESAFE_API_KEY, from the environment or .env.local
(loaded by jevlab/__init__.py), and is never printed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from rlm_jev._client import PRICE_PER_TOKEN

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger.jsonl"
BUDGET_USD = float(os.environ.get("JEV_BUDGET_USD", "20"))  # hard cap on cumulative Jev spend in ledger.jsonl

_lock = threading.Lock()


def _api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (export it or put it in .env.local)")
    return key


_SPENT: list[float] = []  # cumulative ledger total, read once, then kept in memory


def spent() -> float:
    """Cumulative Jev spend recorded in ledger.jsonl: read from disk once per process, then kept up to date in memory
    (so two experiments running at the same time don't see each other's spend)."""
    with _lock:
        if not _SPENT:
            total = 0.0
            if LEDGER.exists():
                for line in LEDGER.read_text().splitlines():
                    if line.strip():
                        total += json.loads(line)["usd"]
            _SPENT.append(total)
        return _SPENT[0]


def _record(tag: str, usage, n_questions: int, latency_s: float) -> None:
    tokens = getattr(usage, "input_tokens", 0) or 0
    row = {
        "t": time.time(),
        "tag": tag,
        "input_tokens": tokens,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "questions": n_questions,
        "latency_s": round(latency_s, 3),
        "usd": tokens * PRICE_PER_TOKEN,
    }
    spent()  # make sure the in-memory total is initialised before adding to it
    with _lock, LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
        _SPENT[0] += row["usd"]


def _check_budget() -> None:
    s = spent()
    if s >= BUDGET_USD:
        raise RuntimeError(f"Budget cap hit: ${s:.4f} >= ${BUDGET_USD}")


class Jev:
    """Async Jev client that logs every request and enforces the budget cap. Close it when done
    (`async with Jev() as jev:` or `await jev.aclose()`)."""

    def __init__(self, model: str = "jev-latest", retries: int = 4):
        self.model = model
        # the SDK retries rate limits, server errors, timeouts and connection errors, with backoff
        policy = RetryPolicy(max_retries=retries, timeout=120.0)
        self.aio = AsyncTypeSafeClient(api_key=_api_key(), model=model, retry=policy)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self.aio.aclose()

    async def aask(self, state, questions: dict, tag: str = "misc"):
        _check_budget()
        t0 = time.time()
        r = await self.aio.system_one(state=state, questions=questions)
        _record(tag, r.usage, len(questions), time.time() - t0)
        return r


def summary() -> str:
    if not LEDGER.exists():
        return "no calls yet"
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    by_tag: dict[str, list] = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    lines = [f"TOTAL ${sum(r['usd'] for r in rows):.4f} over {len(rows)} calls"]
    for tag, rs in sorted(by_tag.items()):
        lines.append(
            f"  {tag:28s} calls={len(rs):5d} tok={sum(r['input_tokens'] for r in rs):>10,d} "
            f"${sum(r['usd'] for r in rs):.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
