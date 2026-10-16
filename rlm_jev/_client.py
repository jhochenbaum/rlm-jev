"""Shared async plumbing: one client per call (safe across repeated asyncio.run), bounded concurrency,
retries on transient errors, request sizing, and optional spend caps."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

PRICE_PER_TOKEN = 0.042 / 1_000_000  # Jev input price (jev-1.13); output tokens are free
MAX_REQUEST_TOKENS = 48_000  # stay well under Jev's ~64k-token request limit
TOKENS_PER_QUESTION = 40  # structure overhead per question
TOKENS_PER_REQUEST = 200  # structure overhead per request

_STATS_LOCK = threading.Lock()
STATS = {"requests": 0, "input_tokens": 0}  # process-wide usage, see rlm_jev.stats()


def stats() -> dict:
    """Process-wide Jev usage so far: requests, input tokens and estimated dollars."""
    with _STATS_LOCK:
        return {**STATS, "usd": round(STATS["input_tokens"] * PRICE_PER_TOKEN, 6)}


def estimate_tokens(*texts: str) -> int:
    """Conservative token count for text sent to Jev: about 3 bytes of UTF-8 per token, so English runs ~3 characters
    per token and CJK about one. With the overheads above it matches Jev's reported usage on English within a few
    percent (e.g. ~230 tokens per TREC record with six labels)."""
    return sum(len(t.encode()) for t in texts) // 3 + 1


class Spend:
    """Spend tracker and optional hard cap, shared by every call that receives it.

    Each call reserves its estimated cost before any request is sent and is refused if that would exceed `max_usd`,
    so concurrent calls cannot overshoot the cap by more than the estimation error."""

    def __init__(self, max_usd: float | None = None):
        self.max_usd, self.usd, self.tokens = max_usd, 0.0, 0
        self._reserved, self._lock = 0.0, threading.Lock()

    def reserve(self, est_tokens: int) -> float:
        est = est_tokens * PRICE_PER_TOKEN
        with self._lock:
            if self.max_usd is not None and self.usd + self._reserved + est > self.max_usd:
                raise RuntimeError(
                    f"Jev budget cap reached: ${self.usd:.4f} spent, ${self._reserved:.4f} in flight, "
                    f"next request ~${est:.4f}, cap ${self.max_usd}"
                )
            self._reserved += est
        return est

    def settle(self, reserved: float, tokens: int):
        with self._lock:
            self._reserved -= reserved
            self.tokens += tokens
            self.usd += tokens * PRICE_PER_TOKEN


def run(coro):
    """Run a coroutine from sync code, even inside an already-running event loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as ex:  # a running loop can't be re-entered; run on a fresh loop in a worker thread
        return ex.submit(asyncio.run, coro).result()


async def ask_many(
    requests, concurrency: int = 24, model: str | None = None, spend: Spend | None = None, retries: int = 4
):
    """requests: list of (state, questions, estimated_tokens). Returns responses in order.

    The whole call's estimate is reserved against `spend` first, so a call that would exceed the cap raises before
    anything is sent. The SDK retries rate limits, server errors, timeouts and connection errors (`retries` times,
    with backoff). Any other error is raised once the remaining requests finish; failed requests are counted at
    their estimate, since they may have been billed."""
    requests = list(requests)
    est_total = sum(e for _, _, e in requests)
    reserved = spend.reserve(est_total) if spend else 0.0
    sem = asyncio.Semaphore(concurrency)
    policy = RetryPolicy(max_retries=retries, timeout=120.0)
    sent = False
    try:
        async with AsyncTypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY"), model=model, retry=policy) as client:

            async def one(state, qs):
                async with sem:
                    return await client.system_one(state=state, questions=qs)

            sent = True
            results = await asyncio.gather(*(one(s, q) for s, q, _ in requests), return_exceptions=True)
    except BaseException:  # setup failed (nothing sent) or cancelled mid-call (count every request at its estimate)
        if spend:
            spend.settle(reserved, est_total if sent else 0)
        raise
    used = [
        r.usage.input_tokens or e
        for r, (_, _, e) in zip(results, requests, strict=True)
        if not isinstance(r, BaseException)
    ]
    with _STATS_LOCK:
        STATS["requests"] += len(used)
        STATS["input_tokens"] += sum(used)
    if spend:
        failed = sum(e for r, (_, _, e) in zip(results, requests, strict=True) if isinstance(r, BaseException))
        spend.settle(reserved, sum(used) + failed)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


def pack(
    items: list[str], fixed_tokens: int, per_question_tokens: int, max_items: int, max_tokens: int = MAX_REQUEST_TOKENS
) -> list[list[int]]:
    """Group item indices into requests of at most `max_items` items and `max_tokens` estimated tokens.
    fixed_tokens: the state; per_question_tokens: what every question repeats (instruction, label options)."""
    fixed_tokens += TOKENS_PER_REQUEST
    if fixed_tokens + per_question_tokens > max_tokens:
        raise ValueError(
            f"the shared context and question text are ~{fixed_tokens + per_question_tokens:,} tokens, too large for "
            f"one Jev request (limit ~{max_tokens:,}); shorten the context"
        )
    batches, cur, size = [], [], fixed_tokens
    for i, it in enumerate(items):
        t = estimate_tokens(it) + per_question_tokens
        if fixed_tokens + t > max_tokens:
            raise ValueError(
                f"item {i} is ~{t:,} tokens, too large for one Jev request "
                f"(limit ~{max_tokens:,}); split it into smaller pieces first"
            )
        if cur and (len(cur) >= max_items or size + t > max_tokens):
            batches.append(cur)
            cur, size = [], fixed_tokens
        cur.append(i)
        size += t
    if cur:
        batches.append(cur)
    return batches
