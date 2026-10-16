"""Claude as the labeling model, through jevlab.claude_cli.

Used for the sub-call swap (LLM labels vs Jev labels on the same records) and the escalation cascade (low-confidence
Jev records go to Claude). Every call's list-price cost is logged to claude_ledger.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

from .claude_cli import run_claude
from .models import resolve

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "claude_ledger.jsonl"
_LEDGER_LOCK = threading.Lock()
SYSTEM = "You are a precise data-labeling function. Output only the requested JSON, nothing else."


def log_call(tag: str, model: str, meta: dict, latency_s: float) -> None:
    row = {
        "t": time.time(),
        "tag": tag,
        "model": meta["models"] or resolve(model),
        "usd_list": meta["usd_list"],
        "latency_s": round(latency_s, 2),
        "usage": meta["usage"],
    }
    with _LEDGER_LOCK, LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def claude_call(
    prompt: str,
    model: str = "haiku",
    tag: str = "claude",
    system: str = SYSTEM,
    timeout: int = 600,
) -> str:
    t0 = time.time()
    reply, meta = await asyncio.to_thread(run_claude, system, prompt, model, timeout)
    log_call(tag, model, meta, time.time() - t0)
    return reply


def label_prompt(chunk: list[str], options: dict[str, str], task: str = "") -> str:
    """The batched labeling prompt: records listed by index, a JSON object of labels back."""
    opt_txt = "\n".join(f"- {k}: {v}" for k, v in options.items())
    recs = "\n".join(f"{i}\t{t}" for i, t in enumerate(chunk))
    return (
        f"{task}\nLabel each record with exactly one of these labels:\n{opt_txt}\n\n"
        f"Records (index<TAB>text):\n{recs}\n\n"
        f"Return a JSON object mapping each index (as a string) to its label, for all {len(chunk)} records."
    )


def parse_labels(reply: str) -> dict | None:
    """The JSON object in a reply, or None if there isn't a parseable one."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        return json.loads(m[0])
    except json.JSONDecodeError:
        return None


def read_labels(labels: dict, n: int, options: dict[str, str]) -> list[str | None]:
    out = [labels.get(str(i)) for i in range(n)]
    return [lab if isinstance(lab, str) and lab in options else None for lab in out]


async def claude_label(
    texts: list[str],
    options: dict[str, str],
    model: str = "haiku",
    batch: int = 100,
    concurrency: int = 4,
    tag: str = "claude_label",
    task: str = "",
) -> list[str | None]:
    sem = asyncio.Semaphore(concurrency)
    out: list[str | None] = [None] * len(texts)

    async def run(lo: int):
        chunk = texts[lo : lo + batch]
        prompt = label_prompt(chunk, options, task)
        labels = {}
        for _ in range(2):  # one retry if the reply isn't parseable JSON
            async with sem:
                res = await claude_call(prompt, model=model, tag=tag)
            labels = parse_labels(res)
            if labels is not None:
                break
        for i, lab in enumerate(read_labels(labels or {}, len(chunk), options)):
            out[lo + i] = lab

    await asyncio.gather(*(run(lo) for lo in range(0, len(texts), batch)))
    return out
