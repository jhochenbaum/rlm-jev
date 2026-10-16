"""DSPy-native Jev sub-calls: dspy.Predict with Choice/Noul outputs on dspy.experimental.TypeSafe.

Two leaf implementations with the same signature, so dspy.RLM can use either as a tool:
  native  : one dspy.Predict call (= one Jev request) per item, parallelised with Predict.batch
  inline  : our batched label_texts (items inlined into each question, 200 per request)
Every native request's token usage is copied from the TypeSafe client's history into
ledger.jsonl so the budget cap and spend accounting stay complete.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import dspy
from dspy.experimental import Choice, TypeSafe

from .client import _check_budget, _record
from .label import label_texts, yesno_texts

_KEY = os.environ.get("TYPESAFE_API_KEY", "")


def typesafe_lm() -> TypeSafe:
    return TypeSafe("jev-latest", api_key=_KEY, cache=False, timeout=30)


def _log_history(lm, start: int, tag: str):
    """Log the Jev calls DSPy made (lm.history[start:]) to the ledger and the in-memory budget total."""
    for h in lm.history[start:]:
        tok = (h.get("usage") or {}).get("prompt_tokens", 0)
        _record(tag, SimpleNamespace(input_tokens=tok, output_tokens=0), 1, 0.0)


def native_classify(
    items: list[str],
    labels: dict[str, str],
    instruction: str | None = None,
    tag: str = "dspy_native",
    threads: int = 24,
) -> list[dict[str, float]]:
    """One Jev request per item via dspy.Predict(item -> label: Choice[...])."""
    _check_budget()
    choice_t = Choice[tuple((k, v) for k, v in labels.items())]
    sig = dspy.Signature(
        {
            "item": (str, dspy.InputField()),
            "label": (choice_t, dspy.OutputField(desc=instruction or "Which label best describes `item`?")),
        },
        "Classify the item.",
    )
    lm = typesafe_lm()
    pred = dspy.Predict(sig)
    pred.set_lm(lm)
    start = len(lm.history)
    with dspy.context(max_history_size=10**6):
        outs = pred.batch([dspy.Example(item=t).with_inputs("item") for t in items], num_threads=threads)
    _log_history(lm, start, tag)
    res = []
    for o in outs:
        c = o.label
        res.append(dict(c.probabilities) if c.probabilities else {str(c.value): 1.0})
    return res


def inline_classify(
    items: list[str],
    labels: dict[str, str],
    instruction: str | None = None,
    tag: str = "dspy_inline",
    use_cache: bool = True,
) -> list[dict[str, float]]:
    """Batched: each item inlined into its own Choice question, 200 per request."""
    kw = {"inline_instruction": instruction} if instruction else {}
    return asyncio.run(
        label_texts(
            [str(x) for x in items], dict(labels), batch=200, concurrency=24, tag=tag, use_cache=use_cache, **kw
        )
    )


def inline_yesno(items: list[str], question: str, tag: str = "dspy_inline") -> list[float]:
    """Batched: each item inlined into its own yes/no question, 200 per request."""
    return asyncio.run(yesno_texts([str(x) for x in items], question, tag=tag))
