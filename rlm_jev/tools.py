"""REPL tools for RLM harnesses. The same functions work for dspy.RLM (tools=list) and alexzhang13/rlm
(custom_tools=dict)."""

from __future__ import annotations

from ._client import Spend
from .locate import locate
from .subcalls import classify, yesno


def rlm_tools(max_usd: float | None = None, style: str = "dspy"):
    """Jev tools with docstrings written for the root model. max_usd caps Jev spend for this tool set: a call that
    would exceed it raises instead of being sent.
    style="dspy": list of functions for dspy.RLM(tools=...)
    style="rlm":  dict for alexzhang13/rlm RLM(custom_tools=...), with each docstring as the tool description"""
    if style not in ("dspy", "rlm"):
        raise ValueError(f'style must be "dspy" or "rlm", not {style!r}')
    spend = Spend(max_usd)

    def jev_classify(items: list, labels: dict, instruction: str = "") -> list:
        """Calibrated classifier (Jev). For EACH item returns {label: probability} over `labels`, a dict mapping
        label name -> short description of when it applies (or a list of label names). Batched and parallel:
        thousands of items take seconds and cost cents. Use it for per-item semantic judgments; do counting,
        dates and arithmetic in Python."""
        return classify(items, labels, instruction or "Which label best describes `item`?", spend=spend)

    def jev_yesno(items: list, question: str) -> list:
        """Calibrated yes/no judge (Jev). For EACH item returns P(yes) for `question` asked about that item."""
        return yesno(items, question, spend=spend)

    def jev_locate(docs, queries: list, k: int = 8) -> dict:
        """Semantic search over a string or list of documents of any size. Reads everything once for all
        `queries` together, then zooms into lines. Returns {query: [{"doc", "offset", "score", "text"}]}."""
        return locate(docs, queries, k=k, spend=spend)

    def jev_spend() -> str:
        """How much Jev budget has been used so far."""
        return f"${spend.usd:.4f} of {'unlimited' if max_usd is None else f'${max_usd}'}"

    fns = [jev_classify, jev_yesno, jev_locate, jev_spend]
    if style == "rlm":
        return {f.__name__: {"tool": f, "description": " ".join(f.__doc__.split())} for f in fns}
    return fns
