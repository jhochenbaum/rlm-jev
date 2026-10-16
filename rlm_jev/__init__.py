"""rlm_jev: Jev (System One) sub-calls for Recursive Language Models.

Drop-in tools for any RLM-style harness (dspy.RLM, alexzhang13/rlm, your own REPL):

    from rlm_jev import classify, yesno, locate
    probs  = classify(items, {"spam": "unsolicited ads or scams", "ham": "a normal message"})
    p_yes  = yesno(items, "Does this review mention shipping?")
    hits   = locate(documents, ["who founded the company?", "when was it acquired?"])

Two rules this library follows, both measured (see the write-up):
  1. Each item goes inside its own question; the shared state holds only common context. Never make Jev find an
     item by position (asking about `records[i]` in an unlabeled list drops accuracy from 94% to 34% at 100 items).
  2. Many questions share one state: a corpus is read once for every query at the same time.

Pass `spend=Spend(max_usd=...)` to any call to track and cap Jev spend.
"""

from ._client import Spend, stats
from .calibrate import apply_weights, fit_weights, mc_confidence
from .locate import locate
from .subcalls import classify, yesno
from .tools import rlm_tools

__all__ = [
    "Spend",
    "apply_weights",
    "classify",
    "fit_weights",
    "locate",
    "mc_confidence",
    "rlm_tools",
    "stats",
    "yesno",
]
