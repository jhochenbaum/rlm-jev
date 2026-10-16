"""dspy.RLM with Jev answering the sub-calls.

pip install "rlm-jev[dspy] @ git+https://github.com/jhochenbaum/rlm-jev"      # dspy.RLM's sandbox also needs Deno: https://deno.land
export TYPESAFE_API_KEY=...  ANTHROPIC_API_KEY=...   (or any model DSPy supports)

The root LM writes Python over the long context; Jev answers the per-item semantic questions in bulk.
NoteRLM is dspy.RLM with one change so it runs on Claude Sonnet 5.5 (see rlm_jev/dspy_rlm.py).
"""

import sys
from pathlib import Path

import dspy

from rlm_jev import rlm_tools
from rlm_jev.dspy_rlm import NoteRLM


def build(lm: dspy.LM | None = None, max_jev_usd: float = 0.50) -> NoteRLM:
    dspy.configure(lm=lm or dspy.LM("anthropic/claude-sonnet-5-5"))
    return NoteRLM("context, query -> answer", tools=rlm_tools(max_usd=max_jev_usd), max_iters=10)


if __name__ == "__main__":
    context = Path(sys.argv[1]).read_text()
    query = sys.argv[2]
    print(build()(context=context, query=query).answer)
