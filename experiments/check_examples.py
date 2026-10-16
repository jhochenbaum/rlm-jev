"""Run one of the public examples end to end on one OOLONG question and score the answer.

The root model is called through the Claude Code CLI (`claude -p`), so no Anthropic API key is needed; the examples
are unchanged, only their LM client is swapped. Jev usage is added to ledger.jsonl.
Usage: uv run python experiments/check_examples.py dspy|rlm
"""

import asyncio
import json
import sys
import time

sys.path[:0] = [".", "examples"]
if len(sys.argv) != 2 or sys.argv[1] not in ("dspy", "rlm"):
    sys.exit("usage: uv run python experiments/check_examples.py dspy|rlm")
import jevlab  # noqa: F401  (loads TYPESAFE_API_KEY from .env.local)
import rlm_jev
from jevlab.claude_cli import run_claude
from jevlab.claude_leaf import log_call
from jevlab.oolong import LABELS, load_val
from jevlab.oolong_program import extract, score

r = load_val().query("dataset == 'trec_coarse' and context_len == 32768").iloc[3]
print("Q:", r.question[:140], "| gold", r.answer)


def claude(system, prompt, model="sonnet"):
    t0 = time.time()
    reply, meta = run_claude(system, prompt, model, timeout=900)
    log_call("check_examples_root", model, meta, time.time() - t0)
    return reply


if sys.argv[1] == "dspy":
    import dspy_rlm_jev

    from jevlab.dspy_claude import ClaudeCLI

    ans = dspy_rlm_jev.build(lm=ClaudeCLI("sonnet"))(context=r.context_window_text, query=r.question).answer
else:
    import rlm.core.rlm as core
    from rlm.clients import BaseLM
    from rlm.core.types import ModelUsageSummary, UsageSummary

    class ClaudeRoot(BaseLM):
        def __init__(self, **kw):
            super().__init__(model_name="claude-cli/sonnet")
            self.calls = 0

        def completion(self, prompt, model=None):
            self.calls += 1
            msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system") or "You are a helpful assistant."
            body = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in msgs if m["role"] != "system")
            return claude(system, body)

        async def acompletion(self, prompt, model=None):
            return await asyncio.to_thread(self.completion, prompt, model)

        def get_usage_summary(self):
            return UsageSummary({self.model_name: ModelUsageSummary(self.calls, 0, 0)})

        def get_last_usage(self):
            return ModelUsageSummary(1, 0, 0)

    core.get_client = lambda *a, **k: ClaudeRoot()
    import alexzhang_rlm_jev

    ans = alexzhang_rlm_jev.build().completion(r.context_window_text, root_prompt=r.question).response

st = rlm_jev.stats()
print("JEV USAGE:", st)
with open("ledger.jsonl", "a") as f:
    f.write(
        json.dumps(
            {
                "t": time.time(),
                "tag": f"local_examples_{sys.argv[1]}",
                "input_tokens": st["input_tokens"],
                "output_tokens": 0,
                "questions": 0,
                "latency_s": 0,
                "usd": st["usd"],
            }
        )
        + "\n"
    )
print("ANSWER:", ans)
print("SCORE:", score(extract(ans, r.answer_type, list(LABELS["trec_coarse"])), r.answer, r.answer_type))
