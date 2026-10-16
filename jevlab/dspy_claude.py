"""DSPy LM adapter that calls Claude through the Claude Code CLI (jevlab.claude_cli).

Uses DSPy 3.4's BaseLM.forward() interface (deprecated in favour of engines; removed in DSPy 3.5). Messages are
flattened into a system prompt plus one user turn. Each call's list-price cost is logged to claude_ledger.jsonl
with tag "dspy_<model>".
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import dspy

from .claude_cli import run_claude
from .claude_leaf import log_call


class ClaudeCLI(dspy.BaseLM):
    def __init__(self, model: str = "sonnet", timeout: int = 900, **kwargs):
        super().__init__(model=f"claude-cli/{model}", model_type="chat", cache=False, **kwargs)
        self.cli_model = model
        self.timeout = timeout

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        system = (
            "\n\n".join(_text(m["content"]) for m in messages if m["role"] == "system")
            or "You are a helpful assistant."
        )
        turns = [m for m in messages if m["role"] != "system"]
        if not turns:
            raise ValueError("ClaudeCLI needs at least one non-system message")
        body = (
            "\n\n".join(f"[{m['role'].upper()}]\n{_text(m['content'])}" for m in turns)
            if len(turns) > 1
            else _text(turns[0]["content"])
        )
        reply = self._call(system, body)
        # DSPy's ChatAdapter needs "[[ ## field ## ]]" markers; Claude sometimes answers with a bare code block.
        # Re-ask once with a format reminder (same spirit as DSPy's own adapter fallback).
        if "[[ ## " in system and "[[ ## " not in reply.choices[0].message.content:
            reminder = (
                "\n\n[SYSTEM]\nYour previous reply did not follow the required output format. Reply again, "
                "starting each output field with its `[[ ## field_name ## ]]` header exactly as specified, "
                "and end with `[[ ## completed ## ]]`."
            )
            reply = self._call(system, body + reminder)
        return reply

    def _call(self, system, body):
        t0 = time.time()
        reply, meta = run_claude(system, body, self.cli_model, timeout=self.timeout)
        log_call(f"dspy_{self.cli_model}", self.cli_model, meta, time.time() - t0)
        u = meta["usage"]
        usage = {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0)}
        return SimpleNamespace(
            model=self.model,
            usage=usage,
            cache_hit=False,
            _hidden_params={"response_cost": meta["usd_list"]},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=reply, tool_calls=None, reasoning_content=None, provider_specific_fields={}
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
        )


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
