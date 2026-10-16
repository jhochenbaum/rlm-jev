"""The one place the experiments call Claude: the Claude Code CLI in headless mode (`claude -p`).

Calls run from a neutral temp directory (no project CLAUDE.md or settings), with built-in tools and MCP servers off,
and without ANTHROPIC_API_KEY in the environment, so they use the CLI's own login. Models are pinned in models.py and
list cost is computed from the reported token counts.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .models import list_cost, resolve

SCRATCH = Path(tempfile.gettempdir()) / "jevlab_claude_cwd"
SCRATCH.mkdir(exist_ok=True)
TOO_LONG = "Prompt is too long"


def run_claude(system: str, prompt: str, model: str, timeout: int = 600) -> tuple[str, dict]:
    """Returns (reply text, {"usd_list", "usage", "models"}). Raises RuntimeError if the CLI fails or reports an
    error, except a prompt over the model's window, which comes back as a reply starting with "Prompt is too long"."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    # A file, not a pipe: a large prompt has to be fully available before the CLI's stdin deadline.
    with tempfile.TemporaryFile() as fin:
        fin.write(prompt.encode())
        fin.seek(0)
        p = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                resolve(model),
                "--output-format",
                "json",
                "--system-prompt",
                system,
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
            ],
            stdin=fin,
            capture_output=True,
            cwd=SCRATCH,
            timeout=timeout,
            env=env,
        )
    try:
        j = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"claude -p failed (exit {p.returncode}): {(p.stderr or p.stdout).decode(errors='replace')[:300]}"
        ) from None
    reply = j.get("result") or ""
    if (j.get("is_error") or p.returncode) and not reply.startswith(TOO_LONG):
        raise RuntimeError(f"claude -p returned an error: {reply[:300]}")
    return reply, {"usd_list": list_cost(j), "usage": j.get("usage", {}), "models": list(j.get("modelUsage") or {})}
