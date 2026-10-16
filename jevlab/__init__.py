"""Experiment harness for rlm-jev. Importing it sets TYPESAFE_API_KEY from .env.local if it isn't already set.

ANTHROPIC_API_KEY is deliberately not loaded here: Claude calls go through the Claude Code CLI (jevlab.claude_cli),
which uses its own login. Only the API-path dspy.RLM run (E8B_API=1) reads it, from the environment or .env.local.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

_ENV = Path(__file__).resolve().parent.parent / ".env.local"
if "TYPESAFE_API_KEY" not in os.environ and _ENV.exists():
    _key = dotenv_values(_ENV).get("TYPESAFE_API_KEY")
    if _key:
        os.environ["TYPESAFE_API_KEY"] = _key
