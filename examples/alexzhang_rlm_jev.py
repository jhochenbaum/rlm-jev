"""alexzhang13/rlm (the reference RLM implementation) with Jev sub-calls via custom_tools.

pip install "rlm-jev[rlm] @ git+https://github.com/jhochenbaum/rlm-jev"
export TYPESAFE_API_KEY=...  ANTHROPIC_API_KEY=...
"""

import os
import sys
from pathlib import Path

from rlm import RLM

from rlm_jev import rlm_tools


def build(backend: str = "anthropic", model: str = "claude-sonnet-5-5", max_jev_usd: float = 0.50) -> RLM:
    # rlm's Anthropic backend takes the key explicitly; it doesn't read ANTHROPIC_API_KEY on its own.
    return RLM(
        backend=backend,
        backend_kwargs={"model_name": model, "api_key": os.environ.get("ANTHROPIC_API_KEY")},
        custom_tools=rlm_tools(max_usd=max_jev_usd, style="rlm"),
        max_iterations=10,
    )


if __name__ == "__main__":
    context = Path(sys.argv[1]).read_text()
    query = sys.argv[2]
    print(build().completion(context, root_prompt=query).response)
