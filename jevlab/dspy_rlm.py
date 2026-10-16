"""dspy.RLM (as shipped: Deno/Pyodide sandbox) with a Claude root via `claude -p` and Jev sub-calls as tools."""

from __future__ import annotations

import json
import time
from pathlib import Path

import dspy

from rlm_jev.dspy_rlm import NoteRLM

from .dspy_claude import ClaudeCLI
from .dspy_jev import inline_classify, inline_yesno


def jev_classify(items: list, labels: dict, instruction: str = "") -> list:
    """Calibrated classifier (Jev). For EACH item returns a dict {label: probability} over `labels`
    (a dict mapping label name -> short description of when it applies). Fast and cheap: 10,000 items
    take a few seconds. Use it for any per-item semantic judgment; do counting/dates/maths in Python."""
    return inline_classify([str(x) for x in items], dict(labels), instruction or None, tag="e8b_leaf")


def jev_yesno(items: list, question: str) -> list:
    """Calibrated yes/no judge (Jev). For EACH item returns P(yes) for `question` asked about that item.
    Fast and cheap: 10,000 items take a few seconds."""
    return inline_yesno([str(x) for x in items], question, tag="e8b_leaf")


API = {"root": None, "sub": None}  # set by configure(api=True): DSPy's own dspy.LM against the Anthropic API


def configure(root="sonnet", api=False, sub="haiku"):
    """Call once from the main thread (DSPy forbids configure() from worker threads).
    api=False routes calls through `claude -p` (ClaudeCLI); api=True uses dspy.LM with ANTHROPIC_API_KEY."""
    if api:
        from .models import resolve

        API["root"] = dspy.LM(f"anthropic/{resolve(root).removesuffix('[1m]')}", cache=False, max_tokens=16000)
        API["sub"] = dspy.LM(f"anthropic/{resolve(sub).removesuffix('[1m]')}", cache=False, max_tokens=16000)
        dspy.configure(lm=API["root"])
    else:
        dspy.configure(lm=ClaudeCLI(root))


def api_cost(lm, start: int) -> tuple[float, int, int]:
    """List-price USD, calls and refusals for lm.history[start:], from each call's token usage."""
    from .models import PRICES

    usd, refusals = 0.0, 0
    entries = lm.history[start:]
    for h in entries:
        r, u = h.get("response"), h.get("usage") or {}
        model = lm.model.split("/", 1)[1]
        inp = u.get("prompt_tokens", 0) or 0
        out = u.get("completion_tokens", 0) or 0
        _, pi, po, _, _ = next(t for t in PRICES[model] if t[0] is None or inp <= t[0])
        usd += (inp * pi + out * po) / 1e6
        try:
            refusals += r.choices[0].finish_reason == "content_filter"
        except Exception:
            refusals += getattr(r, "finish_reason", None) == "content_filter"
    return usd, len(entries), refusals


def make_rlm(sub="haiku", max_iters=10, max_llm_calls=20, note_field=False):
    cls = NoteRLM if note_field else dspy.RLM
    return cls(
        "context, query -> answer",
        max_iters=max_iters,
        max_llm_calls=max_llm_calls,
        tools=[jev_classify, jev_yesno],
        sub_lm=API["sub"] or ClaudeCLI(sub),
    )


ROOT = Path(__file__).resolve().parent.parent
CLAUDE_LEDGER = ROOT / "claude_ledger.jsonl"


def _ledger_len() -> int:
    return len(CLAUDE_LEDGER.read_text().splitlines()) if CLAUDE_LEDGER.exists() else 0


def run(rlm, context: str, query: str, tag: str) -> dict:
    """Runs one question and logs it. Per-run costs are read from the shared ledger, so run one question at a time."""
    t0 = time.time()
    led0 = _ledger_len()
    h0 = {k: len(v.history) for k, v in API.items() if v is not None}
    try:
        pred = rlm(context=context, query=query)
        ans, err = str(pred.answer), None
        traj = getattr(pred, "trajectory", None)
    except Exception as e:
        ans, err, traj = None, repr(e)[:300], None
    rows = (
        [json.loads(l) for l in CLAUDE_LEDGER.read_text().splitlines()[led0:]]
        if API["root"] is None and CLAUDE_LEDGER.exists()
        else []
    )
    rec = {
        "tag": tag,
        "query": query[:300],
        "answer": ans,
        "error": err,
        "wall_s": round(time.time() - t0, 1),
        "root_calls": sum(r["tag"] == "dspy_sonnet" for r in rows),
        "sub_llm_calls": sum(r["tag"] == "dspy_haiku" for r in rows),
        "root_usd_list": sum((r["usd_list"] or 0) for r in rows if r["tag"] == "dspy_sonnet"),
        "sub_usd_list": sum((r["usd_list"] or 0) for r in rows if r["tag"] == "dspy_haiku"),
        "jev_tool_used": bool(traj) and "jev_" in json.dumps(traj, default=str),
    }
    if API["root"] is not None:  # API mode: costs and call counts come from dspy.LM's own history
        (ru, rc, rr), (su, sc, sr) = api_cost(API["root"], h0["root"]), api_cost(API["sub"], h0["sub"])
        rec.update(
            root_calls=rc, sub_llm_calls=sc, root_usd_list=ru, sub_usd_list=su, refusals=rr + sr, path="anthropic_api"
        )
    with open(ROOT / "rlm_ledger.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    (ROOT / "results/rlm_transcripts").mkdir(parents=True, exist_ok=True)
    with open(ROOT / f"results/rlm_transcripts/{tag}_{int(t0)}_{abs(hash(query)) % 10**6}.json", "w") as f:
        json.dump({"query": query, "answer": ans, "trajectory": traj}, f, default=str, indent=1)
    return rec
