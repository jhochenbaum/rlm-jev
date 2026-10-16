"""Minimal Recursive Language Model harness with Jev (System One) sub-calls.

The root LM (Claude, through jevlab.claude_cli) sees only the query and metadata about a `context` string held in a
Python REPL. It writes ```python``` blocks; they run in a persistent namespace and the root sees their (truncated)
output. Sub-call tools:

  jev_classify(items, labels, instruction=None) -> list[dict[label, prob]]
  jev_yesno(items, question)                     -> list[float]
  llm_query(prompt)                              -> str   (the generative sub-call baseline)

The root finishes by calling FINAL(answer).

WARNING: model-written code runs in this Python process with no sandbox: it can read and write files, use the
network and read environment variables (including API keys). Run untrusted contexts (e.g. web documents, which can
carry prompt injections) in a container or VM.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from rlm_jev.subcalls import _as_labels, _as_list

from .claude_cli import TOO_LONG, run_claude
from .label import label_texts, yesno_texts
from .models import resolve

ROOT = Path(__file__).resolve().parent.parent


class _ThreadStdout(io.TextIOBase):
    """sys.stdout that routes writes to a per-thread buffer while that thread runs REPL code, so RLM runs in
    parallel threads each see only their own output (contextlib.redirect_stdout would swap the process-wide one)."""

    def __init__(self, real):
        self.real, self.local = real, threading.local()

    def write(self, s):
        return (getattr(self.local, "buf", None) or self.real).write(s)

    def flush(self):
        (getattr(self.local, "buf", None) or self.real).flush()

    def __getattr__(self, name):  # encoding, isatty, fileno, ... come from the real stream
        return getattr(self.real, name)

    @contextlib.contextmanager
    def capture(self, buf):
        self.local.buf = buf
        try:
            yield
        finally:
            self.local.buf = None


if not isinstance(sys.stdout, _ThreadStdout):
    sys.stdout = _ThreadStdout(sys.stdout)
_STDOUT = sys.stdout
RLM_LEDGER = ROOT / "rlm_ledger.jsonl"
_LEDGER_LOCK = threading.Lock()
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _add_usage(total: dict, usage: dict) -> None:
    """Accumulate token usage, including the 1-hour / 5-minute split of cache writes."""
    for k in USAGE_KEYS:
        total[k] = total.get(k, 0) + (usage.get(k) or 0)
    for k, v in (usage.get("cache_creation") or {}).items():
        total[k] = total.get(k, 0) + (v or 0)


def ledger_rows(tag_prefix: str = "", include_failed: bool = False) -> list[dict]:
    """Rows of rlm_ledger.jsonl whose tag starts with `tag_prefix` (empty if nothing has been logged yet). Runs that
    raised are logged with an `error` (so their spend is counted) and left out unless include_failed."""
    if not RLM_LEDGER.exists():
        return []
    rows = (json.loads(l) for l in RLM_LEDGER.read_text().splitlines() if l.strip())
    return [r for r in rows if r["tag"].startswith(tag_prefix) and (include_failed or r.get("error") is None)]


def _log(rec: dict) -> None:
    with _LEDGER_LOCK, RLM_LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


SYSTEM_JEV = """You are the root of a Recursive Language Model. You operate a persistent Python REPL.
A very long string is stored in the variable `context` - far too long to read directly. Never print it whole;
inspect it with slicing, splitting, regex, etc. (printed output is truncated to 4000 chars).

You have fast, cheap semantic leaf functions backed by a calibrated classifier model (Jev). They do NOT
generate text; they return probabilities. They are batched and parallel: 10,000 items take a few seconds.
  jev_classify(items: list[str], labels: dict[str, str], instruction: str | None = None) -> list[dict[str, float]]
      For each item, a probability distribution over `labels` (keys = label names, values = short descriptions
      of when the label applies). `instruction` optionally rephrases the question asked about each item.
  jev_yesno(items: list[str], question: str) -> list[float]
      For each item, P(yes) for the yes/no `question` asked about that item.
Use them for any semantic judgment over many items; use ordinary Python for parsing, filtering, counting,
dates, arithmetic and joins (the classifier is bad at math, counting and dates). Deduplicate items before
calling Jev when possible.

Reply with exactly one ```python ...``` block per turn. You will see its output. When you have the final
answer, call FINAL(answer_string) inside a python block, formatted exactly as the query requests."""

SYSTEM_LLM = (
    SYSTEM_JEV.split("You have fast")[0]
    + """You have a generative leaf function backed by a small LLM:
  llm_query(prompt: str) -> str
      Sends the prompt to a sub-LLM (context window ~150k chars) and returns its text answer. Each call takes
      several seconds; up to 8 calls can run in parallel via llm_query_batched(prompts: list[str]) -> list[str].
Use it for semantic judgments over chunks of the context; use ordinary Python for parsing, filtering,
counting, dates, arithmetic and joins.

Reply with exactly one ```python ...``` block per turn. You will see its output. When you have the final
answer, call FINAL(answer_string) inside a python block, formatted exactly as the query requests."""
)


def _claude(system: str, prompt: str, model: str) -> tuple[str, dict]:
    return run_claude(system, prompt, model, timeout=600)


class Final(BaseException):  # BaseException so model-written `except Exception:` blocks can't swallow FINAL()
    pass


def run_rlm(
    context: str,
    query: str,
    root_model: str = "haiku",
    leaf: str = "jev",
    max_turns: int = 10,
    leaf_model: str = "haiku",
    tag: str = "rlm",
    verbose: bool = False,
    custom: dict | None = None,
) -> dict:
    """custom = {"system": str, "tools": dict, "vars": dict, "desc": str} overrides the default leaf setup
    (used for corpus tasks such as BrowseComp-Plus)."""
    stats = {
        "jev_calls": 0,
        "jev_items": 0,
        "llm_calls": 0,
        "llm_usd_list": 0.0,
        "root_usd_list": 0.0,
        "turns": 0,
        "root_model": resolve(root_model),
        "llm_model": resolve(leaf_model) if leaf == "llm" else None,
        "root_usage": {},
        "llm_usage": {},
    }
    lock = threading.Lock()
    ns: dict = {"context": context, "re": re, "json": json}

    def jev_classify(items, labels, instruction=None):
        items, labels = _as_list(items), _as_labels(labels)
        stats["jev_calls"] += 1
        stats["jev_items"] += len(items)
        kw = {"inline_instruction": instruction} if instruction else {}
        return asyncio.run(label_texts(items, labels, batch=200, concurrency=24, tag=f"{tag}_leaf", **kw))

    def jev_yesno(items, question):
        items = _as_list(items)
        stats["jev_calls"] += 1
        stats["jev_items"] += len(items)

        return asyncio.run(yesno_texts(items, question, tag=f"{tag}_leaf"))

    def llm_query(prompt):
        res, m = _claude("You are a helpful sub-model. Answer concisely and exactly as asked.", str(prompt), leaf_model)
        with lock:
            stats["llm_calls"] += 1
            stats["llm_usd_list"] += m["usd_list"] or 0
            _add_usage(stats["llm_usage"], m["usage"])
        return res

    def llm_query_batched(prompts):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(llm_query, prompts))

    def FINAL(ans):
        raise Final(str(ans))

    if custom:
        ns.update(custom.get("tools", {}))
        ns.update(custom.get("vars", {}))
        system = custom["system"]
    elif leaf == "jev":
        ns.update(jev_classify=jev_classify, jev_yesno=jev_yesno)
        system = SYSTEM_JEV
    else:
        ns.update(llm_query=llm_query, llm_query_batched=llm_query_batched)
        system = SYSTEM_LLM
    ns["FINAL"] = FINAL

    if custom and custom.get("desc"):
        transcript = f"Query: {query}\n\n{custom['desc']}\n"
    else:
        transcript = (
            f"Query: {query}\n\n`context` is a str of {len(context):,} characters "
            f"(~{len(context) // 4:,} tokens). First 300 chars:\n{context[:300]!r}\n"
        )
    t0 = time.time()
    answer, error, transcript_ref = None, None, [transcript]
    try:
        answer = _turns(ns, system, transcript_ref, root_model, max_turns, custom, stats, verbose)
    except BaseException as e:  # log what the run spent before it failed, then re-raise
        error = repr(e)[:300]
        raise
    finally:
        transcript = transcript_ref[0]
        stats["wall_s"] = round(time.time() - t0, 1)
        rec = {
            "tag": tag,
            "query": query[:300],
            "ctx_chars": len(context) if isinstance(context, str) else -1,
            "answer": answer,
            "error": error,
            **stats,
        }
        _log(rec)
        (ROOT / "results" / "rlm_transcripts").mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "rlm_transcripts" / f"{tag}_{int(t0)}_{uuid.uuid4().hex[:6]}.txt").write_text(
            transcript.replace(str(ROOT) + "/", "")
        )  # repo-relative paths in tracebacks
    return {"answer": answer, **stats, "transcript": transcript}


def _turns(ns, system, transcript_ref, root_model, max_turns, custom, stats, verbose):
    """The root loop. Returns the answer; transcript_ref[0] holds the transcript as it grows."""
    transcript = transcript_ref[0]
    answer = None
    for turn in range(max_turns):
        stats["turns"] = turn + 1
        last = custom is not None and turn == max_turns - 1
        nudge = "\nThis is your LAST turn: call FINAL(...) now with your best answer." if last else ""
        reply, m = _claude(system, transcript + "\nWrite your next python block." + nudge, root_model)
        stats["root_usd_list"] += m["usd_list"] or 0
        _add_usage(stats["root_usage"], m["usage"])
        code = re.findall(r"```python\n(.*?)```", reply, re.S)
        if not code and custom is not None:
            fm = re.search(r"FINAL\((.*)\)", reply, re.S)
            if fm:
                answer = fm[1].strip().strip("\"'")
                transcript += f"\n--- turn {turn} ---\nFINAL (plain text): {answer}\n"
                transcript_ref[0] = transcript
                break
        if not code:
            transcript += f"\n--- turn {turn} ---\nASSISTANT:\n{reply}\nSYSTEM: no python block found; reply with one ```python block.\n"
            transcript_ref[0] = transcript
            continue
        buf = io.StringIO()
        try:
            with _STDOUT.capture(buf):
                exec(code[0], ns)
            out = buf.getvalue()
        except Final as f:
            answer = str(f)
            transcript += f"\n--- turn {turn} ---\n```python\n{code[0]}```\nFINAL: {answer}\n"
            transcript_ref[0] = transcript
            break
        except (Exception, SystemExit):  # model code that raises or calls exit() just gets the traceback back
            out = buf.getvalue() + "\n" + traceback.format_exc(limit=2)
        if len(out) > 4000:
            out = out[:4000] + f"\n...[truncated {len(out) - 4000} chars]"
        transcript += f"\n--- turn {turn} ---\n```python\n{code[0]}```\nOUTPUT:\n{out}\n"
        transcript_ref[0] = transcript
        if verbose:
            print(f"[turn {turn}]\n{code[0]}\n>>> {out[:800]}")
    if answer is None and custom is not None:
        # Out of turns without FINAL: one forced answer from what the transcript already contains (no code run).
        reply, m = _claude(
            system,
            transcript + "\nYou are out of turns. Reply with only FINAL(<your best answer>) "
            "based on what you have found so far.",
            root_model,
        )
        stats["root_usd_list"] += m["usd_list"] or 0
        _add_usage(stats["root_usage"], m["usage"])
        fm = re.search(r"FINAL\((.*)\)", reply, re.S)
        answer = fm[1].strip().strip("\"'") if fm else None
        stats["forced_final"] = True
    transcript_ref[0] = transcript
    return answer


SYSTEM_DIRECT = (
    "You are answering a question about a long document. Read it carefully and answer exactly in the "
    "format the question requests. Give only the final answer, no explanation."
)


def run_direct(
    context: str,
    query: str,
    model: str = "sonnet",
    tag: str = "direct",
    truncate=None,
    max_tokens: int = 900_000,
    chars_per_token: float = 2.2,
) -> dict:
    """No-RLM baseline: the whole context goes into a single prompt. With truncate=fn(text, max_chars) -> (text, kept),
    an over-long context is cut to about max_tokens first (and cut 10% further if the CLI still says it's too long)."""
    t0 = time.time()
    used, kept = context, 1.0
    for attempt in range(4):
        if truncate is not None:
            used, kept = truncate(context, int(max_tokens * chars_per_token * 0.9**attempt))
        try:
            res, m = _claude(SYSTEM_DIRECT, f"<document>\n{used}\n</document>\n\nQuestion: {query}", model)
            err = None
        except Exception as e:
            res, m, err = None, {"usd_list": 0, "usage": {}}, repr(e)[:200]
        if truncate is None or not (res or "").startswith(TOO_LONG):
            break

    rec = {
        "tag": tag,
        "query": query[:300],
        "ctx_chars": len(context),
        "answer": res,
        "error": err,
        "prompt_too_long": (res or "").startswith(TOO_LONG),
        "root_usd_list": m["usd_list"] or 0,
        "root_model": resolve(model),
        "ctx_chars_used": len(used),
        "kept_share": round(kept, 4),
        "llm_usd_list": 0.0,
        "jev_items": 0,
        "llm_calls": 0,
        "turns": 1,
        "wall_s": round(time.time() - t0, 1),
        "root_usage": m.get("usage", {}),
    }
    _log(rec)
    return rec
