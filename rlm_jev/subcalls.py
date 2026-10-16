"""Batched per-item sub-calls: classify (Choice) and yesno (Noul), each item inside its own question."""

from __future__ import annotations

from typesafe_sdk import Choice, Noul

from ._client import TOKENS_PER_QUESTION, TOKENS_PER_REQUEST, Spend, ask_many, estimate_tokens, pack, run

DEFAULT_CONTEXT = "Judge each item independently."

# Each item goes inside its own question; the state carries only shared context. Never point at items by position
# (e.g. "records[17]" in an unlabeled list): accuracy fell from 94% to 34% at 100 items per request.
# Label descriptions go in every question by default. labels_in_state=True writes them once in the state instead:
# 2.6x fewer tokens and the same calibration, but on 4,151 TREC records about 1.5 points lower accuracy and more
# biased label counts (which fitted label weights correct well).


def _as_list(items) -> list[str]:
    return [items] if isinstance(items, str) else [str(x) for x in items]


def _as_labels(labels) -> dict[str, str | None]:
    return dict.fromkeys(labels) if isinstance(labels, list | tuple) else dict(labels)


def _answers(resps, batches, n_items, read):
    out = [None] * n_items
    for r, idx in zip(resps, batches, strict=False):
        for j, i in enumerate(idx):
            if f"q{j}" not in r.answers:
                raise RuntimeError(f"Jev returned no answer for question q{j}")
            out[i] = read(r.answers[f"q{j}"])
    return out


def classify(
    items,
    labels,
    instruction: str = "Which label best describes `item`?",
    context: str | None = None,
    batch: int = 200,
    ensemble: bool = False,
    labels_in_state: bool = False,
    spend: Spend | None = None,
    concurrency: int = 24,
) -> list[dict[str, float]]:
    """One probability distribution over `labels` per item.

    items: a list of strings (a single string is treated as one item)
    labels: {label_name: description of when it applies, or None}, or just a list of label names
    context: shared task description sent once per request as the state (e.g. the header of your dataset)
    batch: maximum items per request; requests are also kept under Jev's request-size limit
    ensemble: also ask with the option order reversed and average (Jev has a mild first-option bias)
    labels_in_state: write label descriptions once in the state instead of in every question (cheaper, less accurate)
    spend: a Spend to account against (and cap); see rlm_jev.Spend
    """
    items, labels = _as_list(items), _as_labels(labels)
    state = context or DEFAULT_CONTEXT
    if labels_in_state and any(labels.values()):
        state += "\nThe labels:\n" + "\n".join(f"- {k}: {d}" if d else f"- {k}" for k, d in labels.items())
        labels = dict.fromkeys(labels)
    fixed = estimate_tokens(state)
    per_q = estimate_tokens(instruction, *labels, *(d or "" for d in labels.values())) + TOKENS_PER_QUESTION
    batches = pack(items, fixed, per_q, batch)

    def ask(opts):
        reqs = [
            (
                state,
                {
                    f"q{j}": Choice(instructions={"item": items[i], "question": instruction}, criteria=opts)
                    for j, i in enumerate(idx)
                },
                fixed + TOKENS_PER_REQUEST + sum(estimate_tokens(items[i]) + per_q for i in idx),
            )
            for idx in batches
        ]
        resps = run(ask_many(reqs, concurrency, spend=spend))
        return _answers(resps, batches, len(items), lambda a: dict(a.probabilities))

    fwd = ask(labels)
    if not ensemble:
        return fwd
    rev = ask(dict(reversed(list(labels.items()))))
    return [{k: (a[k] + b[k]) / 2 for k in a} for a, b in zip(fwd, rev, strict=False)]


def yesno(
    items,
    question: str,
    context: str | None = None,
    batch: int = 200,
    spend: Spend | None = None,
    concurrency: int = 24,
) -> list[float]:
    """P(yes) for `question` asked about each item (a single string is treated as one item)."""
    items = _as_list(items)
    state = context or DEFAULT_CONTEXT
    fixed = estimate_tokens(state)
    per_q = estimate_tokens(question) + TOKENS_PER_QUESTION
    batches = pack(items, fixed, per_q, batch)
    reqs = [
        (
            state,
            {f"q{j}": Noul(instructions={"item": items[i], "question": question}) for j, i in enumerate(idx)},
            fixed + TOKENS_PER_REQUEST + sum(estimate_tokens(items[i]) + per_q for i in idx),
        )
        for idx in batches
    ]
    resps = run(ask_many(reqs, concurrency, spend=spend))
    return _answers(resps, batches, len(items), lambda a: a.noul)
