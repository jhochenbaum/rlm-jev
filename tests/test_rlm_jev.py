"""Offline tests for the rlm_jev library. A fake client stands in for Jev, so no key or network is needed."""

import importlib
from types import SimpleNamespace
from typing import ClassVar

import pytest

import rlm_jev
from rlm_jev import Spend, apply_weights, classify, fit_weights, mc_confidence, rlm_tools, yesno
from rlm_jev import _client as client_mod
from rlm_jev.calibrate import ece

locate_mod = importlib.import_module("rlm_jev.locate")


class FakeJev:
    """Answers Choice questions by naming the label that appears in the item, and Noul questions with 0.9 when the
    item (or, for window-level questions, the state) mentions a heron. Records every request."""

    calls: ClassVar[list] = []
    fail_on_call: int | None = None
    tokens_per_question = 100

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def system_one(self, state, questions):
        FakeJev.calls.append((state, questions))
        if FakeJev.fail_on_call is not None and len(FakeJev.calls) == FakeJev.fail_on_call:
            raise ValueError("bad request")
        answers = {}
        for key, q in questions.items():
            item = (
                q.instructions.get("item") or next(iter(q.instructions.values()))
                if isinstance(q.instructions, dict)
                else ""
            )
            if q.type == "choice":
                labels = list(q.criteria)
                hit = next((lab for lab in labels if lab in str(item)), labels[-1])
                answers[key] = SimpleNamespace(
                    probabilities={lab: 0.9 if lab == hit else 0.1 / (len(labels) - 1) for lab in labels}
                )
            else:
                text = str(item) if isinstance(q.instructions, dict) else str(state)
                answers[key] = SimpleNamespace(noul=0.9 if "heron" in text else 0.1)
        usage = SimpleNamespace(input_tokens=FakeJev.tokens_per_question * len(questions))
        return SimpleNamespace(answers=answers, usage=usage)


@pytest.fixture(autouse=True)
def fake_jev(monkeypatch):
    FakeJev.calls = []
    FakeJev.fail_on_call = None
    monkeypatch.setattr(client_mod, "AsyncTypeSafeClient", FakeJev)
    return FakeJev


# ---------------------------------------------------------------- request sizing


def test_estimate_tokens_counts_utf8_bytes():
    assert client_mod.estimate_tokens("abc" * 10) == 11
    assert client_mod.estimate_tokens("漢" * 30) == 31  # about one token per CJK character


def test_pack_respects_item_and_token_limits():
    batches = client_mod.pack(["x" * 30] * 10, fixed_tokens=0, per_question_tokens=0, max_items=4)
    assert [len(b) for b in batches] == [4, 4, 2]
    assert [i for b in batches for i in b] == list(range(10))
    small = client_mod.pack(["x" * 3000] * 10, fixed_tokens=0, per_question_tokens=0, max_items=100, max_tokens=2500)
    assert all(len(b) <= 2 for b in small)


def test_pack_rejects_oversized_inputs():
    with pytest.raises(ValueError, match="item 1"):
        client_mod.pack(["ok", "x" * 300_000], fixed_tokens=0, per_question_tokens=0, max_items=10)
    with pytest.raises(ValueError, match="shared context"):
        client_mod.pack(["ok"], fixed_tokens=60_000, per_question_tokens=0, max_items=10)


# ---------------------------------------------------------------- classify and yesno


def test_classify_one_distribution_per_item_and_string_is_one_item():
    probs = classify(["a refund request", "a billing question"], ["refund", "billing"])
    assert [max(p, key=p.get) for p in probs] == ["refund", "billing"]
    assert all(abs(sum(p.values()) - 1) < 1e-9 for p in probs)
    assert len(classify("a single refund item", ["refund", "billing"])) == 1


def test_classify_label_descriptions_per_question_or_in_state():
    labels = {"refund": "asks for money back", "billing": "asks about an invoice"}
    classify(["a refund request"], labels)
    state, qs = FakeJev.calls[-1]
    assert dict(next(iter(qs.values())).criteria) == labels and "asks for money back" not in state
    classify(["a refund request"], labels, labels_in_state=True)
    state, qs = FakeJev.calls[-1]
    assert "asks for money back" in state and all(v is None for v in next(iter(qs.values())).criteria.values())


def test_classify_ensemble_reverses_option_order():
    classify(["a refund request"], ["refund", "billing", "other"], ensemble=True)
    orders = [list(next(iter(qs.values())).criteria) for _, qs in FakeJev.calls]
    assert orders == [["refund", "billing", "other"], ["other", "billing", "refund"]]


def test_yesno_returns_p_yes_per_item():
    assert yesno(["a grey heron", "a sparrow"], "Is this a heron?") == [0.9, 0.1]


# ---------------------------------------------------------------- spend accounting


def test_spend_cap_refuses_before_sending():
    with pytest.raises(RuntimeError, match="budget cap"):
        classify(["item"] * 50, ["a", "b"], spend=Spend(max_usd=1e-9))
    assert FakeJev.calls == []


def test_spend_settles_to_actual_usage():
    spend = Spend(max_usd=1.0)
    classify(["item"] * 10, ["a", "b"], batch=5, spend=spend)
    assert spend.tokens == 10 * FakeJev.tokens_per_question
    assert spend._reserved == pytest.approx(0)


def test_failed_request_is_counted_at_its_estimate_and_reraised():
    spend = Spend(max_usd=1.0)
    FakeJev.fail_on_call = 2
    with pytest.raises(ValueError, match="bad request"):
        classify(["item"] * 10, ["a", "b"], batch=5, spend=spend)
    assert spend.tokens > 5 * FakeJev.tokens_per_question  # the good request plus the failed one's estimate
    assert spend._reserved == pytest.approx(0)


# ---------------------------------------------------------------- locate


def test_locate_finds_the_passage_and_offsets_match_the_text():
    filler = "The weather was fair and the road was long. " * 600
    doc = filler + "Old Mr. Tredgold kept a tame heron in the walled orchard. " + filler
    hits = rlm_jev.locate([doc, "nothing here at all. just filler. more filler."], "Who kept a bird?", k=3)
    top = hits["Who kept a bird?"][0]
    assert "heron" in top["text"] and top["doc"] == 0
    for h in hits["Who kept a bird?"]:
        assert [doc, "nothing here at all. just filler. more filler."][h["doc"]][
            h["offset"] : h["offset"] + len(h["text"])
        ] == h["text"]


def test_locate_sentence_splitter_keeps_titles_together():
    unit, units = locate_mod._units("It was late. Mrs. Ingleby dined with Mr. J. Pym. He left.")
    assert unit == "sentence"
    assert [u for _, u in units] == ["It was late.", "Mrs. Ingleby dined with Mr. J. Pym.", "He left."]


def test_locate_edge_cases():
    assert rlm_jev.locate("", ["q"]) == {"q": []}
    with pytest.raises(ValueError, match="window_chars"):
        rlm_jev.locate("text", ["q"], window_chars=10)


# ---------------------------------------------------------------- calibration helpers


def test_fit_weights_corrects_a_biased_label():
    probs = [{"a": 0.6, "b": 0.4}] * 6 + [{"a": 0.8, "b": 0.2}] * 4
    gold = ["b"] * 6 + ["a"] * 4
    w = fit_weights(probs, gold)
    assert apply_weights(probs, w) == gold


def test_mc_confidence_agrees_on_a_certain_answer():
    probs = [{"a": 1.0, "b": 0.0}] * 5
    answer, conf = mc_confidence(lambda labs: labs.count("a"), probs)
    assert (answer, conf) == (5, 1.0)


def test_ece():
    assert ece([0.9, 0.9], [True, True]) == pytest.approx(0.1)
    assert ece([1.0000001], [True]) == pytest.approx(0.0)


# ---------------------------------------------------------------- RLM tools


def test_rlm_tools_styles():
    fns = rlm_tools(max_usd=0.5)
    assert [f.__name__ for f in fns] == ["jev_classify", "jev_yesno", "jev_locate", "jev_spend"]
    tools = rlm_tools(style="rlm")
    assert set(tools) == {"jev_classify", "jev_yesno", "jev_locate", "jev_spend"}
    assert all(callable(t["tool"]) and t["description"] for t in tools.values())
    with pytest.raises(ValueError):
        rlm_tools(style="other")
