"""Zero-cost regex/heuristic leaf baselines (to show the leaf isn't trivially regexable)."""

import re


def trec_rule(t: str) -> str:
    s = t.lower().strip()
    if re.search(r"stand for|abbreviat|acronym|what does \w+ mean\b", s):
        return "abbreviation"
    if re.match(r"(who|whom|whose)\b", s) or re.search(
        r"\bwhat (person|man|woman|actor|president|company|team|group)\b", s
    ):
        return "human being"
    if re.match(r"where\b", s) or re.search(
        r"\b(what|which) (city|country|state|continent|river|mountain|place|island|county)\b", s
    ):
        return "location"
    if re.match(r"(when|how (many|much|long|far|old|tall|big|fast|hot|cold|large))\b", s) or re.search(
        r"\bwhat (year|date|percentage|number|population)\b", s
    ):
        return "numeric value"
    if re.match(r"(why|how)\b", s) or re.match(r"what (is|are|was|were) (a|an|the)? ?\w+ \?$", s) or "definition" in s:
        return "description and abstract concept"
    return "entity"


SPAM_RE = re.compile(
    r"(free|win|won|prize|claim|cash|£|\$\d|txt|text \w+ to|call 0|urgent|award|offer|ringtone|reply|stop|mobile|www\.|http|guaranteed|selected|voucher|subscri)",
    re.I,
)


def spam_rule(t: str) -> str:
    return "spam" if len(SPAM_RE.findall(t)) >= 2 else "ham"


RULES = {"trec_coarse": trec_rule, "spam": spam_rule}
