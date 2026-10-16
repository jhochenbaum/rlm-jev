"""dspy.RLM that runs on Claude 5 models (needs the `dspy` extra).

Stock dspy.RLM asks the root for a step-by-step `reasoning` output field. Claude Sonnet 5.5 declines such prompts
(refusal category `reasoning_extraction`; https://github.com/stanfordnlp/dspy/issues/10641). Anthropic's guidance is to
ask for a short explanation instead:
https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#keep-reasoning-in-thinking-blocks

NoteRLM replaces that one field with a one-sentence `note` and leaves the rest of dspy.RLM unchanged.
"""

from __future__ import annotations

import dspy


class _NotePredict(dspy.Predict):
    """Predict whose `note` output is also exposed as `reasoning`, the attribute dspy.RLM reads."""

    def __call__(self, *args, **kwargs):
        pred = super().__call__(*args, **kwargs)
        pred.reasoning = pred.get("note", "")
        return pred

    async def acall(self, *args, **kwargs):
        pred = await super().acall(*args, **kwargs)
        pred.reasoning = pred.get("note", "")
        return pred


class NoteRLM(dspy.RLM):
    """Drop-in replacement for dspy.RLM: same arguments, `reasoning` field replaced by a one-sentence `note`."""

    def _build_signatures(self):
        action_sig, extract_sig = super()._build_signatures()
        code_field = action_sig.output_fields["code"]
        action_sig = (
            action_sig.delete("reasoning")
            .delete("code")
            .append("note", dspy.OutputField(desc="One short sentence on what this step does."), type_=str)
            .append("code", code_field, type_=str)
        )
        return action_sig, extract_sig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_action = _NotePredict(self._action_signature)
