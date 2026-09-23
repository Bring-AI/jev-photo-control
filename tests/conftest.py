import re

import pytest
from transformers import AutoTokenizer

from jev_photo.jev.preprocessing import Prompter


class FakeProcessor:
    """Real Qwen3.5 tokenizer + chat template; no model weights."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)


@pytest.fixture(scope="session")
def processor():
    return FakeProcessor()


@pytest.fixture
def prompter(processor):
    return Prompter(processor, {"edit_request": "test"})


class ScriptedSession:
    """Stands in for scoring.Session: an oracle that knows the true answer per question.

    - interval choices: picks the option whose "lo <= value < hi" contains the target
    - constrained walks: picks the candidate token that continues the target text
    """

    def __init__(self, tokenizer, interval_targets=None, text_targets=None, base_len=None):
        self.tok = tokenizer
        self.interval_targets = interval_targets or {}
        self.text_targets = text_targets or []
        self.calls = 0

    def score(self, plans):
        self.calls += 1
        return [self._score(plan) for plan in plans]

    def _score(self, plan):
        n = len(plan.targets)
        if plan.suffix:  # lettered interval question
            for marker, value in self.interval_targets.items():
                if marker in plan.suffix:
                    ranges = re.findall(r"^[A-Z]\. (\S+) <= value < (\S+)$", plan.suffix, re.M)
                    hit = next(i for i, (lo, hi) in enumerate(ranges) if float(lo) <= value < float(hi))
                    return [10.0 if i == hit else 0.0 for i in range(n)]
            raise AssertionError("no scripted target for question")
        text = self.tok.decode(plan.suffix_ids)
        for marker, target in self.text_targets:
            if marker in text:
                written = text.rsplit(marker, 1)[1]
                assert target.startswith(written), (target, written)
                rest = target[len(written):]
                for i, tokens in enumerate(plan.targets):
                    piece = self.tok.decode(tokens)
                    if (rest and rest.startswith(piece) and piece) or (not rest and tokens[0] == self.tok.eos_token_id):
                        return [10.0 if j == i else 0.0 for j in range(n)]
                raise AssertionError(f"no legal token continues {rest!r}; options {[self.tok.decode(t) for t in plan.targets]}")
        raise AssertionError("no scripted target for walk")
