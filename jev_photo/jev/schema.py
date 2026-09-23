"""Typed questions and answer assembly.

choice / noul / score are ported from jev-visual (MIT). `number` follows
jev-numeric (MIT): a finite grid [lower, upper) with step `resolution`, decoded
by repeated K-way interval choices (see numeric.py). `box` locates what the
instructions describe via constrained digit scoring (see grounding.py).
"""
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score", "number", "box"]
    instructions: str = Field(min_length=1, max_length=4000)
    criteria: dict[str, str] | list[str] | None = None
    scoring: Literal["label", "single_token", "sequence"] = "label"
    candidates: dict[str, str] | None = None
    # number only
    range: tuple[str | float, str | float] | None = None
    resolution: str | float = "0.01"
    branching: int = Field(default=10, ge=2, le=26)
    decoding: Literal["interval", "digits"] = "interval"

    @model_validator(mode="after")
    def validate_criteria(self):
        c = self.criteria
        if self.type == "box":
            if c is not None or self.candidates is not None or self.range is not None or self.scoring != "label" \
                    or self.decoding != "interval":
                raise ValueError("box takes only instructions (what to locate)")
            return self
        if self.type == "number":
            if c is not None or self.candidates is not None or self.scoring != "label":
                raise ValueError("number takes range/resolution/branching/decoding, not criteria or candidates")
            if self.range is None:
                raise ValueError("number requires range [lower, upper]")
            self.grid()  # validates
            return self
        if self.decoding != "interval":
            raise ValueError("decoding applies only to number")
        if self.type == "choice" and not isinstance(c, dict):
            raise ValueError("choice requires a mapping of option ID to description")
        if self.type == "score" and not isinstance(c, list):
            raise ValueError("score requires an ordered list of level descriptions")
        if self.type == "noul" and c is not None:
            if not isinstance(c, dict) or set(c) != {"true", "false"}:
                raise ValueError("noul criteria must contain exactly true and false")
        if c is not None:
            if not 2 <= len(c) <= 26:
                raise ValueError("provide 2 to 26 options/levels")
            values = c.values() if isinstance(c, dict) else c
            if any(not x.strip() or len(x) > 2000 for x in values):
                raise ValueError("criteria must be nonempty and at most 2000 characters")
            if isinstance(c, dict) and any(not k.strip() for k in c):
                raise ValueError("option IDs must be nonempty")
        if self.candidates is not None:
            if self.scoring == "label":
                raise ValueError("candidates applies only to single_token or sequence scoring")
            if set(self.candidates) != {key for key, _ in self.options()}:
                raise ValueError("candidate keys must exactly match option IDs")
            if any(not value.strip() or len(value) > 1000 for value in self.candidates.values()):
                raise ValueError("candidate text must be nonempty and at most 1000 characters")
        return self

    def grid(self):
        """Return (lower, step, cells) as exact Decimals/int for a number question."""
        try:
            lo, hi = (Decimal(str(x)) for x in self.range)
            step = Decimal(str(self.resolution))
        except InvalidOperation as exc:
            raise ValueError("range and resolution must be finite decimals") from exc
        if not all(x.is_finite() for x in (lo, hi, step)) or hi <= lo or step <= 0:
            raise ValueError("require finite lower < upper and positive resolution")
        cells = (hi - lo) / step
        if cells != cells.to_integral_value():
            raise ValueError("range width must be an integer multiple of resolution")
        if cells > 10**9:
            raise ValueError("at most 1e9 grid cells")
        return lo, step, int(cells)

    def options(self):
        if self.type == "noul":
            c = self.criteria or {"true": "Yes, the condition holds.", "false": "No, the condition does not hold."}
            return [(key, c[key]) for key in ("true", "false")]
        if self.type == "score":
            return [(str(i), value) for i, value in enumerate(self.criteria)]
        return list(self.criteria.items())


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1)
    state: Any = ""
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    temperature: float = Field(default=1.0, gt=0, le=10, allow_inf_nan=False)
    mode: Literal["shared", "independent"] = "shared"


def answer(question: Question, logits: list[float], temperature: float):
    """Conditional candidate softmax, NOT calibrated correctness probabilities."""
    scores = np.asarray(logits, dtype=np.float64) / temperature
    if scores.shape != (len(question.options()),) or not np.isfinite(scores).all():
        raise ValueError("invalid candidate logits")
    p = np.exp(scores - scores.max())
    p /= p.sum()
    keys = [key for key, _ in question.options()]
    probabilities = dict(zip(keys, map(float, p)))
    entropy = -sum(float(v) * math.log(float(v)) for v in p if v > 0)
    result = {"type": question.type, "probabilities": probabilities}
    if question.type == "noul":
        result["noul"] = probabilities["true"]
    else:
        result["concentration"] = max(0.0, 1.0 - entropy / math.log(len(p)))
        if question.type == "choice":
            result["choice"] = keys[int(p.argmax())]
        else:
            result["score"] = sum(i * float(v) for i, v in enumerate(p))
            result["legend"] = dict(question.options())
    return result
