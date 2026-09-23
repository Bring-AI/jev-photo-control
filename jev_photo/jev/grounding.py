"""Boxes via constrained digit scoring in Qwen3.5's native grounding format.

Qwen3.5 was trained to emit `{"bbox_2d": [x1, y1, x2, y2]}` with integer
coordinates on a 0..1000 scale, one token per digit. Instead of sampling, we
teacher-force the JSON scaffold and, at each step, score ONLY the legal next
tokens (digits, and the separator once a number has at least one digit). This is
the jev-numeric K=10 interval tree expressed in decimal digits, in the model's
own coordinate format.
"""
from .constrained import Walker, walk

OPEN = '```json\n[\n\t{"bbox_2d": ['
INSTRUCTION = ('Locate {what} in the image. Output its bounding box as JSON: '
               '[{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}]')
SCALE = 1000


class BoxWalker(Walker):
    def __init__(self, ids, tok):
        super().__init__(ids)
        self.digit = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        self.comma, self.space, self.close = (tok.encode(s, add_special_tokens=False) for s in (",", " ", "]"))
        if any(len(x) != 1 for x in (self.comma, self.space, self.close)) or None in self.digit:
            raise ValueError("tokenizer does not encode digits/separators as single tokens")
        self.coords, self.digits = [], ""

    @property
    def done(self):
        return len(self.coords) == 4

    def text(self):
        return ", ".join(map(str, self.coords + ([self.digits] if self.digits else [])))

    def options(self):
        end = self.close if len(self.coords) == 3 else self.comma + self.space
        digits = [(str(d), [i]) for d, i in enumerate(self.digit)]
        # Integers 0..1000, no leading zeros.
        if self.digits == "":
            return digits
        if self.digits == "0" or len(self.digits) == 4 or (len(self.digits) == 3 and self.digits != "100"):
            return [("end", end)]
        if len(self.digits) == 3:
            return [("0", [self.digit[0]]), ("end", end)]
        return digits + [("end", end)]

    def take(self, label, tokens):
        if label == "end":
            self.coords.append(int(self.digits))
            self.digits = ""
        else:
            self.digits += label


def locate(prompter, session, targets: dict):
    """targets: {name: what to locate} -> {name: {"bbox_2d": 0..1000 ints, "box": fractions}}."""
    walkers = {}
    for name, what in targets.items():
        suffix = f"Question: {INSTRUCTION.format(what=what)}" + prompter.ending + OPEN
        ids = prompter.encode(suffix)
        if prompter.encode(prompter.prefix + suffix) != prompter.encode(prompter.prefix) + ids:
            raise ValueError("unstable prefix/suffix tokenizer boundary")
        walkers[name] = BoxWalker(ids, prompter.tokenizer)
    steps = walk(session, walkers)
    out = {}
    for name, w in walkers.items():
        x0, y0, x1, y1 = w.coords
        out[name] = {"bbox_2d": w.coords, "box": [x0 / SCALE, y0 / SCALE, x1 / SCALE, y1 / SCALE],
                     "trace": w.trace}
    return out, steps
