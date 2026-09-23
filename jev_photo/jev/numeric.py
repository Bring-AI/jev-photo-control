"""`number` questions on a finite grid [lower, upper) with step `resolution`.

decoding="interval" (default) is jev-numeric's partition -> choose -> refine
algorithm (MIT): each round splits the current grid interval into <= K
sub-intervals and asks one lettered Choice question which contains the target.

decoding="digits" writes the number in the model's native form instead: a
constrained token walk over sign / digits / decimal point / end-of-answer (see
constrained.py), then snaps the value onto the grid. Each digit is a 10-way
choice, i.e. the same K=10 tree in decimal notation.

All pending number questions of a request advance in lockstep, so every round is
ONE batched suffix pass over the same shared image prefill. Greedy, no
backtracking. The final grid cell is the output resolution, not a confidence
interval.
"""
from decimal import ROUND_FLOOR, Decimal

from .constrained import Walker, walk
from .schema import Question, answer

INSTRUCTION = ("{target}\nSelect the interval containing this numerical value. Include the lower bound; "
               "exclude the upper bound. Option letters identify intervals, not the value itself.")
DIGITS_INSTRUCTION = ("{target}\nAnswer with a single number from {lo} to {hi}{places}, and nothing else.")


def fmt(x):
    return format(x.normalize(), "f")


def interval_question(question, lo, step, left, right):
    n = min(question.branching, right - left)
    cuts = [left + (right - left) * i // n for i in range(n + 1)]
    criteria = {str(i): f"{fmt(lo + step * cuts[i])} <= value < {fmt(lo + step * cuts[i + 1])}" for i in range(n)}
    return Question(type="choice", instructions=INSTRUCTION.format(target=question.instructions),
                    criteria=criteria), cuts


def result(value, step, trace, calls):
    return {"type": "number", "value": float(value), "exact": fmt(value),
            "interval": [fmt(value), fmt(value + step)], "resolution": fmt(step), "calls": calls, "trace": trace}


def decode_intervals(questions, prompter, session, temperature=1.0, max_rounds=64):
    state = {}
    for name, q in questions.items():
        lo, step, cells = q.grid()
        state[name] = {"q": q, "lo": lo, "step": step, "left": 0, "right": cells, "trace": []}
    rounds = 0
    while True:
        active = [name for name, s in state.items() if s["right"] - s["left"] > 1]
        if not active:
            break
        if rounds >= max_rounds:
            raise RuntimeError("round limit reached before requested resolution")
        rounds += 1
        asked = {name: interval_question(state[name]["q"], state[name]["lo"], state[name]["step"],
                                         state[name]["left"], state[name]["right"]) for name in active}
        scores = session.score([prompter.plan(asked[name][0]) for name in active])
        for name, values in zip(active, scores):
            choice_q, cuts = asked[name]
            picked = answer(choice_q, values, temperature)
            i = int(picked["choice"])
            s = state[name]
            s["left"], s["right"] = cuts[i], cuts[i + 1]
            s["trace"].append({"interval": choice_q.criteria[picked["choice"]],
                               "probabilities": {choice_q.criteria[k]: p for k, p in picked["probabilities"].items()}})
    return {name: result(s["lo"] + s["step"] * s["left"], s["step"], s["trace"], len(s["trace"]))
            for name, s in state.items()}, rounds


class NumberWalker(Walker):
    """Grammar: [-] int-digits [. frac-digits] <end>; no leading zeros, bounded lengths."""

    def __init__(self, ids, tok, lo, hi, places):
        super().__init__(ids)
        self.digit = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
        self.minus, self.dot = tok.convert_tokens_to_ids("-"), tok.convert_tokens_to_ids(".")
        self.end = [tok.eos_token_id]
        self.negative_ok, self.places = lo < 0, places
        self.int_limit = len(str(int(max(abs(lo), abs(hi)))))
        self.chars, self.done = "", False

    def text(self):
        return self.chars

    def options(self):
        digits = [(str(d), [i]) for d, i in enumerate(self.digit)]
        body = self.chars.lstrip("-")
        if body == "":
            return digits + ([("-", [self.minus])] if self.negative_ok and not self.chars else [])
        whole, _, frac = body.partition(".")
        if "." in body:
            return digits + [("end", self.end)] if frac and len(frac) < self.places else \
                (digits if not frac else [("end", self.end)])
        options = [("end", self.end)]
        if self.places:
            options.append((".", [self.dot]))
        if whole != "0" and len(whole) < self.int_limit:
            options = digits + options
        return options

    def take(self, label, tokens):
        if label == "end":
            self.done = True
        else:
            self.chars += label


def decode_digits(questions, prompter, session):
    walkers, grids = {}, {}
    for name, q in questions.items():
        lo, step, cells = q.grid()
        hi = lo + step * cells
        places = max(0, -step.normalize().as_tuple().exponent, -lo.normalize().as_tuple().exponent)
        text = DIGITS_INSTRUCTION.format(target=q.instructions, lo=fmt(lo), hi=fmt(hi),
                                         places=f", with at most {places} decimal places" if places else
                                         " (an integer)")
        suffix = f"Question: {text}" + prompter.ending + "Answer:\n"
        ids = prompter.encode(suffix)
        if prompter.encode(prompter.prefix + suffix) != prompter.encode(prompter.prefix) + ids:
            raise ValueError("unstable prefix/suffix tokenizer boundary")
        walkers[name] = NumberWalker(ids, prompter.tokenizer, lo, hi, places)
        grids[name] = (lo, step, cells)
    steps = walk(session, walkers)
    out = {}
    for name, w in walkers.items():
        lo, step, cells = grids[name]
        raw = Decimal(w.chars)
        # Snap onto the grid cell containing the written value; clamp into range.
        index = int(((raw - lo) / step).to_integral_value(rounding=ROUND_FLOOR))
        index = min(max(index, 0), cells - 1)
        out[name] = {**result(lo + step * index, step, w.trace, len(w.trace)), "written": w.chars}
    return out, steps


def decode_numbers(questions, prompter, session, temperature=1.0):
    """questions: {name: Question(type="number")} -> ({name: answer}, sequential rounds)."""
    intervals = {k: q for k, q in questions.items() if q.decoding == "interval"}
    digits = {k: q for k, q in questions.items() if q.decoding == "digits"}
    out, rounds = {}, 0
    if intervals:
        decoded, r = decode_intervals(intervals, prompter, session, temperature)
        out.update(decoded)
        rounds += r
    if digits:
        decoded, r = decode_digits(digits, prompter, session)
        out.update(decoded)
        rounds += r
    return {k: out[k] for k in questions}, rounds
