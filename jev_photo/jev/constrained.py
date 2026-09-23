"""Constrained token walks: at each step score ONLY the legal next tokens and take the argmax.

No sampling and no free text: a small grammar (a Walker) lists the legal tokens,
the model's next-token logits over the shared image prefill rank them. Several
walkers advance in lockstep, one batched suffix pass per step; steps with a
single legal option are forced without a model call.
"""
import math

from .scoring import Plan


class Walker:
    def __init__(self, ids):
        self.ids = list(ids)
        self.trace = []

    done = False

    def options(self):
        """-> list of (label, [token ids]); label is recorded, tokens are appended."""
        raise NotImplementedError

    def take(self, label, tokens):
        raise NotImplementedError

    def text(self):
        raise NotImplementedError


def walk(session, walkers: dict, max_steps=64):
    steps = 0
    while not all(w.done for w in walkers.values()):
        if steps >= max_steps:
            raise RuntimeError("constrained walk step limit reached")
        active = [w for w in walkers.values() if not w.done]
        legal = [w.options() for w in active]
        scored = [i for i, o in enumerate(legal) if len(o) > 1]
        results = {}
        if scored:
            steps += 1
            plans = [Plan("", list(active[i].ids), [t for _, t in legal[i]], "single_token", "") for i in scored]
            results = dict(zip(scored, session.score(plans)))
        for i, (w, options) in enumerate(zip(active, legal)):
            values = results[i] if i in results else [0.0]
            best = max(range(len(options)), key=values.__getitem__)
            label, tokens = options[best]
            if len(options) > 1:
                top = max(values)
                z = sum(math.exp(v - top) for v in values)
                w.trace.append({"after": w.text(), "choice": label, "p": math.exp(values[best] - top) / z})
            w.ids += tokens
            w.take(label, tokens)
    return steps
