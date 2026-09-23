"""Exact candidate tokenization and teacher-forced scoring; no sampling.

Ported from jev-visual (MIT). The difference: a `Session` keeps the shared
image/context prefill alive so that several scoring rounds (e.g. the successive
interval choices of a `number` question) branch from the same cache.
"""
import hashlib
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Plan:
    suffix: str
    suffix_ids: list[int]
    targets: list[list[int]]
    scoring: str
    prompt_sha256: str


def candidate_tokens(tokenizer, prompt, texts, mode):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix = encode(prompt)
    targets = []
    for text in texts:
        ids = encode(text)
        if not ids or len(ids) > 64 or tokenizer.decode(ids) != text:
            raise ValueError("candidates must round-trip exactly and contain 1..64 tokens")
        if set(ids) & set(tokenizer.all_special_ids):
            raise ValueError("candidate text must not contain special control tokens")
        if encode(prompt + text) != prefix + ids:
            raise ValueError("candidate changes contextual token boundary; use label scoring")
        if mode != "sequence" and len(ids) != 1:
            raise ValueError("single_token candidate is not one token; use sequence or label")
        if mode == "sequence":
            if tokenizer.eos_token_id is None:
                raise ValueError("sequence scoring requires an end-of-turn token")
            ids = ids + [tokenizer.eos_token_id]
        targets.append(ids)
    if len({tuple(ids) for ids in targets}) != len(targets):
        raise ValueError("candidate token sequences collide")
    return targets


def digest(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def sequence_logprob(logits, target):
    """Reference math used by tests: full-vocabulary normalization at EVERY step."""
    x = np.asarray(logits, dtype=np.float64)
    maxima = x.max(axis=-1)
    normalizer = maxima + np.log(np.exp(x - maxima[:, None]).sum(axis=-1))
    return float((x[np.arange(len(target)), target] - normalizer).sum())


def tasks_for(plans):
    """One task per label/single-token question; one per candidate in sequence mode."""
    tasks = []
    for index, plan in enumerate(plans):
        if plan.scoring == "sequence":
            for candidate, target in enumerate(plan.targets):
                # Position len(suffix)-1 predicts target[0], then teacher-force
                # target[:-1]. EOS distinguishes prefix-overlap answers.
                rows = plan.suffix_ids + target[:-1]
                picks = list(range(len(plan.suffix_ids) - 1, len(rows)))
                tasks.append((index, candidate, rows, picks, target))
        else:
            tasks.append((index, None, plan.suffix_ids, [len(plan.suffix_ids) - 1], None))
    return tasks


def consume(task, logits, plans, scores):
    """Turn projected logits (float32, [picks, vocab]) into a candidate score."""
    import torch
    index, candidate, _, _, target = task
    if target is None:
        ids = [tokens[0] for tokens in plans[index].targets]
        scores[index] = logits[0, ids].tolist()
    else:
        # DO NOT normalize over candidate tokens at intermediate positions.
        normalizers = torch.logsumexp(logits, dim=-1)
        rows = torch.arange(len(target), device=logits.device)
        values = logits[rows, torch.tensor(target, device=logits.device)] - normalizers
        scores[index][candidate] = float(values.sum().item())


class Session:
    """Scores question plans against one image + prefix.

    mode="shared": vision + prefix prefill once, then each batch of suffixes runs
    on a fork of that cache. mode="independent": reference path, a full forward
    of image + prompt + suffix per task.
    """

    def __init__(self, adapter, prefix, image, mode="shared", batch_size=16, max_tokens=8000):
        self.adapter, self.prefix, self.image, self.mode = adapter, prefix, image, mode
        self.batch_size, self.max_tokens = batch_size, max_tokens
        self.cache = self.position = None
        self.prefix_tokens = None
        if mode == "shared":
            inputs = adapter.prepare(prefix, image)
            self.prefix_tokens = int(inputs["input_ids"].shape[-1])
            self.cache, self.position, _ = adapter.prefill(inputs)

    def score(self, plans):
        tasks = tasks_for(plans)
        scores = [[0.0] * len(plan.targets) for plan in plans]
        started = time.perf_counter()
        if self.mode == "shared":
            if any(self.prefix_tokens + len(task[2]) > self.max_tokens for task in tasks):
                raise ValueError(f"image, question and candidate exceed {self.max_tokens} tokens")
            # Longest rows first keeps padding low inside each batch.
            order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][2]))
            for start in range(0, len(order), self.batch_size):
                batch = [tasks[i] for i in order[start:start + self.batch_size]]
                logits = self.adapter.suffix(self.cache, self.position, [t[2] for t in batch], [t[3] for t in batch])
                for task, values in zip(batch, logits):
                    consume(task, values, plans, scores)
        else:
            for task in tasks:
                _, _, rows, picks, _ = task
                # The prefix/suffix boundary is verified stable, so appending the
                # suffix (and teacher-forced candidate) ids equals encoding the full text.
                inputs = self.adapter.prepare(self.prefix, self.image, extra_ids=rows)
                total = int(inputs["input_ids"].shape[-1])
                if total > self.max_tokens:
                    raise ValueError(f"image, question and candidate exceed {self.max_tokens} tokens")
                offset = total - len(rows)
                _, _, logits = self.adapter.prefill(inputs, [[offset + p for p in picks]])
                consume(task, logits[0], plans, scores)
        self.adapter.timings.scoring_ms += (time.perf_counter() - started) * 1000
        return scores
