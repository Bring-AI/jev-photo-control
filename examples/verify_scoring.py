"""Check shared-prefix scores against independent forwards and the stock model forward.

    python -m examples.verify_scoring cuda:1 [bfloat16|float32]
"""
import json
import sys
import time

import torch

from jev_photo.jev import Engine, Request
from jev_photo.jev.preprocessing import Prompter, read_image
from jev_photo.jev.scoring import sequence_logprob

QUESTIONS = {
    "animal": {"type": "choice", "instructions": "Which animal is the main subject?",
               "criteria": {"cat": "A cat", "dog": "A dog", "bird": "A bird", "other": "Other"}},
    "eyes": {"type": "noul", "instructions": "Does the animal have green eyes?"},
    # Overlapping candidates: EOS must separate "tabby" from "tabby cat".
    "seq": {"type": "choice", "instructions": "Which description fits the animal?", "scoring": "sequence",
            "criteria": {"t": "tabby", "tc": "tabby cat", "bd": "black dog"},
            "candidates": {"t": "tabby", "tc": "tabby cat", "bd": "black dog"}},
    # Non-ASCII (Chinese) candidates on purpose: checks multi-token scoring outside English.
    "zh": {"type": "choice", "instructions": "图中是什么动物？", "scoring": "sequence",
           "criteria": {"dog": "狗", "cat": "猫"}, "candidates": {"dog": "狗", "cat": "猫"}},
    "legs": {"type": "number", "instructions": "How many legs of the animal are visible?", "range": [0, 10],
             "resolution": 1},
}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    dtype = sys.argv[2] if len(sys.argv) > 2 else "bfloat16"
    engine = Engine(device=device, dtype=dtype)
    base = {"image": "examples/photos/cat.jpg", "state": "", "questions": QUESTIONS}
    engine.judge(Request(**base))  # warmup (triton JIT)
    runs = {}
    for mode in ("shared", "independent"):
        t = time.perf_counter()
        runs[mode] = engine.judge(Request(**base, mode=mode))
        print(mode, f"{(time.perf_counter() - t) * 1000:.0f} ms", runs[mode]["metrics"]["language_forward_calls"],
              "LM forwards")
    worst = 0.0
    for key in QUESTIONS:
        a, b = runs["shared"]["answers"][key], runs["independent"]["answers"][key]
        if a["type"] == "number":
            print(key, "shared", a["value"], "independent", b["value"])
            continue
        diff = max(abs(x - y) for x, y in zip(a["candidate_scores"], b["candidate_scores"]))
        worst = max(worst, diff)
        print(key, "max|shared-independent| =", f"{diff:.4f}", {k: round(v, 3) for k, v in a["probabilities"].items()})

    # Stock full forward (model(**inputs).logits) for the sequence question.
    image = read_image("examples/photos/cat.jpg")
    prompter = Prompter(engine.processor, "")
    from jev_photo.jev.schema import Question
    q = Question.model_validate(QUESTIONS["seq"])
    plan = prompter.plan(q)
    stock = []
    for target in plan.targets:
        inputs = engine.adapter.prepare(prompter.prefix + plan.suffix, image, extra_ids=target[:-1])
        with torch.inference_mode():
            logits = engine.adapter.model(**inputs).logits[0].float()
        n = len(target)
        stock.append(sequence_logprob(logits[-n:].cpu().numpy(), target))
    shared = runs["shared"]["answers"]["seq"]["candidate_scores"]
    stock_diff = max(abs(x - y) for x, y in zip(shared, stock))
    print("seq stock", [round(x, 3) for x in stock], "shared", [round(x, 3) for x in shared],
          f"max diff {stock_diff:.4f}")
    record = {"max_shared_vs_independent": worst, "max_shared_vs_stock_seq": stock_diff,
              "shared_metrics": runs["shared"]["metrics"], "independent_metrics": runs["independent"]["metrics"]}
    print(json.dumps(record, indent=1))
    from pathlib import Path
    Path("artifacts").mkdir(exist_ok=True)
    Path(f"artifacts/scoring-verification-{dtype}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
