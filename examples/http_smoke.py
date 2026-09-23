"""Smoke-test a running server: python -m examples.http_smoke [http://127.0.0.1:8790]"""
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790"
IMAGE = "data:image/jpeg;base64," + base64.b64encode(open("examples/photos/cat.jpg", "rb").read()).decode()

JUDGE = {
    "image": IMAGE,
    "questions": {
        "animal": {"type": "choice", "instructions": "Which animal is the main subject?",
                   "criteria": {"cat": "A cat", "dog": "A dog", "other": "Something else"}},
        "eyes": {"type": "noul", "instructions": "Does the animal have green eyes?"},
        "legs": {"type": "number", "instructions": "How many of the animal's legs are visible?",
                 "range": [0, 10], "resolution": 1},
        "area": {"type": "number", "instructions": "What fraction of the image area is covered by the animal?",
                "range": [0, 1], "resolution": "0.05", "decoding": "digits"},
        "head": {"type": "box", "instructions": "the cat's head"},
    },
}


def post(path, body):
    t = time.perf_counter()
    r = httpx.post(BASE + path, json=body, timeout=300)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t) * 1000


def main():
    print(httpx.get(BASE + "/health").json())
    out, ms = post("/v1/judge", JUDGE)
    print(f"judge {ms:.0f} ms")
    for k, a in out["answers"].items():
        print(" ", k, a.get("choice", a.get("noul", a.get("value", a.get("bbox_2d")))))
    r = httpx.post(BASE + "/v1/judge", json={**JUDGE, "questions": {"x": {"type": "number", "instructions": "x",
                                                                          "range": [0, 1], "resolution": "0.3"}}})
    print("invalid grid ->", r.status_code, r.json()["detail"][:80] if r.status_code == 422 else r.text[:80])
    requests = ["make it a bit warmer", "blur the background"]
    with ThreadPoolExecutor(2) as pool:
        t = time.perf_counter()
        results = list(pool.map(lambda q: post("/v1/edit", {"image": IMAGE, "instruction": q}), requests))
        wall = (time.perf_counter() - t) * 1000
    for q, (res, ms) in zip(requests, results):
        m = res["metrics"]
        print(f"edit {q!r}: {ms:.0f} ms on {m['device']}, params",
              {k: v for k, v in res["params"].items() if v not in (1.0, 0.0, None, [])})
    print(f"two concurrent edits wall time {wall:.0f} ms")


if __name__ == "__main__":
    main()
