"""Compare `number` decodings (interval tree with K=10 / K=4 vs native digits) on edit requests.

The cases and accepted ranges are hand-written (see README caveats): a rough check, not an accuracy study.

    python -m benchmarks.number_decoding cuda:1
"""
import json
import sys
import time
from pathlib import Path

from jev_photo.edit import planner
from jev_photo.jev import Engine
from jev_photo.jev.preprocessing import read_image

# (edit request, parameter, accepted [low, high])
CASES = [
    ("rotate 10 degrees clockwise", "rotation", [-10.5, -9.5]),
    ("rotate 30 degrees counter-clockwise", "rotation", [29.5, 30.5]),
    ("rotate the photo 5 degrees to the left (counter-clockwise)", "rotation", [4.5, 5.5]),
    ("make it a bit warmer", "warmth", [0.1, 0.45]),
    ("make it much cooler and bluer", "warmth", [-1, -0.45]),
    ("set the color temperature shift to -0.4", "warmth", [-0.4, -0.4]),
    ("make it much brighter", "exposure", [1.35, 2]),
    ("make it slightly darker", "exposure", [0.7, 0.95]),
    ("set the brightness multiplier to 1.35", "exposure", [1.35, 1.35]),
    ("convert to black and white", "saturation", [0, 0.05]),
    ("make the colors very vivid", "saturation", [1.4, 2]),
    ("slightly reduce the contrast", "contrast", [0.7, 0.95]),
    ("increase the contrast strongly", "contrast", [1.3, 2]),
    ("sharpen the photo a lot", "sharpness", [2, 4]),
]


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    engine = Engine(device=device)
    image = read_image("examples/photos/cat.jpg")
    variants = {"interval_k10": {"decoding": "interval", "branching": 10},
                "interval_k4": {"decoding": "interval", "branching": 4},
                "digits": {"decoding": "digits"}}
    rows, hits = [], dict.fromkeys(variants, 0)
    for request, param, (low, high) in CASES:
        base = planner.global_number_questions([param], "interval")[param]
        with engine.lock:
            session = engine.session(image, planner.context(request))
            got = {}
            for variant, options in variants.items():
                t = time.perf_counter()
                answers, rounds = session.judge({param: {**base, **options}})
                got[variant] = (answers[param]["value"], rounds, (time.perf_counter() - t) * 1000,
                                answers[param]["trace"])
        row = {"request": request, "param": param, "accepted": [low, high]}
        for variant, (value, rounds, ms, trace) in got.items():
            ok = low - 1e-9 <= value <= high + 1e-9
            hits[variant] += ok
            row[variant] = {"value": value, "ok": ok, "rounds": rounds, "ms": round(ms), "trace": trace}
        rows.append(row)
        print(f"{request[:40]:40s}", "  ".join(f"{v}={row[v]['value']:6.2f} {'OK ' if row[v]['ok'] else 'bad'}"
                                               f"({row[v]['rounds']}r)" for v in variants))
    print({k: f"{v}/{len(CASES)}" for k, v in hits.items()})
    for v in variants:
        print(v, "mean ms", round(sum(r[v]["ms"] for r in rows) / len(rows)))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/number-decoding.json").write_text(json.dumps({"hits": hits, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
