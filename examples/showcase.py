"""Run the fixed showcase prompts against a running server and write docs/showcase/.

The prompts were fixed before the first run; every result is kept, none are re-rolled.

    python -m examples.showcase [http://127.0.0.1:8790]
"""
import base64
import io
import json
import sys
from pathlib import Path

import httpx
from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8790"
OUT = Path("docs/showcase")
THUMB = 360

PROMPTS = {
    "cat": ["blur the background", "crop tightly around the cat's face", "make it black and white"],
    "taxi": ["", "remove the orange color cast so it looks natural",
             "blur everything except the yellow taxi in front"],
    "lake": ["make it warmer, like golden hour", "crop to the snowy mountain and its reflection",
             "make the colors more vivid"],
    "beach": ["", "darken the bright sky", "rotate 3 degrees counter-clockwise"],
}


def thumb(image, path):
    image = image.copy()
    image.thumbnail((THUMB, THUMB))
    image.save(path, quality=85)


def summary(result):
    p, parts = result["params"], []
    names = {"exposure": "exposure ×{}", "contrast": "contrast ×{}", "saturation": "saturation ×{}",
             "warmth": "warmth {:+}", "sharpness": "sharpness ×{}", "rotation": "rotate {:+}°"}
    neutral = {"warmth": 0.0, "rotation": 0.0}
    for key, template in names.items():
        if p[key] != neutral.get(key, 1.0):
            parts.append(template.format(round(p[key], 3)))
    if p["crop"]:
        parts.append("crop box " + str(result["boxes"]["crop"]["bbox_2d"]))
    if p["region"]:
        r = p["region"]
        where = "everything but" if r["where"] == "outside" else "inside"
        parts.append(f"{r['op']} {where} box {result['boxes']['region']['bbox_2d']}")
    parts += p["notes"]
    return "<br>".join(parts) if parts else "no edit"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    for photo, prompts in PROMPTS.items():
        source = Image.open(f"examples/photos/{photo}.jpg")
        thumb(source, OUT / f"{photo}.jpg")
        data = "data:image/jpeg;base64," + base64.b64encode(Path(f"examples/photos/{photo}.jpg").read_bytes()).decode()
        for i, prompt in enumerate(prompts, 1):
            r = httpx.post(BASE + "/v1/edit", json={"image": data, "instruction": prompt}, timeout=600)
            r.raise_for_status()
            result = r.json()
            edited = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[1])))
            name = f"{photo}-{i}.jpg"
            thumb(edited, OUT / name)
            gates = {k: round(v, 3) for k, v in result["gates"].items()}
            records.append({"photo": photo, "prompt": prompt, "gates": gates, "params": result["params"],
                            "boxes": result["boxes"], "amounts": result["amounts"],
                            "total_ms": round(result["metrics"]["total_ms"]),
                            "sequential_rounds": result["metrics"]["sequential_rounds"]})
            label = f"“{prompt}”" if prompt else "*(empty: auto-enhance)*"
            rows.append(f'| <img src="docs/showcase/{photo}.jpg" width="220"> | {label} | {summary(result)} | '
                        f'<img src="docs/showcase/{name}" width="220"> |')
            print(photo, repr(prompt), "->", summary(result).replace("<br>", "; "),
                  f"({result['metrics']['total_ms']:.0f} ms)")
    (OUT / "results.json").write_text(json.dumps(records, indent=1))
    table = "| Input | Prompt | What the model decided | Output |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    (OUT / "table.md").write_text(table)


if __name__ == "__main__":
    main()
