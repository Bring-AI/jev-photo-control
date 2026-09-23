"""Instruction (or auto) photo editing driven by Jev judgments.

1. gate:    one batch of noul questions on the clean photo - which operations are needed?
2. amounts: number questions for the gated global parameters (+ region op choices),
            reusing the SAME image prefill as step 1.
3. boxes:   `box` questions (native bbox_2d digits) for crop / region rectangles,
            still on the same prefill.
4. apply:   deterministic PIL operations (ops.apply) on the full-resolution photo.

The model only ever sees a <=1024 px view; boxes are fractions, so they transfer.
"""
import time

from . import planner
from ..jev.preprocessing import read_image
from .ops import Box, EditParams, Region, apply

GATE_THRESHOLD = 0.5
MAX_OUTPUT_SIDE = 4096


def edit(engine, image, request="", *, mode="shared", threshold=GATE_THRESHOLD):
    with engine.lock:
        engine.adapter.reset()
        started = time.perf_counter()
        ctx = planner.context(request.strip())
        steps, timings = {}, {}
        image = image.convert("RGB")
        image.thumbnail((MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE))
        view = read_image(image)  # copy, <=1024 px

        t = time.perf_counter()
        clean = engine.session(view, ctx, mode)
        gates, _ = clean.judge(planner.gate_questions())
        steps["gates"] = {k.removeprefix("gate_"): v["noul"] for k, v in gates.items()}
        active = [name for name in planner.GLOBAL if steps["gates"][name] > threshold]
        do_crop, do_region = steps["gates"]["crop"] > threshold, steps["gates"]["region"] > threshold
        timings["gate_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        questions = planner.global_number_questions(active)
        if do_region:
            questions["region_op"], questions["region_where"] = planner.REGION_OP, planner.REGION_WHERE
        amounts, rounds = clean.judge(questions) if questions else ({}, 0)
        steps["amounts"] = amounts
        timings["amounts_ms"] = (time.perf_counter() - t) * 1000

        params = EditParams(**{name: amounts[name]["value"] for name in active})
        if do_crop or do_region:
            t = time.perf_counter()
            where = amounts["region_where"]["choice"] if do_region else None
            box_q = {}
            if do_crop:
                box_q["crop"] = planner.box_question(planner.CROP_WHAT, ctx["edit_request"])
            if do_region:
                box_q["region"] = planner.box_question(planner.REGION_WHAT[where], ctx["edit_request"])
            boxes, box_steps = clean.judge(box_q)
            rounds += box_steps
            steps["boxes"] = boxes
            timings["boxes_ms"] = (time.perf_counter() - t) * 1000
            if do_crop:
                crop = Box(*boxes["crop"]["box"])
                if crop.valid(0.1):
                    params.crop = crop
                else:
                    params.notes.append(f"crop box {crop} rejected as degenerate")
            if do_region:
                box = Box(*boxes["region"]["box"])
                if box.valid():
                    params.region = Region(amounts["region_op"]["choice"], box, where)
                else:
                    params.notes.append(f"region box {box} rejected as degenerate")

        t = time.perf_counter()
        result = apply(image, params)
        timings["apply_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        metrics = {**engine.adapter.timings.dict(), **timings, "sequential_rounds": rounds, "device": engine.device,
                   "peak_gpu_memory_gb": engine.adapter.peak_memory_gb()}
        return {"image": result, "params": params, "request": ctx["edit_request"], "auto": not request.strip(),
                "steps": steps, "metrics": metrics}
