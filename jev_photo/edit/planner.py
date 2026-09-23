"""The typed questions the editor asks. Only questions; no model output is parsed as text."""

AUTO_REQUEST = ("Automatically enhance this photo like a careful professional photo editor. Fix only clearly "
                "visible problems: under- or over-exposure, flat contrast, dull or oversaturated colors, a color "
                "cast, softness, a tilted horizon, distracting empty borders. Leave aspects that already look good "
                "unchanged.")


def context(request):
    return {"edit_request": request or AUTO_REQUEST,
            "task": "Decide how to edit the photo so that the result fulfils the edit request."}


# name -> (gate question, number question: instructions, range, resolution)
# Ranges are chosen so that K=10 interval boundaries land on round values
# (e.g. rotation [-50, 50) -> 10 degree intervals, then 1, then 0.5), which the
# model reads far more reliably than boundaries like -36 or 0.65.
GLOBAL = {
    "exposure": (
        "Does fulfilling the edit request require changing the overall brightness / exposure of the photo?",
        "What overall brightness multiplier fulfils the edit request? 1.0 keeps the current brightness. 1.2 is "
        "slightly brighter, 1.5 is clearly brighter, 2.0 doubles the brightness (for a badly underexposed, dark "
        "photo); 0.8 is slightly darker, 0.5 halves it (for a badly overexposed photo).",
        ["0", "2"], "0.05"),
    "contrast": (
        "Does fulfilling the edit request require changing the contrast of the photo?",
        "What contrast multiplier fulfils the edit request? 1.0 keeps the current contrast. 1.2 is slightly "
        "punchier, 1.5 is strong contrast (for a flat, hazy or washed-out photo); 0.8 is slightly softer.",
        ["0", "2"], "0.05"),
    "saturation": (
        "Does fulfilling the edit request require changing color saturation / vividness (including black and white)?",
        "What saturation multiplier fulfils the edit request? 1.0 keeps the current colors; 0 is black and white, "
        "0.6 is muted, 1.3 is more vivid, 1.8 is very vivid.",
        ["0", "2"], "0.05"),
    "warmth": (
        "Does fulfilling the edit request require changing the color temperature (warmer / cooler) or removing a "
        "color cast?",
        "What color temperature shift fulfils the edit request? 0 keeps it unchanged; +0.3 is slightly warmer "
        "(more orange), +0.8 is much warmer; -0.3 is slightly cooler (more blue), -0.8 is much cooler.",
        ["-1", "1"], "0.05"),
    "sharpness": (
        "Does fulfilling the edit request require sharpening or softening the photo?",
        "What sharpness factor fulfils the edit request? 1.0 keeps it unchanged; 1.6 is slightly sharper, "
        "2.4 is much sharper, 0.4 is softer.",
        ["0", "4"], "0.1"),
    "rotation": (
        "Does fulfilling the edit request require rotating the photo, for example to straighten a tilted horizon "
        "or because the request asks for a rotation?",
        "By how many degrees should the photo be rotated to fulfil the edit request? Positive values rotate "
        "counter-clockwise, negative values rotate clockwise, 0 means no rotation. To straighten a tilt, rotate "
        "opposite to the tilt.",
        ["-50", "50"], "0.5"),
}

GATE_CROP = ("Does fulfilling the edit request require cropping the photo (keeping only a rectangular part of it, "
             "e.g. removing borders, zooming in on the subject, or improving the composition)?")
GATE_REGION = ("Does the edit request ask for a local edit of only one area of the photo, such as blurring the "
               "background, pixelating a face or a license plate, or brightening / darkening / desaturating a "
               "specific object?")

REGION_OP = {
    "type": "choice",
    "instructions": "Which local edit does the request ask for?",
    "criteria": {
        "blur": "Blur the area (e.g. a blurred background)",
        "pixelate": "Pixelate / mosaic the area (e.g. hide a face or a plate)",
        "brighten": "Brighten the area",
        "darken": "Darken the area",
        "desaturate": "Remove the color of the area (make it black and white)",
    },
}
REGION_WHERE = {
    "type": "choice",
    "instructions": "Should the local edit apply inside the target object's rectangle, or to everything outside "
                    "it? For example 'blur the background' keeps the subject sharp, so it applies outside the "
                    "subject.",
    "criteria": {"inside": "Inside the rectangle around the object named by the request",
                 "outside": "Everything outside the rectangle around the main subject (the background)"},
}

# `box` questions: what to locate, phrased with the request so the model knows which object is meant.
CROP_WHAT = "the part of the photo that should be kept when cropping to fulfil this edit request: {request!r}"
REGION_WHAT = {"inside": "the object or area that this local edit applies to: {request!r}",
               "outside": "the main subject that must stay untouched by this edit (the edit applies to everything "
                          "around it): {request!r}"}


def gate_questions():
    questions = {f"gate_{name}": {"type": "noul", "instructions": gate} for name, (gate, *_rest) in GLOBAL.items()}
    questions["gate_crop"] = {"type": "noul", "instructions": GATE_CROP}
    questions["gate_region"] = {"type": "noul", "instructions": GATE_REGION}
    return questions


def global_number_questions(names, decoding="digits"):
    # digits beat the lettered interval tree 14/14 vs 10/14 on benchmarks/number_decoding.py.
    return {name: {"type": "number", "instructions": GLOBAL[name][1], "range": GLOBAL[name][2],
                   "resolution": GLOBAL[name][3], "decoding": decoding} for name in names}


def box_question(what, request):
    return {"type": "box", "instructions": what.format(request=request)}
