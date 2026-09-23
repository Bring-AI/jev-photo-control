"""Deterministic photo operations. The model never touches pixels; it only picks numbers."""
import math
from dataclasses import asdict, dataclass, field

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

REGION_OPS = ("blur", "pixelate", "brighten", "darken", "desaturate")


@dataclass
class Box:
    """Fractions of image width/height, (x0, y0) top-left, (x1, y1) bottom-right."""
    x0: float
    y0: float
    x1: float
    y1: float

    def valid(self, min_size=0.05):
        return 0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1 and \
            self.x1 - self.x0 >= min_size and self.y1 - self.y0 >= min_size

    def pixels(self, width, height):
        return (round(self.x0 * width), round(self.y0 * height), round(self.x1 * width), round(self.y1 * height))


@dataclass
class Region:
    op: str
    box: Box
    where: str = "inside"  # inside the box, or everything outside the subject in it (e.g. blur background)


@dataclass
class EditParams:
    exposure: float = 1.0      # brightness multiplier
    contrast: float = 1.0
    saturation: float = 1.0
    warmth: float = 0.0        # -1 cooler .. +1 warmer
    sharpness: float = 1.0     # PIL sharpness factor, 1 = unchanged
    rotation: float = 0.0      # degrees, counter-clockwise
    crop: Box | None = None
    region: Region | None = None
    notes: list[str] = field(default_factory=list)

    def dict(self):
        return asdict(self)


def subject_mask(image, box, work_side=512):
    """GrabCut initialised with the model's box: separates the subject from the background
    inside the rectangle. Classic, deterministic CV; falls back to the rectangle."""
    import cv2
    import numpy as np
    small = image.copy()
    small.thumbnail((work_side, work_side))
    arr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = box.pixels(*small.size)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    mask = np.zeros(arr.shape[:2], np.uint8)
    try:
        cv2.grabCut(arr, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if fg.sum() / 255 < 0.05 * rect[2] * rect[3]:
        return None
    return Image.fromarray(fg, "L").resize(image.size, Image.BILINEAR)


def region_mask(image, box, where, feather):
    size = image.size
    mask = subject_mask(image, box) if where == "outside" else None
    if mask is None:
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle(box.pixels(*size), fill=255)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    if where == "outside":
        mask = Image.eval(mask, lambda v: 255 - v)
    return mask


def apply_region(image, region):
    if region.op not in REGION_OPS:
        raise ValueError(f"unknown region op {region.op}")
    w, h = image.size
    scale = max(w, h)
    if region.op == "blur":
        edited = image.filter(ImageFilter.GaussianBlur(scale / 60))
    elif region.op == "pixelate":
        block = max(4, scale // 48)
        edited = image.resize((max(1, w // block), max(1, h // block)), Image.BILINEAR).resize((w, h), Image.NEAREST)
    elif region.op == "brighten":
        edited = ImageEnhance.Brightness(image).enhance(1.35)
    elif region.op == "darken":
        edited = ImageEnhance.Brightness(image).enhance(0.6)
    else:
        edited = ImageEnhance.Color(image).enhance(0.0)
    return Image.composite(edited, image, region_mask(image, region.box, region.where, scale / 300))


def rotate_and_fill(image, degrees):
    """Rotate counter-clockwise, then crop the largest axis-aligned rectangle without empty corners."""
    if abs(degrees) < 1e-6:
        return image
    w, h = image.size
    rotated = image.rotate(degrees, resample=Image.BICUBIC, expand=True)
    a = math.radians(abs(degrees))
    # Largest inscribed rectangle of a rotated w x h rectangle.
    long_side, short_side = max(w, h), min(w, h)
    sin_a, cos_a = math.sin(a), math.cos(a)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        cw, ch = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        cw, ch = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    rw, rh = rotated.size
    left, top = (rw - cw) / 2, (rh - ch) / 2
    return rotated.crop((round(left), round(top), round(left + cw), round(top + ch)))


def apply_exposure(image, factor):
    """Gamma curve: factor 1.5 brightens midtones by roughly that much while black and white stay put,
    so bright subjects do not clip the way a plain multiply does."""
    if abs(factor - 1) < 1e-6:
        return image
    gamma = 1 / max(factor, 0.05)
    lut = [round(255 * (v / 255) ** gamma) for v in range(256)]
    return image.point(lut * 3)


def apply_warmth(image, warmth):
    if abs(warmth) < 1e-6:
        return image
    r, g, b = image.split()
    k = 0.18 * warmth
    r = r.point(lambda v: min(255, round(v * (1 + k))))
    b = b.point(lambda v: min(255, round(v * (1 - k))))
    return Image.merge("RGB", (r, g, b))


def apply(image, params: EditParams):
    """Order: local region (coordinates refer to the input) -> crop -> rotation -> global tone/color."""
    image = image.convert("RGB")
    if params.region is not None:
        image = apply_region(image, params.region)
    if params.crop is not None:
        image = image.crop(params.crop.pixels(*image.size))
    image = rotate_and_fill(image, params.rotation)
    image = apply_exposure(image, params.exposure)
    if params.contrast != 1:
        image = ImageEnhance.Contrast(image).enhance(params.contrast)
    if params.saturation != 1:
        image = ImageEnhance.Color(image).enhance(params.saturation)
    image = apply_warmth(image, params.warmth)
    if params.sharpness != 1:
        image = ImageEnhance.Sharpness(image).enhance(params.sharpness)
    return image
