from PIL import Image

from jev_photo.edit.ops import Box, EditParams, Region, apply, apply_exposure, rotate_and_fill


def gradient(w=200, h=100):
    img = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), (x * 255 // w, y * 255 // h, 128))
    return img


def test_identity_params_change_nothing():
    img = gradient()
    assert apply(img, EditParams()).tobytes() == img.tobytes()


def test_exposure_gamma_keeps_black_and_white():
    img = Image.new("RGB", (3, 1))
    for x, v in enumerate((0, 128, 255)):
        img.putpixel((x, 0), (v, v, v))
    out = [apply_exposure(img, 1.5).getpixel((x, 0)) for x in range(3)]
    assert out[0] == (0, 0, 0) and out[2] == (255, 255, 255) and out[1][0] > 128


def test_crop_uses_fractions():
    out = apply(gradient(), EditParams(crop=Box(0.25, 0.1, 0.75, 0.9)))
    assert out.size == (100, 80)


def test_rotation_leaves_no_empty_corners():
    img = Image.new("RGB", (300, 200), (200, 50, 50))
    out = rotate_and_fill(img, 10)
    assert out.width < 300 and out.height < 200
    for corner in [(0, 0), (out.width - 1, 0), (0, out.height - 1), (out.width - 1, out.height - 1)]:
        assert out.getpixel(corner)[0] > 150


def test_region_inside_only_touches_box():
    img = Image.new("RGB", (200, 200), (100, 100, 100))
    out = apply(img, EditParams(region=Region("darken", Box(0.5, 0.5, 1.0, 1.0), "inside")))
    assert out.getpixel((10, 10)) == (100, 100, 100)
    assert out.getpixel((190, 190))[0] < 80
