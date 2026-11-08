"""Crop the white margin around a saved figure to an equal padding on all sides.

Matplotlib leaves uneven margins when a figure is cut out with a fixed bbox or the
content does not fill the axes (a text table whose last column ends early). Stacked
in LaTeX at the same width, such figures then look shifted relative to each other.
"""


def trim_png(path, pad_px=20, threshold=250):
    """Crop path in place to its non-white content plus pad_px on every side.
    Pixels lighter than threshold (0-255, all channels) count as background."""
    from PIL import Image

    with Image.open(path) as im:
        im.load()
    mask = im.convert("L").point(lambda p: 255 if p < threshold else 0)
    box = mask.getbbox()
    if box is None:                      # blank image: leave it alone
        return
    x0, y0, x1, y1 = box
    box = (max(x0 - pad_px, 0), max(y0 - pad_px, 0),
           min(x1 + pad_px, im.width), min(y1 + pad_px, im.height))
    im.crop(box).save(path, dpi=im.info.get("dpi", (72, 72)))
