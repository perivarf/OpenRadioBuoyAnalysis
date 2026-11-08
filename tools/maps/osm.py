"""OpenStreetMap basemaps and Web Mercator, used by mapplot.py and
script_lora_range.py.

Projection, tile download with a disk cache (~/.cache/orb_mapplot), the stitched
basemap, great-circle distance, scale bar and north arrow. Figures made from the
tiles must carry "© OpenStreetMap contributors".
"""

import math
import os
import socket
import urllib.error
import urllib.request

import numpy as np

# --- Tiles -------------------------------------------------------------------
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_PX = 256
# The OSM tile policy requires an identifiable User-Agent.
USER_AGENT = "ORB-mapplot/1.0 (per.ivar.faust@gmail.com)"
TILE_CACHE = os.path.expanduser("~/.cache/orb_mapplot")
TILE_TIMEOUT = 20            # s per tile
MAX_TILES = 36               # ceiling on tile count -> picks a zoom under this
MAX_ZOOM = 19                # OSM has no standard tiles above this

R_EARTH = 6378137.0          # Web Mercator sphere (EPSG:3857)


# --- Web Mercator ------------------------------------------------------------
def lonlat_to_merc(lon, lat):
    x = R_EARTH * np.radians(lon)
    y = R_EARTH * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def merc_per_metre(lat_deg):
    """Mercator units per metre on the ground at this latitude (1/cos(lat))."""
    return 1.0 / math.cos(math.radians(lat_deg))


def merc_world(zoom):
    """The width of the world in Mercator units per tile pixel at a given zoom."""
    return 2.0 * math.pi * R_EARTH / (TILE_PX * (2 ** zoom))


def merc_to_pixel(x, y, zoom):
    half = math.pi * R_EARTH
    scale = TILE_PX * (2 ** zoom) / (2.0 * half)
    return (x + half) * scale, (half - y) * scale


def pick_zoom(bbox, max_tiles=MAX_TILES, zmax=MAX_ZOOM):
    """The largest zoom where the bbox is covered by <= max_tiles tiles."""
    x0, y0, x1, y1 = bbox
    for z in range(zmax, 0, -1):
        px0, py1 = merc_to_pixel(x0, y0, z)     # y is flipped
        px1, py0 = merc_to_pixel(x1, y1, z)
        nx = int(px1 // TILE_PX) - int(px0 // TILE_PX) + 1
        ny = int(py1 // TILE_PX) - int(py0 // TILE_PX) + 1
        if nx * ny <= max_tiles:
            return z
    return 1


def prefer_ipv4():
    """Make getaddrinfo return IPv4 first (urllib would otherwise wait for a timeout
    per tile on networks without working IPv6). IPv6 stays as a fallback."""
    orig = socket.getaddrinfo

    def sorted_getaddrinfo(*args, **kwargs):
        res = orig(*args, **kwargs)
        return sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)

    socket.getaddrinfo = sorted_getaddrinfo


def fetch_tile(z, x, y, url_tmpl, cache_dir):
    """One tile, disk cached. Returns a PIL image, or None on failure."""
    from PIL import Image
    n = 2 ** z
    if not (0 <= x < n and 0 <= y < n):
        return None
    path = os.path.join(cache_dir, str(z), str(x), f"{y}.png")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(
            url_tmpl.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TILE_TIMEOUT) as resp:
            data = resp.read()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)                  # never half a tile in the cache
    return Image.open(path).convert("RGB")


def build_basemap(bbox, zoom, url_tmpl=TILE_URL, cache_dir=TILE_CACHE, quiet=False):
    """Stitch together the tiles covering bbox -> (numpy image, extent in Mercator).
    Returns (None, None) if the tiles cannot be fetched."""
    from PIL import Image
    x0, y0, x1, y1 = bbox
    px0, py1 = merc_to_pixel(x0, y0, zoom)
    px1, py0 = merc_to_pixel(x1, y1, zoom)
    tx0, tx1 = int(px0 // TILE_PX), int(px1 // TILE_PX)
    ty0, ty1 = int(py0 // TILE_PX), int(py1 // TILE_PX)
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    canvas = Image.new("RGB", (nx * TILE_PX, ny * TILE_PX), (238, 238, 233))
    n_net = 0
    try:
        for i, tx in enumerate(range(tx0, tx1 + 1)):
            for j, ty in enumerate(range(ty0, ty1 + 1)):
                cached = os.path.isfile(
                    os.path.join(cache_dir, str(zoom), str(tx), f"{ty}.png"))
                img = fetch_tile(zoom, tx, ty, url_tmpl, cache_dir)
                if img is not None:
                    canvas.paste(img, (i * TILE_PX, j * TILE_PX))
                    n_net += 0 if cached else 1
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"  WARNING: could not fetch map tiles ({exc}) - drawing without a basemap.")
        return None, None
    if not quiet:
        print(f"  tiles: {nx}x{ny} @ z{zoom} ({n_net} downloaded, "
              f"{nx * ny - n_net} from cache)")
    # Extent = the outer edges of the tile grid in Mercator units.
    step = merc_world(zoom) * TILE_PX
    half = math.pi * R_EARTH
    ex0 = -half + tx0 * step
    ex1 = -half + (tx1 + 1) * step
    ey1 = half - ty0 * step
    ey0 = half - (ty1 + 1) * step
    return np.asarray(canvas), (ex0, ex1, ey0, ey1)


# --- Distance ----------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))



# --- Decoration --------------------------------------------------------------
def nice_scale(span_m):
    """A tidy scale-bar length (1/2/5 series) at about 25 % of the map width."""
    target = span_m * 0.25
    exp = math.floor(math.log10(target))
    for mult in (1, 2, 5, 10):
        val = mult * 10 ** exp
        if val >= target:
            return val
    return 10 ** (exp + 1)


def halo(lw=3.0):
    """White outline that keeps black map decorations readable on any tile."""
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw, foreground="white"), pe.Normal()]


def draw_scalebar(ax, bbox, lat_mid, corner="bl", fontsize=7):
    """Scale bar in metres in corner "bl" (bottom left) or "br"."""
    mpm = merc_per_metre(lat_mid)
    w, h_ax = bbox[2] - bbox[0], bbox[3] - bbox[1]
    span_m = w / mpm
    bar_m = nice_scale(span_m)
    bar = bar_m * mpm
    x0 = (bbox[0] + 0.045 * w) if corner == "bl" else (bbox[2] - 0.045 * w - bar)
    y0 = bbox[1] + 0.045 * h_ax
    h = 0.008 * h_ax
    ax.plot([x0, x0 + bar], [y0, y0], "-", color="black", lw=1.8, zorder=8,
            path_effects=halo())
    for xx in (x0, x0 + bar):
        ax.plot([xx, xx], [y0 - h, y0 + h], "-", color="black", lw=1.8, zorder=8,
                path_effects=halo())
    # Label in line with the bar, on its inner side.
    label = f"{bar_m:.0f} m" if bar_m < 1000 else f"{bar_m / 1000:g} km"
    if corner == "bl":
        ax.text(x0 + bar + 0.012 * w, y0, label, fontsize=fontsize, ha="left",
                va="center", zorder=8, path_effects=halo())
    else:
        ax.text(x0 - 0.012 * w, y0, label, fontsize=fontsize, ha="right",
                va="center", zorder=8, path_effects=halo())


def draw_north(ax, bbox, corner="tl"):
    """North arrow in the top left ("tl") or top right corner."""
    frac = 0.05 if corner == "tl" else 0.95
    x = bbox[0] + frac * (bbox[2] - bbox[0])
    y0 = bbox[1] + 0.78 * (bbox[3] - bbox[1])
    y1 = y0 + 0.10 * (bbox[3] - bbox[1])
    ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                path_effects=halo()))
    ax.text(x, y1 + 0.012 * (bbox[3] - bbox[1]), "N", fontsize=8.5, weight="bold",
            ha="center", va="bottom", zorder=8, path_effects=halo())
