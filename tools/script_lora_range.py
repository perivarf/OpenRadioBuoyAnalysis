#!/usr/bin/env python3
"""Map figure for the LoRa range test: base station + the drifter positions where
a message got through, on OpenStreetMap tiles.

Each drifter position is drawn as a numbered badge with the measurement number, the
great-circle distance to the base station and whether the antenna was held above
the surface or the buoy floated in the water, and a thin dashed line runs from the
base station out to the point so the distance being labelled is unambiguous. The
two cases also differ in colour, with the key in the legend - the range from a
floating buoy is a different number from the range with the antenna held clear, and
the figure must not let the two be read as one series.

The positions are the constants BASE and POINTS below, and IN_WATER says which of
them were logged afloat - the test was a handful of fixed points, not a logged
track, so there is no file to read. --csv reads them from a file instead (columns
nr,lat,lon[,in_water] with a header, or just lat,lon), and --in-water overrides
which numbers count as afloat.

The tiles come from tile.openstreetmap.org and are cached on disk in
~/.cache/orb_mapplot by tools/maps/osm.py - the SAME code and cache mapplot.py uses, so tiles already fetched for
a session map in this area are reused and nothing is downloaded twice. Keep
"© OpenStreetMap contributors" on any figure made from them; the attribution line
below the map is there for that reason.

Usage:
    python3 tools/script_lora_range.py        # -> lora_range/lora_range.png
    python3 tools/script_lora_range.py -o range.pdf --dpi 300
    python3 tools/script_lora_range.py --csv points.csv
    python3 tools/script_lora_range.py --in-water 6        # which points were afloat
    python3 tools/script_lora_range.py --no-map            # without tile download (offline)
    python3 tools/script_lora_range.py --show
"""

import argparse
import math
import os
import sys

import numpy as np

# Projection, tiles and the scale bar/north arrow are tools/maps/osm.py, shared with
# mapplot.py.
import _paths                                       # noqa: E402,F401
from osm import (TILE_CACHE, TILE_URL, build_basemap,  # noqa: E402
                 draw_north, draw_scalebar, halo, haversine, lonlat_to_merc,
                 merc_per_metre, pick_zoom, prefer_ipv4)

# --- The measurement ---------------------------------------------------------
BASE = (59.048100, 11.055689)            # base station (lat, lon)

POINTS = [                               # nr, lat, lon - successful messages
    (1, 59.047944, 11.065292), # 18:26
    (2, 59.046872, 11.079422), # 18:32
    (3, 59.047006, 11.083911), # 18:36
    (4, 59.046781, 11.090108), # 18:44
    (5, 59.047722, 11.099486), # 18:52
    (6, 59.047000, 11.089397), # 19:01
]

# The antenna sat just above the surface for every point but one, where the buoy
# floated as it will in use - a difference worth seeing on the map, because the
# last half metre of water is what costs the link its range.
IN_WATER = {6}                           # measurement numbers logged afloat

# --- Map extent --------------------------------------------------------------
PAD_FRAC = 0.14              # margin around the point bbox, fraction of the longest side
MIN_ASPECT = 0.62            # map height / width floor: the points lie almost on an
                             # east-west line, and the raw bbox would be a 20:1 sliver
MIN_SPAN_M = 300.0           # smallest map extent

# --- Colours -----------------------------------------------------------------
C_BASE = "#d73027"
C_POINT = "#1f4e9c"          # antenna above the surface
C_WATER = "#0b7a6b"          # buoy afloat
C_LINK = "#1f4e9c"


# --- Geometry ----------------------------------------------------------------
def bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing, degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


# --- Label placement ---------------------------------------------------------
# A label is placed by trying candidate spots around its point and keeping the one
# that scores best. The weights are in POINTS, not pixels, so --dpi does not change
# where the labels land - at dpi 300 every box and every gap is the same size on
# paper as at dpi 100, and only a dpi-independent score keeps the layout that way.
W_OVERLAP = 1 / 20.0         # per pt^2 of overlap with a label, marker or legend
W_OUTSIDE = 1 / 10.0         # per pt^2 falling outside the map frame - worse, a
                             # label half off the map cannot be read at all
W_CROSS = 14.0               # per base->point link the box lies across
W_FAR = 1 / 5.0              # per pt away from its own point: the tie-breaker that
                             # keeps a label next to what it labels
LABEL_PAD_PT = 2.5           # breathing room around each box, on top of its own pad
LEADER_MIN_PT = 26.0         # a label pushed further than this gets a leader line,
                             # or nothing says which point it belongs to
MARKER_R_PT = 7.0            # the numbered badge, as an obstacle


def label_candidates(r_min=15.0, r_max=95.0, n_r=6, n_a=12):
    """Offsets to try, in points: n_a directions from straight up, n_r rings out.
    Ordered nearest ring first, so an unobstructed label stays close to its point."""
    out = []
    for k in range(n_r):
        r = r_min + (r_max - r_min) * k / (n_r - 1)
        for i in range(n_a):
            a = math.radians(90.0 - i * 360.0 / n_a)
            out.append((r * math.cos(a), r * math.sin(a), r))
    return out


def _overlap(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0.0 and dy > 0.0 else 0.0


def _outside(box, frame):
    return (box[2] - box[0]) * (box[3] - box[1]) - _overlap(box, frame)


def _box_pt(bb, s, pad=0.0):
    """A matplotlib display bbox -> (x0, y0, x1, y1) in points, grown by pad."""
    return (bb.x0 * s - pad, bb.y0 * s - pad, bb.x1 * s + pad, bb.y1 * s + pad)


def place_labels(fig, ax, anns, anchors, obstacles, seg_xy, seg_id):
    """Move each label to the best free spot and return the offsets actually used.

    anns are annotations already created at their own point with the final text;
    anchors holds each one's point in display pixels. The crowded points are placed
    FIRST: a label that has only one free spot must claim it before a label with the
    whole fjord to choose from takes it and leaves nothing."""
    fig.canvas.draw()                      # renderer + transforms must exist first
    rend = fig.canvas.get_renderer()
    s = 72.0 / fig.dpi                     # display px -> points
    frame = _box_pt(ax.get_window_extent(rend), s)
    taken = list(obstacles)
    seg = seg_xy * s if len(seg_xy) else seg_xy
    cands = label_candidates()

    # Most crowded first: how many other points sit within a label's reach.
    near = [int(np.count_nonzero(
        np.hypot(anchors[:, 0] - ax_, anchors[:, 1] - ay_) < 150.0 / s) - 1)
        for ax_, ay_ in anchors]
    order = sorted(range(len(anns)), key=lambda i: (-near[i], i))

    offsets = [(0.0, 0.0)] * len(anns)
    for i in order:
        ann = anns[i]
        best, best_score, best_r = (0.0, 16.0), None, 0.0
        for dx, dy, r in cands:
            ann.xyann = (dx, dy)
            ann.set_ha("left" if dx > 1.0 else "right" if dx < -1.0 else "center")
            ann.set_va("bottom" if dy > 1.0 else "top" if dy < -1.0 else "center")
            box = _box_pt(ann.get_window_extent(rend), s, LABEL_PAD_PT)
            score = (W_FAR * r
                     + W_OUTSIDE * _outside(box, frame)
                     + W_OVERLAP * sum(_overlap(box, t) for t in taken))
            if len(seg):
                hit = ((seg[:, 0] > box[0]) & (seg[:, 0] < box[2]) &
                       (seg[:, 1] > box[1]) & (seg[:, 1] < box[3]))
                score += W_CROSS * len(np.unique(seg_id[hit]))
            if best_score is None or score < best_score:
                best, best_score, best_r = (dx, dy), score, r
            if score <= W_FAR * r:         # free spot on the nearest ring: done
                break
        dx, dy = best
        ann.xyann = (dx, dy)
        ann.set_ha("left" if dx > 1.0 else "right" if dx < -1.0 else "center")
        ann.set_va("bottom" if dy > 1.0 else "top" if dy < -1.0 else "center")
        taken.append(_box_pt(ann.get_window_extent(rend), s, LABEL_PAD_PT))
        offsets[i] = (dx, dy, best_r)
    return offsets


# --- Extent ------------------------------------------------------------------
def map_bbox(xs, ys, lat_mid):
    """Padded Mercator bbox around all the points, widened to MIN_ASPECT.

    The five points lie nearly on an east-west line, so the raw bbox is a sliver a
    few metres tall: the vertical extent is grown about the centre until the map is
    a sane shape, and both axes are kept at least MIN_SPAN_M across."""
    mpm = merc_per_metre(lat_mid)
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pad = PAD_FRAC * max(x1 - x0, y1 - y0, MIN_SPAN_M * mpm)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    # x first, then y against the FINAL width - the y floor depends on it.
    if x1 - x0 < MIN_SPAN_M * mpm:
        cx, half = 0.5 * (x0 + x1), 0.5 * MIN_SPAN_M * mpm
        x0, x1 = cx - half, cx + half
    y_floor = max(MIN_SPAN_M * mpm, MIN_ASPECT * (x1 - x0))
    if y1 - y0 < y_floor:
        cy, half = 0.5 * (y0 + y1), 0.5 * y_floor
        y0, y1 = cy - half, cy + half
    return x0, y0, x1, y1


# --- Figure ------------------------------------------------------------------
def make_figure(base, points, water, args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_b, lon_b = base
    lats = np.array([lat_b] + [p[1] for p in points])
    lons = np.array([lon_b] + [p[2] for p in points])
    xs, ys = lonlat_to_merc(lons, lats)
    lat_mid = float(np.mean(lats))

    dists = [haversine(lat_b, lon_b, p[1], p[2]) for p in points]
    brgs = [bearing(lat_b, lon_b, p[1], p[2]) for p in points]

    bbox = map_bbox(xs, ys, lat_mid)
    width = 10.0
    height = width * (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])

    # The figure is the map plus a strip at the top for the heading and one at the
    # bottom for the attribution; the map axes itself keeps the bbox aspect so the
    # tiles are not stretched.
    fig = plt.figure(figsize=(width + 0.6, height + 1.35))
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([0.3 / fw, 0.55 / fh, width / fw, height / fh])

    if not args.no_map:
        prefer_ipv4()
        img, extent = build_basemap(bbox, pick_zoom(bbox), args.tile_url,
                                    args.tile_cache)
        if img is not None:
            ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear",
                      zorder=0)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    xb, yb = xs[0], ys[0]

    # Base -> point links first, so the markers sit on top of their own lines. The
    # links are also sampled here, in display pixels, so the label placement can
    # see them: a box across a dashed line is not a collision, but it is untidy,
    # and the sampled points let the score prefer a spot that is clear of them.
    seg_xy, seg_id = [], []
    for k, ((nr, lat, lon), d) in enumerate(zip(points, dists)):
        x, y = lonlat_to_merc(lon, lat)
        ax.plot([xb, x], [yb, y], "-", color=C_LINK, lw=1.0, alpha=0.55,
                dashes=(5, 3), zorder=3)
        t = np.linspace(0.0, 1.0, 400)
        seg_xy.append(np.column_stack([xb + t * (x - xb), yb + t * (y - yb)]))
        seg_id.append(np.full(t.size, k))
    seg_xy = ax.transData.transform(np.vstack(seg_xy))
    seg_id = np.concatenate(seg_id)

    # The markers and the base station go down first; the labels are created at
    # their points and then moved to a free spot by place_labels() below.
    anns, anchors, cols, obstacles = [], [], [], []
    for (nr, lat, lon), d in zip(points, dists):
        x, y = lonlat_to_merc(lon, lat)
        afloat = nr in water
        col = C_WATER if afloat else C_POINT
        ax.plot([x], [y], marker="o", ms=9.5, mfc=col, mec="white", mew=1.4,
                ls="none", zorder=6)
        ax.text(x, y, str(nr), fontsize=7.5, weight="bold", color="white",
                ha="center", va="center", zorder=7)
        anns.append(ax.annotate(
            f"#{nr}\n{d:.0f} m\n{'In water' if afloat else 'Above water'}",
            xy=(x, y), xytext=(0, 16), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color="black", zorder=7,
            path_effects=halo(2.6),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col,
                      lw=1.1 if afloat else 0.7, alpha=0.88)))
        anchors.append(ax.transData.transform((x, y)))
        cols.append(col)
    anchors = np.array(anchors)

    ax.plot([xb], [yb], marker="*", ms=20, mfc=C_BASE, mec="white", mew=1.3,
            ls="none", zorder=6, label="Base station")
    base_ann = ax.annotate("Base station", xy=(xb, yb), xytext=(0, -17),
                           textcoords="offset points", ha="center", va="top",
                           fontsize=9, weight="bold", color=C_BASE, zorder=7,
                           path_effects=halo(2.6))

    draw_scalebar(ax, (bbox[0], bbox[1], bbox[2], bbox[3]), lat_mid, fontsize=7.5)
    draw_north(ax, (bbox[0], bbox[1], bbox[2], bbox[3]))

    # The legend carries the colour key: the label at each point says the same
    # thing in words, but the key is what makes the one green badge read as a
    # different CASE and not just a different colour.
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], ls="none", marker="*", ms=14, mfc=C_BASE,
                             mec="white", mew=1.0, label="Base station"),
               mlines.Line2D([], [], ls="none", marker="o", ms=9, mfc=C_POINT,
                             mec="white", mew=1.2, label="Above water"),
               mlines.Line2D([], [], ls="none", marker="o", ms=9, mfc=C_WATER,
                             mec="white", mew=1.2, label="In water")]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8.5,
                    framealpha=0.9, borderpad=0.6, labelspacing=0.5)
    leg.set_zorder(9)

    # --- Nothing on the map may be written over -------------------------------
    # Everything already drawn becomes an obstacle for the labels: the badges, the
    # base station and its name, the legend, and the corners the scale bar and the
    # north arrow sit in (those are drawn as plain lines, so their extents are
    # taken from the axes fractions draw_scalebar/draw_north use).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    s = 72.0 / fig.dpi
    for px, py in np.vstack([anchors, ax.transData.transform((xb, yb))]):
        obstacles.append((px * s - MARKER_R_PT, py * s - MARKER_R_PT,
                          px * s + MARKER_R_PT, py * s + MARKER_R_PT))
    obstacles.append(_box_pt(base_ann.get_window_extent(rend), s, LABEL_PAD_PT))
    obstacles.append(_box_pt(leg.get_window_extent(rend), s, LABEL_PAD_PT))
    for frac in ((0.015, 0.005, 0.52, 0.10),      # scale bar + its label
                 (0.015, 0.74, 0.11, 0.95)):      # north arrow
        bb = ax.transAxes.transform([(frac[0], frac[1]), (frac[2], frac[3])])
        obstacles.append((bb[0][0] * s, bb[0][1] * s, bb[1][0] * s, bb[1][1] * s))

    offsets = place_labels(fig, ax, anns, anchors, obstacles, seg_xy, seg_id)

    # A label that had to move far gets a leader line back to its badge - without
    # one, #4 and #6 are 47 m apart on the ground and nothing would say which label
    # belongs to which.
    for (nr, lat, lon), (dx, dy, r), col in zip(points, offsets, cols):
        if r < LEADER_MIN_PT:
            continue
        x, y = lonlat_to_merc(lon, lat)
        ax.annotate("", xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.85,
                                    shrinkA=1.0, shrinkB=6.5))

    n_water = sum(1 for p in points if p[0] in water)
    d_water = [d for p, d in zip(points, dists) if p[0] in water]
    d_air = [d for p, d in zip(points, dists) if p[0] not in water]
    # Two ranges, not one: the longest message from an antenna held clear of the
    # surface says nothing about the longest one from a floating buoy.
    span = f"max range {max(d_air):.0f} m above water" if d_air else ""
    if d_water:
        span += f"   ·   {max(d_water):.0f} m in water"
    fig.text(0.5, 1 - 0.30 / fh, args.title, ha="center", va="top", fontsize=13,
             weight="bold")
    fig.text(0.5, 1 - 0.60 / fh,
             f"base station {lat_b:.6f}°N {lon_b:.6f}°E   ·   "
             f"{len(points)} successful messages "
             f"({len(points) - n_water} above water, {n_water} in water)   ·   "
             f"{span}",
             ha="center", va="top", fontsize=9, color="#444444")
    fig.text(0.3 / fw, 0.16 / fh, "Map data © OpenStreetMap contributors",
             ha="left", va="bottom", fontsize=7, color="#555555")
    return fig, dists, brgs


def read_csv(path):
    """nr,lat,lon[,in_water] (or lat,lon) with an optional header.

    Returns (points, in_water). A fourth column of 1 marks a point logged afloat;
    without it no point is, and --in-water sets them on the command line."""
    out, water = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            if not parts or parts[0].startswith("#"):
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:                 # header row
                continue
            if len(vals) >= 3:
                out.append((int(vals[0]), vals[1], vals[2]))
                if len(vals) >= 4 and vals[3]:
                    water.add(int(vals[0]))
            elif len(vals) == 2:
                out.append((len(out) + 1, vals[0], vals[1]))
    if not out:
        sys.exit(f"{path}: no coordinates found")
    return out, water


def main():
    here = os.path.join(os.path.dirname(_paths.TOOLS), "lora_range")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="read the drifter points from this file instead "
                                  "of the POINTS constant")
    ap.add_argument("-o", "--out", default=os.path.join(here, "lora_range.png"),
                    help="output file (default: lora_range/lora_range.png in the repo root)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="LoRa range test - base station and drifter")
    ap.add_argument("--no-map", action="store_true",
                    help="draw without OSM tiles (offline)")
    ap.add_argument("--show", action="store_true", help="open a window as well")
    ap.add_argument("--in-water", help="measurement numbers logged with the buoy "
                                       "afloat, e.g. 6 or 3,6 (default: the "
                                       "IN_WATER constant)")
    ap.add_argument("--tile-url", default=TILE_URL)
    ap.add_argument("--tile-cache", default=TILE_CACHE)
    args = ap.parse_args()

    if args.csv:
        points, water = read_csv(args.csv)
    else:
        points, water = POINTS, set(IN_WATER)
    if args.in_water is not None:
        water = {int(s) for s in args.in_water.replace(";", ",").split(",") if s.strip()}
    unknown = water - {p[0] for p in points}
    if unknown:
        sys.exit(f"--in-water: no such measurement: {sorted(unknown)}")

    fig, dists, brgs = make_figure(BASE, points, water, args)

    print(f"  base station: {BASE[0]:.6f}, {BASE[1]:.6f}")
    for (nr, lat, lon), d, b in zip(points, dists, brgs):
        where = "in water   " if nr in water else "above water"
        print(f"  #{nr}: {lat:.6f}, {lon:.6f}   {d:8.0f} m   at {b:5.1f}°   {where}")
    d_air = [d for p, d in zip(points, dists) if p[0] not in water]
    d_water = [d for p, d in zip(points, dists) if p[0] in water]
    if d_air:
        print(f"  max range above water: {max(d_air):.0f} m")
    if d_water:
        print(f"  max range in water:    {max(d_water):.0f} m")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"  wrote {args.out}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
