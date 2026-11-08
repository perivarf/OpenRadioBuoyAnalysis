#!/usr/bin/env python3
"""Map figures for ORB logging sessions: the drift track on OpenStreetMap with a
fact table, and an overview map of all the sessions (also of one).

Per session <stamp>_map.png:

    heading + description     (from <stamp>_notat.txt; else directory name + date)
    map                       (OSM tiles, track, time ticks, scale bar, insets)
    table                     (ses.csv, cfg.csv, Hs/Tp from _ana_python.csv, KVS)

The position comes from <stamp>_gps.csv. postprocess runs first on each session
(unless skip_postprocess), and the wave parameters are read from
<stamp>_ana_python.csv.

The sessions, numbered in the given order (or by `nr`), get an overview
written first, plus two crops of it:

    [<prefix>_]overview.png   map and table
    [<prefix>_]map.png        the maps
    [<prefix>_]table.png      the summary table

The overview has no heading (the caption goes in LaTeX).
Every figure gets _<config> in its name with a config. Tiles: maps/osm.py.

    mapplot.make_maps([session dirs...], MapOptions(taper_f1=0.2, taper_f2=0.3))

Run with script_mapplot.py or script_plot.py.
"""

import contextlib
import io
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob

import numpy as np

import _paths                                     # noqa: E402,F401
import postprocess                                # noqa: E402
from configs import check_config, config_suffix   # noqa: E402
from freq_axis import log_hz_axis                 # noqa: E402
from pipeline.output import plot_spectrum_from_data, write_outputs  # noqa: E402
from pipeline.session_input import read_kv, read_csv_stream  # noqa: E402
from pipeline.settings import Settings, normalise  # noqa: E402
from trim import trim_png                          # noqa: E402
from osm import (TILE_CACHE, TILE_URL, build_basemap,      # noqa: E402
                 draw_north, draw_scalebar, haversine, lonlat_to_merc,
                 merc_per_metre, pick_zoom, prefer_ipv4)

# --- Map extent --------------------------------------------------------------
PAD_FRAC = 0.12              # margin around the track bbox, fraction of the longest side
MIN_SPAN_M = 200.0           # smallest map extent [m]
TRACK_DECIM_HZ = 1.0         # the track is drawn decimated to this rate
PATH_DECIM_S = 10.0          # path length on 10 s steps (position noise ~0.35 m)

# --- Colours -----------------------------------------------------------------
C_TRACK = "#1f4e9c"
C_TICK = "#1f4e9c"
C_START = "#1a9850"
C_STOP = "#d73027"
C_BADGE = "#b03000"          # number badge
C_LABEL = "#12305e"
C_MUTED = "#555555"

METHOD_ROWS = [              # (key suffix in the ana file, display name)
    ("madgwick", "Madgwick"),
    ("sflp", "SFLP"),
    ("kalman", "Kalman"),
    ("nxp", "Kalman NXP"),
    ("gps", "GPS"),
]

# The curves in the spectrum plot: (column suffix in _spec_python.csv, name, style, colour)
PSD_CURVES = [
    ("madgwick", "Madgwick", "-", None),
    ("sflp", "SFLP", "-", None),
    ("kalman", "Kalman", "-", None),
    ("nxp", "Kalman NXP", "-", "#9467bd"),
    ("gps", "GPS", ":", "#d73027"),
]

DASH = "–"              # en dash for a missing value


@dataclass
class MapOptions:
    """Every choice for make_maps(). script_mapplot.py fills one from the command line."""
    nr: str = None                  # measurement numbers, comma separated
    badge_colors: str = None        # badge colour per session, comma separated;
                                    # an empty entry -> C_BADGE
    badge_legend: str = None        # what the badge colours mean, after
                                    # "number = measurement (see table): "
    out: str = None                 # output file (one session only)
    format: str = None              # "png" | "pdf"; None -> from out, else png
    dpi: int = 200
    width: float = 7.5              # figure width [in]
    taper_f1: float = 0.1           # passed to postprocess [Hz]
    taper_f2: float = 0.2
    seglen: int = None              # passed to postprocess; None -> its default
    fir_ntap: int = None            # passed to postprocess; None -> its default
    raw: str = None                 # passed to postprocess
    vacc_source: str = "auto"       # passed to postprocess
    psd_fmin: float = None          # spectrum plot range; None -> taper f1
    psd_fmax: float = 2.0
    skip_postprocess: bool = False  # use _ana_python.csv as it is
    plot_only: bool = False         # skip_postprocess + _spec_acc/_spec_eta.png from data/
    tick_sec: float = 30.0          # track points every N s
    label_sec: float = 300.0        # clock labels every N s
    zoom: int = None                # tile zoom; None -> automatic
    no_map: bool = False            # no tiles, grid only
    no_inset: bool = False
    inset_km: float = 1.5           # near inset width [km]
    inset_far_km: float = 50.0      # far inset width [km]
    overview_inset_km: str = "10,100"
    prefix: str = ""                # prefix on the overview filenames
    config: str = ""                # suffix _<config> on every figure
    overview_out: str = None
    overview_method: str = "madgwick"
    no_overview: bool = False
    only_overview: bool = False
    tile_url: str = TILE_URL
    tile_cache: str = TILE_CACHE
    init_note: bool = False         # write <stamp>_notat.txt stubs and stop
    no_kvs: bool = False            # no Kystvarslingssenteret row in the wave table
    psd_table: str = "all"          # stamps (comma separated, "all" or "") that also
                                    # get <stamp>_psd_table.png: the spectrum on its own


def resolve_session(path):
    """Directory, *_gps.csv or *_imu.csv -> (gps path, stamp, session directory)."""
    if os.path.isdir(path):
        cand = sorted(glob(os.path.join(path, "*_gps.csv")))
        if not cand:
            sys.exit(f"Found no *_gps.csv in {path}")
        gps = cand[0]
    elif os.path.isfile(path):
        base = os.path.basename(path)
        if base.endswith("_imu.csv"):          # convenience: accept the imu path too
            gps = path[: -len("_imu.csv")] + "_gps.csv"
            if not os.path.isfile(gps):
                sys.exit(f"Did not find {gps}")
        else:
            gps = path
    else:
        sys.exit(f"Did not find {path}")
    directory = os.path.dirname(os.path.abspath(gps))
    stamp = os.path.basename(gps)[: -len("_gps.csv")]
    return gps, stamp, directory


def read_gps_track(gps_path):
    """gps.csv -> dict of numpy arrays (3D fixes only), with t_s [s], n_raw,
    n_dropped and trunc (what read_csv_stream cut off the tail)."""
    cols = {}
    idx, rows, trunc = read_csv_stream(gps_path, "rel_ms")
    for nm in ("rel_ms", "lat", "lon"):
        if nm not in idx:
            sys.exit(f"Missing column '{nm}' in {gps_path}")
    want = [c for c in ("rel_ms", "lat", "lon", "gspeed", "hAccuracy",
                        "fix", "sats", "alt_msl") if c in idx]
    acc = {c: [float(r[idx[c]]) for r in rows] for c in want}
    n_raw = len(acc["rel_ms"])
    if n_raw < 2:
        sys.exit(f"Too few GPS rows in {gps_path}")
    for c in want:
        cols[c] = np.asarray(acc[c], dtype=float)
    cols["t_s"] = cols.pop("rel_ms") / 1000.0

    # Only 3D fixes.
    if "fix" in cols:
        ok = cols["fix"] >= 3
    else:
        ok = np.ones(n_raw, dtype=bool)
    n_drop = int(n_raw - ok.sum())
    if ok.sum() < 2:
        sys.exit(f"Too few rows with a 3D fix in {gps_path}")
    for c in list(cols):
        cols[c] = cols[c][ok]
    cols["n_raw"] = n_raw
    cols["n_dropped"] = n_drop
    cols["trunc"] = trunc
    return cols


def method_rate(ana, key):
    """" (480 Hz)" for the method's rate from the ana file, or "" if unknown."""
    try:
        hz = float(ana.get(f"rate_{key}_hz", ""))
    except (TypeError, ValueError):
        return ""
    return f" ({hz:g} Hz)" if hz > 0 else ""


def read_note(directory, stamp, ses, config=""):
    """<stamp>_notat.txt -> (heading, description): line 1 and the rest. Without the
    file: "<measurement directory> – <date>"."""
    path = note_path(directory, stamp)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # Skip leading blank lines and # comments from the stub.
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
            lines.pop(0)
        if lines:
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            return title, body
    up = os.path.dirname(os.path.abspath(directory))
    if config and os.path.basename(up) == config:
        up = os.path.dirname(up)
    parent = os.path.basename(up)
    iso = ses.get("start_utc_iso", "")
    date = iso[:10] if len(iso) >= 10 else stamp[:8]
    return f"{parent} {DASH} {date}", ""


def note_path(directory, stamp):
    return os.path.join(directory, f"{stamp}_notat.txt")


def write_note_stub(directory, stamp):
    path = note_path(directory, stamp)
    if os.path.isfile(path):
        print(f"  {path} already exists - leaving it alone.")
        return path
    parent = os.path.basename(os.path.dirname(os.path.abspath(directory)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{parent} {DASH} Measurement 1 {DASH} {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}\n")
        f.write("Short description of the conditions, e.g. calm sea in bay, light SW breeze.\n")
    print(f"  wrote {path} - edit line 1 (heading) and the rest (description).")
    return path


def run_postprocess(directory, stamp, opts, quiet=True):
    """Run postprocess on the session with the taper/seglen/vacc source/raw from
    opts (everything else at postprocess's defaults), writing _ana_python.csv and
    _spec_python.csv. Its output is printed only on failure (or quiet=False)."""
    print(f"  running postprocess {stamp} "
          f"(taper {opts.taper_f1:g}-{opts.taper_f2:g} Hz) ...", flush=True)
    st = Settings.from_flat(
        taper_f1=opts.taper_f1, taper_f2=opts.taper_f2,
        vacc_source=opts.vacc_source, config=opts.config,
        seglen=opts.seglen or None, raw=opts.raw or None,
        **({"fir_ntap": opts.fir_ntap} if opts.fir_ntap else {}))
    normalise(st)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            write_outputs(postprocess.analyse_session(directory, st), st)
    except BaseException:
        # BaseException: postprocess stops with sys.exit().
        tail = "\n".join(buf.getvalue().strip().splitlines()[-15:])
        print(f"postprocess failed for {stamp}:\n{tail}", file=sys.stderr)
        raise
    if not quiet:
        print(buf.getvalue())


def read_spec(directory, stamp):
    """<stamp>_spec_python.csv -> {column name: numpy array}, or None."""
    path = os.path.join(directory, f"{stamp}_spec_python.csv")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        acc = {nm: [] for nm in header}
        for line in f:
            fld = line.rstrip("\n").split(",")
            if len(fld) < len(header):
                break
            try:
                vals = [float(v) for v in fld[:len(header)]]
            except ValueError:
                break
            for nm, v in zip(header, vals):
                acc[nm].append(v)
    if len(acc.get("f_hz", [])) < 2:
        return None
    return {nm: np.asarray(v, dtype=float) for nm, v in acc.items()}


def pick_ana(directory, stamp):
    """<stamp>_ana_python.csv -> (dict, filename), or ({}, None)."""
    py = os.path.join(directory, f"{stamp}_ana_python.csv")
    if os.path.isfile(py):
        return read_kv(py), os.path.basename(py)
    return {}, None


# --- Statistics --------------------------------------------------------------
def track_stats(gps):
    """Path length, net drift, mean position and speed."""
    t, lat, lon = gps["t_s"], gps["lat"], gps["lon"]
    # Path length on PATH_DECIM_S steps.
    grid = np.arange(t[0], t[-1] + PATH_DECIM_S, PATH_DECIM_S)
    k = np.searchsorted(t, grid)
    k = np.unique(np.clip(k, 0, len(t) - 1))
    dist = float(np.sum(haversine(lat[k[:-1]], lon[k[:-1]], lat[k[1:]], lon[k[1:]])))
    net = float(haversine(lat[0], lon[0], lat[-1], lon[-1]))
    brg = math.degrees(math.atan2(
        math.sin(math.radians(lon[-1] - lon[0])) * math.cos(math.radians(lat[-1])),
        math.cos(math.radians(lat[0])) * math.sin(math.radians(lat[-1]))
        - math.sin(math.radians(lat[0])) * math.cos(math.radians(lat[-1]))
        * math.cos(math.radians(lon[-1] - lon[0])))) % 360.0
    out = dict(dist=dist, net=net, bearing=brg,
               lat_mid=float(np.mean(lat)), lon_mid=float(np.mean(lon)))
    # The actual fix rate (the set one is in cfg.csv).
    if len(t) > 1:
        out["gps_rate"] = len(t) / max(float(t[-1] - t[0]), 1e-9)
    if "gspeed" in gps:
        out["v_mean"] = float(np.mean(gps["gspeed"]))
        out["v_max"] = float(np.max(gps["gspeed"]))
    if "hAccuracy" in gps:
        out["hacc"] = float(np.median(gps["hAccuracy"]))
    if "sats" in gps:
        out["sats"] = float(np.median(gps["sats"]))
    return out


def fit_bbox(bbox, ratio):
    """Inflate the bbox about its centre to height/width = ratio (the map axis
    shape), so aspect="equal" leaves no white space."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if h / w > ratio:
        w = h / ratio
    else:
        h = w * ratio
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def map_bbox(lat, lon, pad_frac=PAD_FRAC, min_span_m=MIN_SPAN_M):
    """A roughly square Mercator bbox around the track, with margin and minimum size."""
    x, y = lonlat_to_merc(lon, lat)
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    lat_mid = float(np.mean(lat))
    min_span = min_span_m * merc_per_metre(lat_mid)
    dx, dy = max(x1 - x0, min_span), max(y1 - y0, min_span)
    pad = pad_frac * max(dx, dy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return (cx - dx / 2 - pad, cy - dy / 2 - pad,
            cx + dx / 2 + pad, cy + dy / 2 + pad)


# --- Formatting --------------------------------------------------------------
def fmt_val(d, key, unit="", scale=None, dec=None):
    """Take key from a cfg/ses/ana dict and format it, or return an en dash."""
    raw = d.get(key)
    if raw is None or raw == "":
        return DASH
    try:
        v = float(raw)
    except ValueError:
        return str(raw)
    if scale:
        v *= scale
    if dec is None:
        txt = f"{v:g}"
    else:
        txt = f"{v:.{dec}f}"
    return f"{txt}{unit}"


def joined(parts, template):
    """template.format(*parts), or one en dash when every part is missing."""
    if all(p == DASH for p in parts):
        return DASH
    return template.format(*parts)


def fmt_segment(ana, cfg):
    """The Welch segment as "Segment 1024 (102.4 s), 75% overlap". The ana file
    wins over cfg.csv (cfg is the fallback for sessions before build 66)."""
    def lookup(key, conv=float):
        for src in (ana, cfg):
            if src.get(key):
                try:
                    v = conv(float(src[key]))
                except (ValueError, ZeroDivisionError):
                    continue
                if v:
                    return v
        return None

    n = lookup("welch_seglen", int)
    if not n:
        return None
    fs = lookup("vacc_fs_hz") or lookup("vacc_bucket_ms", lambda v: 1000.0 / v)
    txt = f"Segment {n}"
    if fs:
        secs = f"{n / fs:.1f}".removesuffix(".0")
        txt += f" ({secs} s)"
    div = lookup("welch_overlap_div")
    if div and div > 0:
        txt += f", {(1.0 - 1.0 / div) * 100:.0f}% overlap"
    return txt


def fmt_dur(ms):
    try:
        s = int(round(float(ms) / 1000.0))
    except (TypeError, ValueError):
        return DASH
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_time(iso):
    """'2026-07-31T13:15:28Z' -> ('2026-07-31 13:15:28 UTC', '15:15:28 local')."""
    if not iso:
        return DASH, None
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return iso, None
    utc = dt.strftime("%Y-%m-%d %H:%M:%S")
    try:
        from zoneinfo import ZoneInfo
        loc = dt.astimezone(ZoneInfo("Europe/Oslo")).strftime("%H:%M:%S")
    except Exception:
        loc = None
    return utc, loc


def fmt_latlon(lat, lon):
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"{abs(lat):.5f}°{ns}  {abs(lon):.5f}°{ew}"


# --- Table content -----------------------------------------------------------
def build_blocks(cfg, ses, ana, stats, kvs=None):
    """The rows in the "Session" and "Logging" blocks + the wave table (with a
    Kystvarslingssenteret row last when kvs is given)."""
    t_start, l_start = fmt_time(ses.get("start_utc_iso", ""))
    session_rows = [
        ("Date", t_start.split()[0] if t_start != DASH else DASH),
        ("Start (UTC)", t_start.split()[-1] if t_start != DASH else DASH),
    ]
    if l_start:
        session_rows.append(("Start (local)", l_start))
    session_rows += [
        ("Duration", fmt_dur(ses.get("duration_ms"))),
        ("Position (mean)", fmt_latlon(stats["lat_mid"], stats["lon_mid"])),
        ("Path length (10 s)", f"{stats['dist']:.0f} m"),
        ("Net drift", f"{stats['net']:.0f} m at {stats['bearing']:.0f}°"),
        ("Speed mean/max",
         f"{stats['v_mean']:.2f} / {stats['v_max']:.2f} m/s"
         if "v_mean" in stats else DASH),
        ("GPS quality",
         f"hAcc {stats['hacc']:.2f} m · {stats.get('sats', 0):.0f} sats"
         if "hacc" in stats else DASH),
        # imu_rows is in ses.csv on older sessions, else in the ana file.
        ("Rows IMU / GPS",
         joined([fmt_val(ses if "imu_rows" in ses else ana, "imu_rows"),
                 fmt_val(ses, "gps_rows")], "{} / {}")),
    ]

    # Build: seq (cfg.csv, or ses.csv on older sessions) / date (+ short hash).
    date = fmt_val(cfg, "build_date")
    sha = (cfg.get("build_commit") or "")[:7]
    # With a hash, drop the clock time from "Aug 22 2026 15:31:22".
    if sha:
        head, _, last = date.rpartition(" ")
        if head and ":" in last:
            date = head
    build = joined([fmt_val(cfg if "build_seq" in cfg else ses, "build_seq"), date],
                   "{} / {}")
    if sha:
        build = f"{build}  ({sha})" if build != DASH else sha
    logging_rows = [
        # Actual ODR from the samples per row (postprocess).
        ("IMU ODR set / actual",
         joined([fmt_val(cfg, "imu_odr_hz"),
                 fmt_val(ana, "imu_odr_actual_hz", dec=1)], "{} / {} Hz")),
        ("Output rate / window",
         joined([fmt_val(cfg, "output_rate_hz"), fmt_val(cfg, "window_ms")],
                "{} Hz / {} ms")),
        ("Full-scale",
         joined([fmt_val(cfg, "accel_fs_g"), fmt_val(cfg, "gyro_fs_dps")],
                "±{} g / ±{} °/s")),
        ("SFLP ODR", fmt_val(cfg, "sflp_odr_hz", " Hz", dec=0)),
        # Actual rate from the track (track_stats).
        ("GPS rate set / actual",
         joined([fmt_val(cfg, "gps_rate_hz"),
                 f"{stats['gps_rate']:.2f}" if "gps_rate" in stats else DASH],
                "{} / {} Hz")),
        ("Build seq / date", build),
    ]

    wave_rows = []
    for key, name in METHOD_ROWS:
        row = [name + method_rate(ana, key)]
        for p in ("Hs", "Tp", "Tz"):
            row.append(fmt_val(ana, f"{p}_{key}", dec=(3 if p == "Hs" else 2)))
        wave_rows.append(row)
    if kvs:
        wave_rows.append(["KVS", f"{kvs['hs_mean']:.3f}",
                          f"{kvs['tp_mean']:.2f}", DASH])
    return session_rows, logging_rows, wave_rows


# --- Drawing -----------------------------------------------------------------
def corner_load(x, y, bbox):
    """Number of track points in each of the map's four corners."""
    fx = (x - bbox[0]) / (bbox[2] - bbox[0])
    fy = (y - bbox[1]) / (bbox[3] - bbox[1])
    rects = {"tl": (0.00, 0.54, 0.34, 1.00), "tr": (0.66, 0.54, 1.00, 1.00),
             "bl": (0.00, 0.00, 0.34, 0.46), "br": (0.66, 0.00, 1.00, 0.46)}
    return {k: int(np.sum((fx >= r[0]) & (fx <= r[2])
                          & (fy >= r[1]) & (fy <= r[3])))
            for k, r in rects.items()}


def place_decorations(load):
    """Distribute inset / north arrow / scale bar over the least busy corners."""
    free = sorted(load, key=lambda k: (load[k], k))
    inset = free[0]
    top = [k for k in free if k.startswith("t") and k != inset] or ["tl"]
    bot = [k for k in free if k.startswith("b") and k != inset] or ["bl"]
    return inset, top[0], bot[0]


def draw_map(ax, gps, stats, opts, bbox, quiet=False):
    import matplotlib.patheffects as pe

    lat, lon, t = gps["lat"], gps["lon"], gps["t_s"]
    zoom = opts.zoom if opts.zoom else pick_zoom(bbox)

    if not opts.no_map:
        img, extent = build_basemap(bbox, zoom, opts.tile_url, opts.tile_cache, quiet)
        if img is not None:
            ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear",
                      zorder=0)
        else:
            opts.no_map = True
    if opts.no_map:
        ax.set_facecolor("#eef2f5")   # neutral background; the scale bar gives the scale

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#8a8a8a")
        s.set_linewidth(0.8)

    halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]
    x, y = lonlat_to_merc(lon, lat)

    # Track decimated to TRACK_DECIM_HZ.
    dt_med = float(np.median(np.diff(t))) if len(t) > 2 else 0.1
    step = max(1, int(round(1.0 / (TRACK_DECIM_HZ * max(dt_med, 1e-3)))))
    ax.plot(x[::step], y[::step], "-", color=C_TRACK, lw=1.6, zorder=3,
            solid_capstyle="round", path_effects=halo, label="Drift track")

    # A point every tick_sec, a clock label every label_sec.
    def at_times(period):
        if period <= 0:
            return np.array([], dtype=int)
        grid = np.arange(t[0], t[-1] + 1e-9, period)
        k = np.searchsorted(t, grid)
        return np.unique(np.clip(k, 0, len(t) - 1))

    kt = at_times(opts.tick_sec)
    ax.plot(x[kt], y[kt], "o", ms=3.4, mfc=C_TICK, mec="white", mew=0.7,
            ls="none", zorder=4, label=f"every {opts.tick_sec:g} s")

    kl = at_times(opts.label_sec)
    if len(kl):
        ax.plot(x[kl], y[kl], "o", ms=5.6, mfc="white", mec=C_TICK, mew=1.4,
                ls="none", zorder=5)
        span = bbox[2] - bbox[0]
        for k in kl:
            el = t[k] - t[0]
            if el < 1.0 or (t[-1] - t[k]) < opts.label_sec * 0.4:
                continue                        # do not clutter the start/end markers
            ax.text(x[k] + 0.012 * span, y[k] + 0.008 * span,
                    f"{int(el) // 60:02d}:{int(el) % 60:02d}",
                    fontsize=6.5, color=C_LABEL, ha="left", va="bottom", zorder=6,
                    path_effects=[pe.Stroke(linewidth=2.0, foreground="white"),
                                  pe.Normal()])

    ax.plot(x[0], y[0], "o", ms=9, mfc=C_START, mec="white", mew=1.4,
            ls="none", zorder=7, label="Start")
    ax.plot(x[-1], y[-1], "s", ms=9, mfc=C_STOP, mec="white", mew=1.4,
            ls="none", zorder=7, label="End")

    load = corner_load(x[::step], y[::step], bbox)
    inset_c, north_c, scale_c = place_decorations(load)
    draw_scalebar(ax, bbox, stats["lat_mid"], scale_c)
    draw_north(ax, bbox, north_c)
    # The legend is drawn below the map by make_figure.
    return inset_c, ax.get_legend_handles_labels()


def badge_scale(nr):
    """Badge diameter relative to a one-character one."""
    return 1.0 + 0.3 * (len(str(nr)) - 1)


def draw_badge(ax, x, y, nr, size=9.0, fs=7.5, zorder=10, color=C_BADGE):
    """A numbered marker, the same on the overview and the single maps."""
    size *= badge_scale(nr)                     # room for e.g. "1A"
    ax.plot([x], [y], marker="o", ms=size, mfc=color, mec="white", mew=1.3,
            ls="none", zorder=zorder)
    ax.text(x, y, str(nr), fontsize=fs, weight="bold", color="white",
            ha="center", va="center", zorder=zorder + 1)


def draw_on_inset(ax, cx, cy, lat_mid, span_km, opts,
                  frame_bbox=None, badge_nr=None, badge_color=C_BADGE):
    """An inset map of span_km × span_km around (cx, cy) in ax, with frame_bbox as a
    red frame. Returns the inset's bbox (for cascading), or None without tiles."""
    half = span_km * 1000.0 * merc_per_metre(lat_mid) / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)
    zoom = pick_zoom(bbox, max_tiles=16)
    img, extent = build_basemap(bbox, zoom, opts.tile_url, opts.tile_cache, quiet=True)
    if img is None:
        ax.remove()
        return None
    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if frame_bbox is not None:
        import matplotlib.patches as mpatches
        w = frame_bbox[2] - frame_bbox[0]
        h = frame_bbox[3] - frame_bbox[1]
        # A dot when the frame would be only a few pixels.
        if w < 0.045 * (bbox[2] - bbox[0]):
            ax.plot([(frame_bbox[0] + frame_bbox[2]) / 2],
                    [(frame_bbox[1] + frame_bbox[3]) / 2], marker="o", ms=5,
                    mfc=C_STOP, mec="white", mew=1.0, zorder=5)
        else:
            ax.add_patch(mpatches.Rectangle(
                (frame_bbox[0], frame_bbox[1]), w, h, fill=False, ec=C_STOP,
                lw=1.2, zorder=5))
    if badge_nr:
        draw_badge(ax, cx, cy, badge_nr, size=11, fs=7.5, color=badge_color)
    ax.text(0.5, 0.015, f"{span_km:g} km", transform=ax.transAxes,
            fontsize=5.8, color="#222222", ha="center", va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0), zorder=8)
    for s in ax.spines.values():
        s.set_edgecolor("#444444")
        s.set_linewidth(1.0)
    return bbox


def draw_kv_blocks(ax, blocks, w_key=0.42, rows_total=None):
    """One column of key/value blocks stacked: [(title, rows)].
    rows_total gives a shared row height across several axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = rows_total or sum(len(r) + 1.5 for _, r in blocks)
    dy = 1.0 / n
    y = 1.0
    for title, rows in blocks:
        ax.text(0, y, title, fontsize=8.0, weight="bold", color="#222222",
                va="top")
        for k, v in rows:
            y -= dy
            ax.text(0, y, k, fontsize=7.0, color=C_MUTED, va="top")
            ax.text(w_key, y, v, fontsize=7.0, color="#111111", va="top",
                    family="DejaVu Sans Mono")
        y -= 1.5 * dy


def draw_wave_table(ax, wave_rows, ana_src, ana, cfg, opts, kvs=None):
    """Hs/Tp/Tz per method (+ KVS), with taper, segment and sources as footnotes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    dy = 1.0 / (len(wave_rows) + 6.4 + (0.8 if kvs else 0.0))
    heads = ["Method", "Hs [m]", "Tp [s]", "Tz [s]"]
    ax.text(0, 1.0, "Wave parameters", fontsize=8.0, weight="bold",
            color="#222222", va="top")
    cols = [0.0, 0.40, 0.56, 0.72]
    yy = 1.0 - 1.25 * dy
    for cx_, h in zip(cols, heads):
        ax.text(cx_, yy, h, fontsize=7.0, color="#222222", weight="bold", va="top")
    for i, row in enumerate(wave_rows):
        yy = 1.0 - (i + 2.35) * dy
        ax.text(cols[0], yy, row[0], fontsize=7.0, color=C_MUTED, va="top")
        for cx_, v in zip(cols[1:], row[1:]):
            ax.text(cx_, yy, v, fontsize=7.0, color="#111111", va="top",
                    family="DejaVu Sans Mono")
    foot = [f"Taper {opts.taper_f1:g}–{opts.taper_f2:g} Hz"]
    seg = fmt_segment(ana, cfg)
    if seg:
        foot.append(seg)
    # The source per filter, from the ana file.
    def ran(prefix):
        return sorted(name for key, name in METHOD_ROWS
                      if str(ana.get(f"src_{key}", "")).startswith(prefix))
    parts = []
    for names, how in ((ran("raw"), "raw stream, ODR"),
                       (ran("onboard"), "on board, row rate"),
                       (ran("csv"), "replayed, row rate")):
        if names:
            parts.append(f"{', '.join(names)}: {how}")
    if parts:
        foot.append("; ".join(parts) + ".")
        foot.append("SFLP is the on-chip fusion. Rates in parentheses.")
    foot.append(f"Source: {ana_src}" if ana_src else "No _ana_python.csv found")
    if kvs:
        foot.append("KVS: kystvarslingssenteret.no, 100 m grid, "
                    f"downloaded {kvs['downloaded']}")
    for i, line in enumerate(foot):
        ax.text(0, 1.0 - (len(wave_rows) + 2.6 + 0.8 * i) * dy, line, fontsize=6.2,
                color=C_MUTED, va="top", style="italic")


def draw_psd(ax, spec, opts, ana=None, title="Elevation spectrum", skip=()):
    """The elevation spectrum S_eta(f) in log-log, one curve per method (except the
    keys in skip), with the taper ramp shaded."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    if spec is None:
        ax.text(0.5, 0.5, "No _spec_python.csv", fontsize=8, color=C_MUTED,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    f = spec["f_hz"]
    # From the taper's f1 (zero below) unless psd_fmin is set.
    fmin = opts.psd_fmin if opts.psd_fmin else opts.taper_f1
    band = (f >= fmin) & (f <= opts.psd_fmax)
    any_line = False
    for key, name, style, color in PSD_CURVES:
        col = f"psd_eta_{key}"
        if col not in spec or key in skip:
            continue
        y = np.where(spec[col] > 0.0, spec[col], np.nan)   # 0 => outside the taper
        labelled = name + (method_rate(ana, key) if ana else "")
        # Hs from _ana_python.csv, as in _spec_eta.png ("<name> - <Hs> m").
        try:
            hs = float((ana or {}).get(f"Hs_{key}", ""))
        except ValueError:
            hs = -1.0
        if hs > 0.0:
            labelled += f" - {hs:.3f} m"
        ax.plot(f[band], y[band], style, lw=1.3, label=labelled,
                **({"color": color} if color else {}))
        any_line = True
    ax.axvspan(opts.taper_f1, opts.taper_f2, color="orange", alpha=0.16, lw=0)
    # Plain numbers on x: decades as major, 2 and 5 as minor (freq_axis).
    log_hz_axis(ax, (fmin, opts.psd_fmax))
    ax.set_xlabel("frequency [Hz]", fontsize=7.5)
    ax.set_ylabel("PSD elevation  [m²/Hz]", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.tick_params(axis="x", which="minor", labelsize=5.8)
    if any_line:
        ax.legend(fontsize=5.8, framealpha=0.85, borderpad=0.35,
                  handletextpad=0.5, labelspacing=0.28, loc="best")
    ax.set_title(title, fontsize=8.0, weight="bold", loc="left",
                 color="#222222", pad=3)


def check_ana_taper(ana, ana_src, opts):
    """With skip_postprocess the existing _ana_python.csv is used as it is: stop
    unless it was made with this taper."""
    try:
        same = (ana.get("cutoff_mode") == "taper"
                and abs(float(ana["taper_f1_hz"]) - opts.taper_f1) < 1e-6
                and abs(float(ana["taper_f2_hz"]) - opts.taper_f2) < 1e-6)
    except (KeyError, ValueError):
        same = False
    if not same:
        sys.exit(f"--skip-postprocess: {ana_src or 'no _ana_python.csv'} was not made "
                 f"with taper {opts.taper_f1:g}-{opts.taper_f2:g} Hz - run without "
                 "--skip-postprocess")


def load_kvs(directory, stamp, opts, quiet=False):
    """Kystvarslingssenteret's modelled Hs/Tp over the session
    (kystvarsling.session_waves), or None when off or unavailable."""
    if opts.no_kvs:
        return None
    import kystvarsling
    try:
        return kystvarsling.session_waves(os.path.join(directory, f"{stamp}_ses.csv"))
    except (OSError, ValueError) as e:
        if not quiet:
            print(f"  KVS: not available ({e})")
        return None


def load_session(path, opts, nr=None):
    """Everything about one session -> dict, for the single map and the overview.
    Runs postprocess first unless opts.skip_postprocess; with opts.plot_only the
    spectrum figures are redrawn from data/ instead."""
    gps_path, stamp, directory = resolve_session(path)
    skip = opts.skip_postprocess or opts.plot_only
    if opts.plot_only:
        plot_spectrum_from_data(directory, stamp, opts.config)
    elif not skip:
        run_postprocess(directory, stamp, opts)
    ses = read_kv(os.path.join(directory, f"{stamp}_ses.csv"))
    cfg = read_kv(os.path.join(directory, f"{stamp}_cfg.csv"))
    ana, ana_src = pick_ana(directory, stamp)
    if skip:
        check_ana_taper(ana, ana_src, opts)
    gps = read_gps_track(gps_path)
    if not ses.get("duration_ms") and len(gps["t_s"]):
        # A session stopped abnormally has no duration in ses.csv: the GPS span.
        ses["duration_ms"] = f"{gps['t_s'][-1] * 1000.0:.0f}"
    title, body = read_note(directory, stamp, ses, opts.config)
    return dict(nr=nr, stamp=stamp, directory=directory, gps=gps,
                stats=track_stats(gps), ses=ses, cfg=cfg, ana=ana,
                ana_src=ana_src, spec=read_spec(directory, stamp),
                title=title, body=body, kvs=load_kvs(directory, stamp, opts),
                has_note=os.path.isfile(note_path(directory, stamp)))


def draw_header(ax, sess, title_fs=13.5):
    """Heading with the number badge in front, and the description below. Everything
    in 0..1 data coordinates."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0 = 0.0
    if sess.get("nr"):
        extra = len(str(sess["nr"])) - 1          # wider badge for e.g. "2B"
        ax.plot([0.010 + 0.004 * extra], [0.86], marker="o", ms=15 + 5 * extra,
                mfc=sess.get("badge_color", C_BADGE), mec="none", ls="none",
                clip_on=False)
        ax.text(0.010 + 0.004 * extra, 0.86, str(sess["nr"]), fontsize=9,
                weight="bold", color="white", ha="center", va="center")
        x0 = 0.026 + 0.012 * extra
    ax.text(x0, 1.0, sess["title"], fontsize=title_fs, weight="bold", va="top",
            color="#111111")
    if sess["body"]:
        ax.text(x0, 0.52, sess["body"], fontsize=8.6, va="top", color=C_MUTED,
                wrap=True)


def make_psd_table_figure(sess, opts):
    """The spectrum of the single map on its own, titled with the measurement name
    (for a figure in the report)."""
    import matplotlib.pyplot as plt
    fw = 0.475 * (opts.width - 0.9) + 0.2
    psd_h = 1.95
    m_b, m_t = 0.42, 0.2
    fh = m_b + psd_h + m_t
    fig = plt.figure(figsize=(fw, fh))
    title = "Elevation spectrum" + (f" – {sess['nr']}" if sess.get("nr") else "")
    draw_psd(fig.add_axes([0.44 / fw, m_b / fh, (fw - 0.54) / fw, psd_h / fh]),
             sess["spec"], opts, sess["ana"], title=title)
    return fig


def make_figure(sess, opts):
    """One A4-friendly page per measurement:

        [nr] heading + description
        ┌──────────────────────┬───────────┐
        │  GPS track on OSM    │ near inset│
        │                      ├───────────┤
        │                      │ far inset │
        └──────────────────────┴───────────┘
        ┌───────────────────┬──────────────┐
        │ elevation spectrum│ Session      │
        ├───────────────────┤ Logging      │
        │ Hs/Tp/Tz + taper  │              │
        └───────────────────┴──────────────┘
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gps, stats = sess["gps"], sess["stats"]
    fw = opts.width
    m_l, m_r = 0.45, 0.45
    cw = fw - m_l - m_r                      # content width

    # Two square insets stacked set the top row height; the map bbox is stretched
    # to fit, so every session figure has the same size.
    gap_x, gap_y = 0.14, 0.10
    ins_w = 0.265 * cw
    map_w = cw - ins_w - gap_x
    top_h = 2 * ins_w + gap_y
    bbox = fit_bbox(map_bbox(gps["lat"], gps["lon"]), top_h / map_w)

    psd_w = 0.475 * cw
    tab_w = cw - psd_w - 0.22
    psd_h, wave_h = 1.95, 1.28      # wave_h holds 5 method rows + footnotes
    bot_h = psd_h + 0.42 + wave_h            # 0.42 = room for the axis labels

    body = sess["body"]
    head_h = 0.52 + (0.17 * (1 + len(body) // 120) if body else 0.0)
    attr_h = 0.24
    m_b, m_t = 0.26, 0.26
    fh = m_b + bot_h + 0.30 + attr_h + top_h + 0.16 + head_h + m_t

    fig = plt.figure(figsize=(fw, fh))

    def rect(x, y, w, h):
        return [x / fw, y / fh, w / fw, h / fh]

    y_top = m_b + bot_h + 0.30 + attr_h      # lower edge of the map row
    draw_header(fig.add_axes(rect(m_l, y_top + top_h + 0.16, cw, head_h)), sess)

    ax_map = fig.add_axes(rect(m_l, y_top, map_w, top_h))
    _, (handles, labels) = draw_map(ax_map, gps, stats, opts, bbox)

    # Inset column: near inset (frame = the map) above the far one (frame = near).
    x_ins = m_l + map_w + gap_x
    if not opts.no_inset and not opts.no_map:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        spans = [opts.inset_km] + [float(s) for s in
                                   str(opts.inset_far_km).split(",") if s.strip()]
        frame = bbox
        for i, span in enumerate(spans[:2]):
            y_i = y_top + (top_h - ins_w if i == 0 else 0.0)
            axi = fig.add_axes(rect(x_ins, y_i, ins_w, ins_w))
            frame = draw_on_inset(axi, cx, cy, stats["lat_mid"], span, opts,
                                  frame_bbox=frame,
                                  badge_nr=sess.get("nr") if i == 0 else None,
                                  badge_color=sess.get("badge_color", C_BADGE))
            if frame is None:
                break

    # Legend and attribution below the map row.
    y_attr = m_b + bot_h + 0.30
    fig.legend(handles, labels, loc="lower left", ncol=4, fontsize=6.6,
               frameon=False, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(m_l / fw, (y_attr - 0.02) / fh))
    attr = ("Basemap © OpenStreetMap contributors" if not opts.no_map
            else "No basemap (--no-map)")
    fig.text(1.0 - m_r / fw, y_attr / fh, attr,
             fontsize=6.0, color=C_MUTED, ha="right", va="bottom")

    draw_psd(fig.add_axes(rect(m_l + 0.34, m_b + wave_h + 0.42,
                               psd_w - 0.34, psd_h)), sess["spec"], opts,
             sess["ana"])

    session_rows, logging_rows, wave_rows = build_blocks(
        sess["cfg"], sess["ses"], sess["ana"], stats, sess.get("kvs"))
    draw_wave_table(fig.add_axes(rect(m_l, m_b, psd_w, wave_h)),
                    wave_rows, sess["ana_src"], sess["ana"], sess["cfg"], opts,
                    sess.get("kvs"))
    draw_kv_blocks(fig.add_axes(rect(fw - m_r - tab_w, m_b, tab_w, bot_h)),
                   [("Session", session_rows), ("Logging", logging_rows)], w_key=0.44)
    return fig


# --- Overview map ------------------------------------------------------------
def draw_caption(fig, x, y, badge_legend=None, fs=6.2):
    """The key under the overview map, left aligned at (x, y) in figure fractions:
    the start and end markers as symbols, then what the numbers (and their colours)
    mean."""
    from matplotlib.lines import Line2D
    renderer = fig.canvas.get_renderer()
    fw_px = fig.bbox.width
    y_mid = y + 0.45 * fs / 72.0 / fig.get_figheight()    # middle of the text line
    numbers = "number = measurement (see table)"
    if badge_legend:
        numbers += f": {badge_legend}"
    for part in [("o", C_START), "start  ·  ", ("s", C_STOP), "end  ·  ", numbers]:
        if isinstance(part, tuple):
            marker, color = part
            fig.add_artist(Line2D([x + 2.5 / fw_px], [y_mid], marker=marker, ms=4.0,
                                  mfc=color, mec="white", mew=0.6, ls="none",
                                  transform=fig.transFigure))
            x += (4.0 + 2.5) / 72.0 * fig.dpi / fw_px + 2.0 / fw_px
            continue
        t = fig.text(x, y, part, fontsize=fs, color=C_MUTED, ha="left", va="bottom")
        x += t.get_window_extent(renderer).width / fw_px


def spread_badges(pts, bbox, min_frac=0.055, iters=60):
    """Push the number badges at least min_frac of the map apart -> new positions."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    p = np.array(pts, dtype=float)
    if len(p) < 2:
        return p
    dmin = min_frac * max(w, h)
    # Coinciding badges (copies of one session): a row centred on the point, left to
    # right in number order.
    groups = {}
    for i, q in enumerate(p):
        groups.setdefault((round(q[0], 6), round(q[1], 6)), []).append(i)
    for idx in groups.values():
        for k, i in enumerate(idx):
            p[i, 0] += (k - (len(idx) - 1) / 2.0) * dmin
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = p[j] - p[i]
                r = math.hypot(d[0], d[1])
                if r >= dmin:
                    continue
                if r < 1e-9:            # exact coincidence: push in an arbitrary direction
                    r = 1e-9 * dmin     # (not dmin - that makes the push zero)
                    d = np.array([r, 0.0])
                push = (dmin - r) / 2.0 * d / r
                p[i] -= push
                p[j] += push
                moved = True
        if not moved:
            break
    return p


def make_overview(sessions, opts, out_dir):
    """One map with all the measurements, numbered, and a summary table.

    Returns (fig, bands): bands = {"map": Bbox, "table": Bbox} in inches, for
    writing the two halves as crops of the same figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    lat = np.concatenate([s["gps"]["lat"] for s in sessions])
    lon = np.concatenate([s["gps"]["lon"] for s in sessions])

    fw = opts.width
    m_l, m_r, m_b, m_t = 0.45, 0.45, 0.26, 0.26
    cw = fw - m_l - m_r
    # 2/3 map, 1/3 two stacked insets; the bbox is stretched to the map's shape.
    gap_x, gap_y = 0.14, 0.10
    ins_w = (cw - gap_x) / 3.0
    map_w = cw - ins_w - gap_x
    map_h = 2 * ins_w + gap_y
    bbox = fit_bbox(map_bbox(lat, lon, pad_frac=0.18, min_span_m=1000.0),
                    map_h / map_w)

    # No heading: the caption is added in LaTeX.
    tab_h = 0.56 + 0.175 * len(sessions) + (0.14 if any(s.get("kvs") for s in sessions) else 0.0)
    gap, attr_h = 0.30, 0.24
    fh = m_b + tab_h + gap + attr_h + map_h + m_t

    fig = plt.figure(figsize=(fw, fh))

    def rect(x, y, w, h):
        return [x / fw, y / fh, w / fw, h / fh]

    y_map = m_b + tab_h + gap + attr_h

    ax = fig.add_axes(rect(m_l, y_map, map_w, map_h))
    zoom = opts.zoom if opts.zoom else pick_zoom(bbox)
    if not opts.no_map:
        img, extent = build_basemap(bbox, zoom, opts.tile_url, opts.tile_cache)
        if img is not None:
            ax.imshow(img, extent=extent, origin="upper",
                      interpolation="bilinear", zorder=0)
        else:
            opts.no_map = True
    if opts.no_map:
        ax.set_facecolor("#eef2f5")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#8a8a8a")
        s.set_linewidth(0.8)

    import matplotlib.patheffects as pe
    halo = [pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()]
    anchors = []
    for s in sessions:
        x, y = lonlat_to_merc(s["gps"]["lon"], s["gps"]["lat"])
        ax.plot(x[::10], y[::10], "-", color=C_TRACK, lw=1.4, zorder=3,
                solid_capstyle="round", path_effects=halo)
        ax.plot([x[0]], [y[0]], "o", ms=4.5, mfc=C_START, mec="white", mew=0.9,
                ls="none", zorder=4)
        ax.plot([x[-1]], [y[-1]], "s", ms=4.5, mfc=C_STOP, mec="white", mew=0.9,
                ls="none", zorder=4)
        anchors.append((float(np.mean(x)), float(np.mean(y))))

    placed = spread_badges(anchors, bbox, min_frac=0.055 * max(
        badge_scale(s["nr"]) for s in sessions))
    for s, (ax0, ay0), (bx, by) in zip(sessions, anchors, placed):
        color = s.get("badge_color", C_BADGE)
        if math.hypot(bx - ax0, by - ay0) > 0.004 * (bbox[2] - bbox[0]):
            ax.plot([ax0, bx], [ay0, by], "-", color=color, lw=0.8, alpha=0.7,
                    zorder=9)
        draw_badge(ax, bx, by, s["nr"], size=15, fs=8.5, color=color)

    lat_mid = float(np.mean(lat))
    # North arrow and scale bar in the least busy top and bottom corners.
    load = corner_load(np.array([p[0] for p in placed]),
                       np.array([p[1] for p in placed]), bbox)
    north_c = min(["tl", "tr"], key=lambda k: (load[k], k))
    scale_c = min(["bl", "br"], key=lambda k: (load[k], k))
    draw_scalebar(ax, bbox, lat_mid, scale_c)
    draw_north(ax, bbox, north_c)

    # Cascading insets (e.g. 10 and 100 km), each framing the extent below it.
    spans = [float(s) for s in str(opts.overview_inset_km).split(",") if s.strip()]
    x_ins = m_l + map_w + gap_x
    if not opts.no_inset and not opts.no_map:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if len(spans) > 2:
            print("  (the column holds 2 insets - using the first two "
                  "of --overview-inset-km)")
        frame = bbox
        for i, span in enumerate(spans[:2]):
            y_i = y_map + (map_h - ins_w if i == 0 else 0.0)
            axi = fig.add_axes(rect(x_ins, y_i, ins_w, ins_w))
            frame = draw_on_inset(axi, cx, cy, lat_mid, span, opts,
                                  frame_bbox=frame)
            if frame is None:
                break

    y_attr = (m_b + tab_h + gap + 0.05) / fh
    attr = ("Basemap © OpenStreetMap contributors" if not opts.no_map
            else "No basemap (--no-map)")
    draw_caption(fig, m_l / fw, y_attr, opts.badge_legend)
    fig.text(1.0 - m_r / fw, y_attr, attr,
             fontsize=6.0, color=C_MUTED, ha="right", va="bottom")

    # The table runs full width.
    draw_overview_table(fig.add_axes(rect(m_l, m_b, cw, tab_h)),
                        sessions, opts.overview_method, opts)

    # Cut in the empty gap above the table; the map crop keeps the heading and the
    # OSM attribution.
    y_cut = m_b + tab_h + gap
    bands = {"map": Bbox([[0.0, y_cut - 0.02], [fw, fh]]),
             "table": Bbox([[0.0, 0.0], [fw, m_b + tab_h + 0.10]])}
    return fig, bands


def taper_text(sessions, opts=None):
    """"0.2–0.3 Hz" from the sessions' ana files ("MIXED" if they differ), with
    opts as the fallback."""
    pairs = {(a.get("taper_f1_hz"), a.get("taper_f2_hz"))
             for a in (s["ana"] for s in sessions)
             if a.get("taper_f1_hz") and a.get("taper_f2_hz")}
    if len(pairs) == 1:
        f1, f2 = pairs.pop()
        return f"{float(f1):g}–{float(f2):g} Hz"
    if len(pairs) > 1:
        return "MIXED (sessions do not share a taper)"
    if opts is not None:
        return f"{opts.taper_f1:g}–{opts.taper_f2:g} Hz (from arguments)"
    return DASH


def draw_overview_table(ax, sessions, method="madgwick", opts=None, imu=True):
    """One row per measurement: nr, heading, date, time, duration, drift, Hs/Tp and
    Kystvarslingssenteret's modelled Hs/Tp. imu=False leaves out the buoy's own
    Hs/Tp (the KVS columns move in)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    kvs = [s.get("kvs") for s in sessions]
    has_kvs = any(kvs)
    dy = 1.0 / (len(sessions) + (3.0 if has_kvs else 2.2))
    # Column positions for figure width 7.5".
    cols = [0.000, 0.035, 0.300, 0.400, 0.505, 0.585, 0.650, 0.722, 0.790, 0.900]
    heads = ["#", "Measurement", "Date", "Start (UTC)", "Duration", "Drift",
             "Hs [m]", "Tp [s]"] + (["KVS Hs [m]", "KVS Tp [s]"] if has_kvs else [])
    keep = [j for j in range(len(heads)) if imu or j not in (6, 7)]
    heads = [heads[j] for j in keep]
    if not imu:
        cols = cols[:6] + [0.660, 0.770]
    yy = 1.0
    for cx_, h in zip(cols, heads):
        ax.text(cx_, yy, h, fontsize=7.2, weight="bold", color="#222222", va="top")
    val_kw = dict(fontsize=7.2, color="#111111", va="top",
                  family="DejaVu Sans Mono")
    for i, s in enumerate(sessions):
        yy = 1.0 - (i + 1.30) * dy
        t0, _ = fmt_time(s["ses"].get("start_utc_iso", ""))
        ana, m = s["ana"], method
        vals = [str(s["nr"]),
                s["title"],
                t0.split()[0] if t0 != DASH else DASH,
                t0.split()[-1] if t0 != DASH else DASH,
                fmt_dur(s["ses"].get("duration_ms")),
                f"{s['stats']['net']:.0f} m",
                fmt_val(ana, f"Hs_{m}", dec=3),
                fmt_val(ana, f"Tp_{m}", dec=2),
                f"{kvs[i]['hs_mean']:.3f}" if kvs[i] else DASH,
                f"{kvs[i]['tp_mean']:.2f}" if kvs[i] else DASH]
        vals = [vals[j] for j in keep]
        for j, (cx_, v) in enumerate(zip(cols, vals)):
            if j == 1:                       # the heading is proportional text
                ax.text(cx_, yy, v, fontsize=7.2, color=C_MUTED, va="top")
            else:
                ax.text(cx_, yy, v, **val_kw)
    foot_y = 1.9
    if imu:
        ax.text(0, 1.0 - (len(sessions) + foot_y) * dy,
                f"Wave parameters: {method.capitalize()}. "
                f"Taper {taper_text(sessions, opts)}. "
                "See individual figures for all methods and source file.",
                fontsize=6.4, color=C_MUTED, va="top", style="italic")
        foot_y += 0.7
    if has_kvs:
        dates = ", ".join(sorted({k["downloaded"] for k in kvs if k}))
        ax.text(0, 1.0 - (len(sessions) + foot_y) * dy,
                "KVS: kystvarslingssenteret.no, 100 m grid. "
                f"Downloaded {dates}.",
                fontsize=6.4, color=C_MUTED, va="top", style="italic")


def make_kvs_table(sessions, opts):
    """The overview table without the buoy's own Hs/Tp (KVS only), as its own figure
    of the same width as the overview."""
    import matplotlib.pyplot as plt
    fw, m_l, m_r, m_b = opts.width, 0.45, 0.45, 0.26
    tab_h = 0.56 + 0.175 * len(sessions) + 0.14
    fh = m_b + tab_h + 0.26
    fig = plt.figure(figsize=(fw, fh))
    draw_overview_table(fig.add_axes([m_l / fw, m_b / fh, (fw - m_l - m_r) / fw,
                                      tab_h / fh]),
                        sessions, opts.overview_method, opts, imu=False)
    return fig


# --- Entry point -------------------------------------------------------------
def make_maps(paths, opts):
    """The map figure for each session in paths, and the overview map when there are
    several. paths are session directories or *_gps.csv, in measurement order."""
    opts.config = check_config(opts.config)
    prefer_ipv4()

    # Agg for the whole run (postprocess also draws _spec_acc/_spec_eta.png).
    import matplotlib
    matplotlib.use("Agg")

    if opts.out and len(paths) > 1:
        sys.exit("--out can only be used with a single session; with several, "
                 "<stamp>_map.png is written in each session directory (use "
                 "--overview-out for the overview map).")

    numbers = list(range(1, len(paths) + 1))
    if opts.nr:
        want = [s.strip() for s in opts.nr.split(",") if s.strip()]
        if len(want) != len(paths):
            sys.exit(f"--nr has {len(want)} values, but {len(paths)} sessions were "
                     f"given.")
        numbers = want
    colors = [C_BADGE] * len(paths)
    if opts.badge_colors:
        want = [s.strip() for s in opts.badge_colors.split(",")]
        if len(want) != len(paths):
            sys.exit(f"--badge-colors has {len(want)} values, but {len(paths)} "
                     f"sessions were given.")
        colors = [c or C_BADGE for c in want]

    if opts.init_note:
        for p in paths:
            _, stamp, directory = resolve_session(p)
            print(f"Session: {stamp}  ({directory})")
            write_note_stub(directory, stamp)
        return

    ext = opts.format or (os.path.splitext(opts.out)[1].lstrip(".").lower()
                          if opts.out else "png")
    # No number badge on a single standalone figure.
    show_nr = len(paths) > 1 or bool(opts.nr)

    sessions = []
    for p, nr, color in zip(paths, numbers, colors):
        sess = load_session(p, opts, nr if show_nr else None)
        sess["nr_label"] = nr
        sess["badge_color"] = color
        sessions.append(sess)
        print(f"Measurement {nr}: {sess['stamp']}  ({sess['directory']})")
        if not sess["cfg"]:
            print("  WARNING: found no _cfg.csv - the logging parameters will be empty.")
        if not sess["ana_src"]:
            print("  WARNING: found no _ana_python.csv - the wave parameters "
                  "will be empty.")
        else:
            print(f"  wave parameters from {sess['ana_src']}")
        if sess["gps"]["n_dropped"]:
            print(f"  {sess['gps']['n_dropped']} GPS rows without a 3D fix discarded "
                  f"(of {sess['gps']['n_raw']}).")
        trunc = sess["gps"]["trunc"]           # an interrupted capture
        if trunc.reason:
            print(f"  gps.csv ends after {trunc.rows} rows: {trunc.reason}. "
                  f"{trunc.total - trunc.used} B of {trunc.total} is tail from an "
                  f"interrupted capture (the file sits at its preallocated length).")
        st = sess["stats"]
        print(f"  {len(sess['gps']['t_s'])} points, {st['dist']:.0f} m "
              f"travelled, net {st['net']:.0f} m towards {st['bearing']:.0f}°")
        if not sess["has_note"]:
            print(f"  (no {sess['stamp']}_notat.txt - autogenerated heading. "
                  f"Make one with --init-note.)")

    # The overview first - also for a single session.
    if sessions and not opts.no_overview:
        out_dir = opts.overview_out and os.path.dirname(
            os.path.abspath(opts.overview_out))
        if not out_dir:
            # The parent of the work directories (<config>/), also with one session.
            out_dir = os.path.commonpath([os.path.dirname(s["directory"])
                                          for s in sessions])
        for s in sessions:                     # the overview always numbers
            s["nr"] = s["nr_label"]
        print(f"Overview: {len(sessions)} measurements")
        fig, bands = make_overview(sessions, opts, out_dir)
        out = opts.overview_out or os.path.join(
            out_dir, config_suffix(
                f"{opts.prefix + '_' if opts.prefix else ''}overview.{ext}",
                opts.config))
        fig.savefig(out, dpi=opts.dpi, facecolor="white")
        print(f"  wrote {out}")

        # The map and table crops: foo_overview_<config>.png -> foo_map_<config>.png,
        # foo_table_<config>.png.
        stem = os.path.splitext(out)[0]
        tail = f"_{opts.config}" if opts.config else ""
        if tail and stem.endswith(tail):
            stem = stem[:-len(tail)]
        if stem.endswith("_overview"):
            stem = stem[:-len("_overview")]
        for part, box in (("map", bands["map"]), ("table", bands["table"])):
            part_out = f"{stem}_{part}{tail}.{ext}"
            fig.savefig(part_out, dpi=opts.dpi, facecolor="white",
                        bbox_inches=box)
            trim_png(part_out)          # equal margins, so the two stack centred
            print(f"  wrote {part_out}")
        # ... and the table with only the modelled (KVS) Hs/Tp: foo_table_kvs_<config>.png.
        if any(s.get("kvs") for s in sessions):
            part_out = f"{stem}_table_kvs{tail}.{ext}"
            make_kvs_table(sessions, opts).savefig(
                part_out, dpi=opts.dpi, facecolor="white", bbox_inches="tight")
            trim_png(part_out)
            print(f"  wrote {part_out}")
        for s in sessions:                     # back to the single-figure rule
            s["nr"] = s["nr_label"] if show_nr else None

    if opts.only_overview:
        return

    import matplotlib.pyplot as plt
    for sess in sessions:
        fig = make_figure(sess, opts)
        out = opts.out or os.path.join(
            sess["directory"],
            config_suffix(f"{sess['stamp']}_map.{ext}", opts.config))
        fig.savefig(out, dpi=opts.dpi, facecolor="white")
        plt.close(fig)          # otherwise the figures pile up with many sessions
        print(f"  wrote {out}")
        if opts.psd_table and (opts.psd_table == "all" or sess["stamp"] in
                               [x.strip() for x in opts.psd_table.split(",")]):
            fig = make_psd_table_figure(sess, opts)
            out = os.path.join(sess["directory"], config_suffix(
                f"{sess['stamp']}_psd_table.{ext}", opts.config))
            fig.savefig(out, dpi=opts.dpi, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            trim_png(out, pad_px=2)
            print(f"  wrote {out}")
