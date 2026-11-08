"""Modelled wave height at the time and place of each session, next to the buoy's
own Hs for one taper (run with script_met_waves.py).

  KVS   Kystvarslingssenteret's (NORCE) historical 100 m coastal forecasts, the
        archive behind https://kystvarslingssenteret.no/histmap.html
  IMU   Hs and Tp per filter and GPS from the existing <stamp>_ana_python.csv
        with that taper (written by plot.sh / script_postprocess.py; nothing is
        recomputed).

One table per session: the sources as rows, Hs [m] and Tp [s] as columns.

KVS: POST {"lat", "lon"} to cgi-bin/histforecast.php -> GeoJSON with one feature per
3-hour step from 2019-11 on (hs, tp, dr = mean direction from, model area and grid
point). There is no date filter, so the whole series comes every time (~200 kB
gzipped); it is cached per position. The values are the forecasts closest to real
time (6-12 h old), not hindcasts, unchecked and updated monthly, and wind sea with
periods below 3.5 s is not included. Use requires a reference to
kystvarslingssenteret.no and the download date.

    rows = compare(paths, taper=(0.03, 0.05))     # one dict per session
    print_table(rows); write_csv(rows, path); save_png(rows[0], png_path)
"""

import csv
import gzip
import http.client
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from glob import glob

import numpy as np

import _paths
from opendap import distance_km
from pipeline.session_input import read_kv
from session_points import find_sessions, position, window
from trim import trim_png

ROOT = os.path.dirname(_paths.TOOLS)
URL = "https://kystvarslingssenteret.no/cgi-bin/histforecast.php"
CACHE_DIR = os.path.expanduser("~/.cache/orb_met/kystvarsling")
TIMEOUT_S = 180
STEP_S = 3 * 3600.0          # the archive is 3-hourly

# (<name> in Hs_<name>/Tp_<name> of ana_python.csv, table label)
IMU_METHODS = (("madgwick", "Madgwick"), ("sflp", "SFLP"), ("kalman", "Kalman"),
               ("nxp", "Kalman NXP"), ("gps", "GPS"))

# The CSV: one row per session.
FIELDS = (["session", "start_utc", "minutes", "taper", "lat", "lon",
           "kvs_hs", "kvs_tp", "kvs_dir", "kvs_km", "kvs_model", "kvs_downloaded"]
          + [f"{p}_{name}" for name, _ in IMU_METHODS for p in ("hs", "tp")]
          + ["odr_hz", "output_rate_hz", "sflp_odr_hz", "ana_file", "note"])


# --- KVS ----------------------------------------------------------------------

def _connect_ipv4(address, timeout, source_address=None):
    # The server's IPv6 address does not answer; IPv4 connects at once.
    host, port = address
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
    return socket.create_connection(addr, timeout, source_address)


def _fetch(lat, lon):
    host, path = URL.split("/", 3)[2], "/" + URL.split("/", 3)[3]
    conn = http.client.HTTPSConnection(host, timeout=TIMEOUT_S)
    conn._create_connection = _connect_ipv4
    try:
        conn.request("POST", path, body=json.dumps({"lat": lat, "lon": lon}),
                     headers={"Content-Type": "application/json",
                              "Accept-Encoding": "gzip"})
        r = conn.getresponse()
        if r.status != 200:
            raise OSError(f"HTTP {r.status}")
        body = r.read()
        if r.getheader("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    finally:
        conn.close()
    body = body.decode("utf-8").strip()
    # Outside the model areas the reply is empty.
    return json.loads(body) if body else {}


def point(lat, lon, need_until=None):
    """The archive for the grid point nearest (lat, lon) -> dict with model, lat, lon,
    km (to the grid point), downloaded (epoch s), time (epoch s), hs, tp, dr arrays;
    None outside the model areas. Cached per position, and fetched again when
    need_until (epoch s) is past the end of the cached series."""
    path = os.path.join(CACHE_DIR, f"{lat:.5f}_{lon:.5f}.npz")
    if os.path.isfile(path):
        c = np.load(path, allow_pickle=False)
        pt = {k: (c[k].item() if c[k].ndim == 0 else c[k]) for k in c.files}
        if need_until is None or (len(pt["time"]) and pt["time"][-1] >= need_until):
            return pt
    feats = _fetch(lat, lon).get("features") or []
    if not feats:
        return None
    p0 = feats[0]["properties"]
    t = np.array([datetime.fromisoformat(f["properties"]["time"].replace("Z", "+00:00"))
                  .timestamp() for f in feats])
    pt = dict(model=str(p0.get("model", "")), lat=float(p0["model_lat"]),
              lon=float(p0["model_lon"]),
              km=float(distance_km(lat, lon, p0["model_lat"], p0["model_lon"])),
              downloaded=time.time(), time=t)
    for k in ("hs", "tp", "dr"):
        pt[k] = np.array([f["properties"].get(k) for f in feats], dtype=np.float64)
    # hs = 0 marks a missing forecast (e.g. the start of the archive).
    bad = ~(pt["hs"] > 0.0)
    for k in ("hs", "tp", "dr"):
        pt[k][bad] = np.nan
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **pt)
    return pt


def waves(pt, t0, t1):
    """hs/tp/dr of pt over [t0, t1] (epoch s, UTC) -> dict with hs_mean, tp_mean,
    dr_mean (the 3-hourly series interpolated on a 1-min grid); None when the window
    is not bracketed by valid steps within STEP_S."""
    ok = np.isfinite(pt["hs"])
    tv = pt["time"][ok]
    if not len(tv) or tv[0] > t0 or tv[-1] < t1:
        return None
    k0 = int(np.searchsorted(tv, t0, side="right")) - 1
    k1 = int(np.searchsorted(tv, t1, side="left"))
    if t0 - tv[k0] > STEP_S or tv[k1] - t1 > STEP_S:
        return None                           # a gap in the archive
    tt = np.arange(t0, t1 + 1.0, 60.0)
    out = {f"{k}_mean": float(np.mean(np.interp(tt, tv, pt[k][ok]))) for k in ("hs", "tp")}
    d = np.interp(tt, tv, np.unwrap(np.radians(pt["dr"][ok])))
    out["dr_mean"] = float(np.degrees(np.arctan2(np.mean(np.sin(d)),
                                                 np.mean(np.cos(d)))) % 360.0)
    return out


def session_waves(ses_path):
    """KVS over one session (<stamp>_ses.csv + _gps.csv) -> dict with hs_mean,
    tp_mean, dr_mean, km, model, downloaded (YYYY-MM-DD); None without time,
    position or KVS data."""
    ses = read_kv(ses_path)
    win, pos = window(ses_path, ses), position(ses_path, ses)
    if win is None or pos is None:
        return None
    pt = point(*pos, need_until=win[1])
    if pt is None:
        return None
    k = waves(pt, *win)
    if k is None:
        return None
    k.update(km=float(pt["km"]), model=pt["model"],
             downloaded=f"{datetime.fromtimestamp(pt['downloaded']):%Y-%m-%d}")
    return k


# --- IMU results ----------------------------------------------------------------

def measurement_dir(ses_path):
    """The top directory under the repository that holds the session."""
    d = os.path.dirname(os.path.abspath(ses_path))
    while os.path.dirname(d) not in (ROOT, d):
        d = os.path.dirname(d)
    return d


def find_ana(ses_path, stamp, taper=None, config=None):
    """The session's <stamp>_ana_python.csv -> (path, dict), or (None, None).
    config: the one in <measurement>/<config>/<session folder>/, whatever its taper.
    Else the one with this fixed taper, the config directory named after the taper
    (e.g. 0.03-0.05/) first. Only results under a directory named like the session
    folder count: two folders can hold the same capture (plot.sh RAW_AVERAGE)."""
    meas = measurement_dir(ses_path)
    folder = os.path.basename(os.path.dirname(os.path.abspath(ses_path)))
    if config:
        path = os.path.join(meas, config, folder, f"{stamp}_ana_python.csv")
        return (path, read_kv(path)) if os.path.isfile(path) else (None, None)
    name = f"{taper[0]:g}-{taper[1]:g}"
    found = sorted((p for p in glob(os.path.join(meas, "**", f"{stamp}_ana_python.csv"),
                                    recursive=True)
                    if folder in os.path.relpath(p, meas).split(os.sep)),
                   key=lambda p: os.path.relpath(p, meas).split(os.sep)[0] != name)
    for path in found:
        ana = read_kv(path)
        try:
            if (ana.get("cutoff_mode") == "taper"
                    and abs(float(ana["taper_f1_hz"]) - taper[0]) < 1e-6
                    and abs(float(ana["taper_f2_hz"]) - taper[1]) < 1e-6):
                return path, ana
        except (KeyError, ValueError):
            continue
    return None, None


# --- The table ------------------------------------------------------------------

def _imu(row, ses_path, stamp, taper, config):
    ana_path, ana = find_ana(ses_path, stamp, taper, config)
    if not ana:
        return False
    row["ana_file"] = os.path.relpath(ana_path, ROOT)
    cfg_path = ana_path.replace("_ana_python.csv", "_cfg.csv")
    if os.path.isfile(cfg_path):
        cfg = read_kv(cfg_path)
        for key, src in (("odr_hz", "imu_odr_hz"), ("output_rate_hz", "output_rate_hz"),
                         ("sflp_odr_hz", "sflp_odr_hz")):
            try:
                row[key] = f"{float(cfg[src]):g}"
            except (KeyError, ValueError):
                pass
    if ana.get("cutoff_mode") == "taper":
        row["taper"] = f"{float(ana['taper_f1_hz']):g}-{float(ana['taper_f2_hz']):g}"
    else:
        row["taper"] = ana.get("cutoff_mode", "")
    for name, _ in IMU_METHODS:
        for p, fmt in (("hs", "{:.3f}"), ("tp", "{:.1f}")):
            v = ana.get(f"{p.capitalize()}_{name}")
            if v not in (None, "") and float(v) > 0:
                row[f"{p}_{name}"] = fmt.format(float(v))
    return True


def _kvs(row, pos, t0, t1, notes):
    try:
        pt = point(*pos, need_until=t1)
    except (OSError, ValueError) as e:
        notes.append(f"KVS failed ({e})")
        return
    if pt is None:
        notes.append("outside KVS")
        return
    row.update(kvs_model=pt["model"], kvs_km=f"{pt['km']:.2f}",
               kvs_downloaded=f"{datetime.fromtimestamp(pt['downloaded']):%Y-%m-%d}")
    k = waves(pt, t0, t1)
    if k is None:
        notes.append("no KVS data")
        return
    row.update(kvs_hs=f"{k['hs_mean']:.3f}", kvs_tp=f"{k['tp_mean']:.1f}",
               kvs_dir=f"{k['dr_mean']:.0f}")


def compare(paths, taper=None, all_sessions=False, config=None):
    """One row per session under paths that has IMU results (every session with
    all_sessions): KVS and IMU Hs/Tp -> list of dicts with the keys of FIELDS
    (strings, "" where missing). IMU results come from <measurement>/<config>/
    when config is given, else from any ana_python.csv with this fixed taper."""
    rows = []
    sessions = find_sessions(paths)
    stamps = [os.path.basename(p)[:-len("_ses.csv")] for p in sessions]
    for ses_path, stamp in zip(sessions, stamps):
        ses = read_kv(ses_path)
        row = {k: "" for k in FIELDS}
        # The stamp, or the folder name when two folders hold the same capture.
        folder = os.path.basename(os.path.dirname(os.path.abspath(ses_path)))
        row["session"] = folder if stamps.count(stamp) > 1 else stamp
        row["stamp"], row["folder"] = stamp, folder
        notes = []
        has_imu = _imu(row, ses_path, stamp, taper, config)
        if not has_imu and not all_sessions:
            continue
        win, pos = window(ses_path, ses), position(ses_path, ses)
        if win is None or pos is None:
            notes.append("no GPS time" if win is None else "no position")
        else:
            t0, t1 = win
            row.update(start_utc=datetime.fromtimestamp(t0, tz=timezone.utc)
                       .strftime("%Y-%m-%d %H:%M"),
                       minutes=f"{(t1 - t0) / 60.0:.0f}",
                       lat=f"{pos[0]:.5f}", lon=f"{pos[1]:.5f}")
            print(f"  {stamp}: KVS ...", file=sys.stderr, flush=True)
            _kvs(row, pos, t0, t1, notes)
        if not has_imu:
            notes.append(f"no ana_python.csv in {config}/" if config else
                         f"no ana_python.csv with taper {taper[0]:g}-{taper[1]:g}")
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows


SOURCES = "KVS: kystvarslingssenteret.no, historical wave forecasts, 100 m grid."


def downloaded(kvs_dates):
    """The download line: the date(s) the KVS series was fetched (to be cited)."""
    kvs = sorted(set(kvs_dates) - {""})
    return f"Downloaded {', '.join(kvs) or '-'}."


def session_table(row):
    """[(source label, Hs, Tp)] for one session, the model first."""
    out = [("KVS", row["kvs_hs"], row["kvs_tp"])]
    out += [(label, row[f"hs_{name}"], row[f"tp_{name}"]) for name, label in IMU_METHODS]
    return [(label, hs or "-", tp or "-") for label, hs, tp in out]


def caption(row):
    """Two lines: the session and what the numbers are."""
    return (f"{row['session']}: {row['start_utc'] or '-'} UTC, {row['minutes'] or '-'} min\n"
            f"IMU with taper {row['taper'] or '-'} Hz")


def print_table(rows):
    for row in rows:
        print(caption(row))
        print(f"  {'':<20}  {'Hs [m]':>6}  {'Tp [s]':>6}")
        for label, hs, tp in session_table(row):
            print(f"  {label:<20}  {hs:>6}  {tp:>6}")
        if row["note"]:
            print(f"  note: {row['note']}")
        print()
    dl = [r["kvs_downloaded"] for r in rows]
    print(SOURCES)
    print(downloaded(dl))


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_rows(rows, path):
    """compare()'s rows with every key (stamp and folder too) -> the data file the
    figures are drawn from. The values are strings and stay so."""
    keys = list(FIELDS) + [k for r in rows for k in r if k not in FIELDS]
    keys = list(dict.fromkeys(keys))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote data/{os.path.basename(path)}")


def read_rows(path):
    """write_rows' file -> the rows, as compare() returned them."""
    if not os.path.isfile(path):
        sys.exit(f"--plot-only: {path} is missing - run script_met_waves.py without "
                 f"--plot-only first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_png(row, path, title=""):
    """One session's table as a figure."""
    import matplotlib.pyplot as plt

    cells = [list(r) for r in session_table(row)]
    fig, ax = plt.subplots(figsize=(4.2, 0.25 * (len(cells) + 1)))
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=["", "Hs [m]", "Tp [s]"],
                   cellLoc="center", colWidths=[0.5, 0.25, 0.25], bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (i, j), cell in tab.get_celld().items():
        if i == 0 or j == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        if j == 0 and i > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
        if i == 0 and j == 0:
            cell.set_facecolor("white")
    ax.set_title("\n".join(t for t in (title, caption(row)) if t), fontsize=8)
    ax.text(0.5, -0.04, SOURCES + "\n"
            + downloaded([row["kvs_downloaded"]]),
            transform=ax.transAxes, ha="center", va="top", fontsize=6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    trim_png(path)


def combined_table(rows):
    """[(source label, [Hs, Tp, Hs, Tp, ...])] for several sessions side by side, in
    the order of rows."""
    tables = [session_table(r) for r in rows]
    return [(tables[0][k][0], [v for t in tables for v in t[k][1:]])
            for k in range(len(tables[0]))]


def print_combined(rows, labels):
    print("  " + " " * 20 + "".join(f"  {lab:^14s}" for lab in labels))
    print("  " + " " * 20 + "  " + "  ".join(f"{'Hs [m]':>6}  {'Tp [s]':>6}" for _ in labels))
    for label, vals in combined_table(rows):
        print(f"  {label:<20}  " + "  ".join(f"{v:>6}" for v in vals))


def save_grouped_table(groups, subheads, body, path, footer="", extra_heads=(),
                       corner="", row_labels=True, col_width=1.1):
    """A table with a row label column, then one group of columns per entry in
    groups (its label drawn across the group) with the column heads subheads under
    it, and extra_heads (more header rows, [label, value, ...]) under those.
    subheads is one list for every group, or a list of lists (one per group, of any
    length). corner heads the row label column. body = [[row label, value, ...], ...].
    row_labels=False: no row label column - body rows hold the group columns only.
    col_width: the width of each group column [in], or a list with one per column."""
    import matplotlib.pyplot as plt

    per = ([list(h) for h in subheads] if subheads and isinstance(subheads[0], (list, tuple))
           else [list(subheads)] * len(groups))
    lab = 1 if row_labels else 0             # the row label column, or none
    ncol = lab + sum(len(h) for h in per)
    heads = ([[""] * ncol, [corner] * lab + [h for hs in per for h in hs]]
             + [list(r) for r in extra_heads])
    nhead = len(heads)
    cells = heads + body
    cw = (list(col_width) if isinstance(col_width, (list, tuple))
          else [col_width] * (ncol - lab))
    total = 1.6 * lab + sum(cw)
    widths = [1.6 / total] * lab + [w / total for w in cw]
    # Column j (after the label column) -> (group, position, group size).
    where = {}
    j = lab
    for n, hs in enumerate(per):
        for pos in range(len(hs)):
            where[j] = (n, pos, len(hs))
            j += 1
    fig, ax = plt.subplots(figsize=(total, 0.25 * len(cells)))
    ax.axis("off")
    tab = ax.table(cellText=cells, cellLoc="center", colWidths=widths, bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (i, j), cell in tab.get_celld().items():
        if not row_labels:
            j += 1                           # the styling below counts the label column
        if i < nhead or j == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        if j == 0 and i > 1:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
        if 1 < i < nhead and j > 0:
            cell.set_text_props(weight="normal")
        if j == 0 and i <= 1:
            cell.set_facecolor("white")
            cell.visible_edges = "" if i == 0 else "B"
        if i == 0 and j > 0:
            # One label over each group: no inner edges. An open edge path also
            # clips the fill, so the label row is left unfilled.
            cell.set_facecolor("white")
            _, pos, k = where[j - 1 + lab]
            cell.visible_edges = ("TB" + ("L" if pos == 0 else "")
                                  + ("R" if pos == k - 1 else ""))
    start = lab
    for text, hs in zip(groups, per):
        x0 = sum(widths[:start])
        x = x0 + sum(widths[start:start + len(hs)]) / 2.0     # middle of the group
        ax.text(x, 1.0 - 0.5 / len(cells), text, transform=ax.transAxes, ha="center",
                va="center", fontsize=8, weight="bold")
        start += len(hs)
    if footer:
        ax.text(0.5, -0.04, footer, transform=ax.transAxes, ha="center", va="top",
                fontsize=6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    trim_png(path)


def save_combined_png(rows, labels, path):
    """Several sessions in one table: the sources as rows, an Hs/Tp column pair per
    session under a header row with labels (e.g. the measurement names). No heading:
    the caption (with the taper) goes in LaTeX."""
    body = [[label] + vals for label, vals in combined_table(rows)]
    save_grouped_table(labels, ["Hs [m]", "Tp [s]"], body, path,
                       SOURCES + "\n" + downloaded([r["kvs_downloaded"] for r in rows]))


# (<name> in Hs_<name> of ana_python.csv, row label) for the A/B table.
AB_METHODS = (("sflp", "SFLP"), ("madgwick", "Madgwick"), ("kalman", "Kalman"),
              ("nxp", "Kalman NXP"))


def ab_table(pairs):
    """[[filter, Hs A, Hs B, B - A, ...]] for [(row A, row B), ...] (rows from
    compare()), one row per filter in AB_METHODS."""
    body = []
    for name, label in AB_METHODS:
        line = [label]
        for a, b in pairs:
            ha, hb = a[f"hs_{name}"], b[f"hs_{name}"]
            diff = f"{float(hb) - float(ha):+.3f}" if ha and hb else "-"
            line += [ha or "-", hb or "-", diff]
        body.append(line)
    return body


def print_ab(pairs, labels):
    print("  " + " " * 12 + "".join(f"  {lab:^24s}" for lab in labels))
    print("  " + " " * 12 + "".join(f"  {'A':>6}  {'B':>6}  {'B-A':>6}" for _ in labels))
    for line in ab_table(pairs):
        print(f"  {line[0]:<12}" + "".join(f"  {v:>6}" for v in line[1:]))


def save_ab_png(pairs, labels, path):
    """Hs per filter for buoy A and B side by side, a column group per location."""
    def hz(row, key):
        return f"{row[key]} Hz" if row[key] else "-"
    extra = [["ODR"] + [v for a, b in pairs for v in (hz(a, "odr_hz"), hz(b, "odr_hz"), "")],
             ["Output rate"] + [v for a, b in pairs
                                for v in (hz(a, "output_rate_hz"), hz(b, "output_rate_hz"), "")]]
    save_grouped_table(labels, ["A Hs [m]", "B Hs [m]", "B − A [m]"], ab_table(pairs),
                       path, extra_heads=extra)
