"""Time, position and drift of each logged session, for comparison with models.

    for ses_path in find_sessions(paths):
        ses = read_kv(ses_path)
        window(ses_path, ses)      # (t0, t1) epoch s UTC, or None
        position(ses_path, ses)    # (lat, lon) mean, or None
        drift(ses_path, ses)       # (m/s, direction towards [deg]), or None
"""

import math
import os
import re
from glob import glob

import numpy as np

import _paths                                               # noqa: F401
from pipeline.session_input import read_csv_stream

# Directories under a measurement that are not sessions of their own.
SKIP_DIR = re.compile(r"^([0-9.]+-[0-9.]+|taper_sweep|.*_raw)$")


def find_sessions(paths):
    """Every <stamp>_ses.csv under paths, outside config/raw work directories."""
    out = []
    for p in paths:
        for ses in sorted(glob(os.path.join(p, "**", "*_ses.csv"), recursive=True)):
            rel = os.path.relpath(os.path.dirname(ses), p)
            if not any(SKIP_DIR.match(part) for part in rel.split(os.sep)):
                out.append(ses)
    return out



def read_gps(ses_path):
    """<stamp>_gps.csv beside ses.csv -> (column index, rows), empty if missing."""
    gps = ses_path.replace("_ses.csv", "_gps.csv")
    if not os.path.isfile(gps):
        return {}, []
    idx, rows, _ = read_csv_stream(gps, "rel_ms")
    return idx, rows



def position(ses_path, ses):
    """(lat, lon) for the session: ses.csv start/stop mean, else gps.csv mean."""
    try:
        lats = [float(ses[k]) for k in ("lat_start", "lat_stop") if k in ses]
        lons = [float(ses[k]) for k in ("lon_start", "lon_stop") if k in ses]
        if lats and lons and all(abs(v) > 1e-6 for v in lats + lons):
            return float(np.mean(lats)), float(np.mean(lons))
    except ValueError:
        pass
    idx, rows = read_gps(ses_path)
    # lat/lon in degrees, or lat_e7/lng_e7 (older firmware) in 1e-7 degrees.
    for la, lo, scale in (("lat", "lon", 1.0), ("lat_e7", "lng_e7", 1e-7)):
        if la in idx and lo in idx:
            pts = [(float(r[idx[la]]) * scale, float(r[idx[lo]]) * scale) for r in rows]
            pts = [p for p in pts if abs(p[0]) > 1e-6 and abs(p[1]) > 1e-6]
            if pts:
                return tuple(float(v) for v in np.mean(pts, axis=0))
    return None



def window(ses_path, ses):
    """(t0, t1) in epoch seconds UTC, or None. Without stop time or duration in
    ses.csv, the end is the last gps.csv row (rel_ms)."""
    try:
        t0 = float(ses["start_utc_epoch"])
    except (KeyError, ValueError):
        return None
    if t0 < 1e9:                              # no GPS time at start
        return None
    try:
        t1 = float(ses["stop_utc_epoch"])
    except (KeyError, ValueError):
        t1 = t0 + float(ses.get("duration_ms", 0)) / 1000.0
    if t1 <= t0:
        idx, rows = read_gps(ses_path)
        if rows:
            t1 = t0 + float(rows[-1][idx["rel_ms"]]) / 1000.0
    return t0, max(t1, t0)



def drift(ses_path, ses):
    """Net drift from the first to the last GPS fix -> (speed m/s, direction the
    buoy moved towards, degrees from north), or None."""
    idx, rows = read_gps(ses_path)
    for la, lo, scale in (("lat", "lon", 1.0), ("lat_e7", "lng_e7", 1e-7)):
        if la in idx and lo in idx:
            pts = [(float(r[idx["rel_ms"]]) / 1000.0, float(r[idx[la]]) * scale,
                    float(r[idx[lo]]) * scale) for r in rows]
            pts = [p for p in pts if abs(p[1]) > 1e-6 and abs(p[2]) > 1e-6]
            if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
                return None
            (ta, lat_a, lon_a), (tb, lat_b, lon_b) = pts[0], pts[-1]
            dn = math.radians(lat_b - lat_a) * 6371000.0
            de = math.radians(lon_b - lon_a) * 6371000.0 * math.cos(math.radians(lat_a))
            return (math.hypot(dn, de) / (tb - ta),
                    math.degrees(math.atan2(de, dn)) % 360.0)
    return None
