"""Surface current, wind and sea level from MET Norway's NorKyst v3 ocean model at a
point and time window, through OPeNDAP (thredds.met.no, fou-hi/norkystv3).

NorKyst v3 (ROMS) has no waves: zeta is the sea surface elevation (tide and surge,
hourly), not wave height. What it adds is the surface current (to compare with the
buoy's drift), the forcing wind and the sea level.

  norkystv3_his_files/YYYY/MM/DD/<model>_his_zdepth_YYYYMMDDT00Z_<member>_AN.nc
    norkyst160 m71   160 m, Oslofjord (59.24-60.03 N, 9.9-11.4 E)
    norkyst800 m00   800 m, the whole coast (used where 160 m does not reach,
                     e.g. Hvaler)
  24 hourly steps per file; u_eastward/v_northward at 15 depths (index 0 = 0 m),
  Uwind_eastward/Vwind_northward and zeta; Int16 x 0.001.

    cell = nearest_wet_cell(59.0, 11.08, t0)     # tries 160 m, then 800 m
    m = series(cell, t0, t1)                     # epoch seconds, UTC
"""

import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from opendap import distance_km, fetch_ascii

BASE = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_his_files"
CACHE_DIR = os.path.expanduser("~/.cache/orb_met")
FILL = -32767
SCALE = 0.001

# (model, member, label, (Y, X) grid shape), in the order they are tried.
DOMAINS = (("norkyst160", "m71", "NorKyst 160 m Oslofjord", (377, 417)),
           ("norkyst800", "m00", "NorKyst 800 m", (1148, 2747)))
COARSE_STRIDE = 8            # stride of the cached coarse lat/lon used to search
WINDOW = 12                  # full-resolution cells either side of the coarse hit


def day_url(domain, day):
    """The daily analysis file of domain for a UTC date."""
    model, member = domain[0], domain[1]
    return (f"{BASE}/{day:%Y/%m/%d}/{model}_his_zdepth_{day:%Y%m%d}T00Z_"
            f"{member}_AN.nc")


def _day(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)


def _array(url, var, *ranges):
    """var[ranges] as a float array (no scaling)."""
    sl = "".join(f"[{a}:1:{b}]" for a, b in ranges)
    return np.array(fetch_ascii(f"{url}.ascii?{var}{sl}"), dtype=np.float64)


def coarse_grid(domain, day):
    """Every COARSE_STRIDE-th lat/lon of domain -> (lat, lon), cached."""
    path = os.path.join(CACHE_DIR, f"{domain[0]}_{domain[1]}_coarse.npz")
    if os.path.isfile(path):
        g = np.load(path)
        return g["lat"], g["lon"]
    ny, nx = domain[3]
    url = day_url(domain, day)
    sl = f"[0:{COARSE_STRIDE}:{ny - 1}][0:{COARSE_STRIDE}:{nx - 1}]"
    shape = (len(range(0, ny, COARSE_STRIDE)), len(range(0, nx, COARSE_STRIDE)))
    lat = np.array(fetch_ascii(f"{url}.ascii?lat{sl}")).reshape(shape)
    lon = np.array(fetch_ascii(f"{url}.ascii?lon{sl}")).reshape(shape)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, lat=lat, lon=lon)
    return lat, lon


def _nearest_in(domain, lat, lon, day):
    """The nearest wet cell of one domain -> (j, i, cell lat, cell lon, km)."""
    clat, clon = coarse_grid(domain, day)
    d = distance_km(lat, lon, clat, clon)
    cj, ci = np.unravel_index(int(np.argmin(d)), d.shape)
    ny, nx = domain[3]
    j0 = max(0, cj * COARSE_STRIDE - WINDOW)
    j1 = min(ny - 1, cj * COARSE_STRIDE + WINDOW)
    i0 = max(0, ci * COARSE_STRIDE - WINDOW)
    i1 = min(nx - 1, ci * COARSE_STRIDE + WINDOW)
    url = day_url(domain, day)
    shape = (j1 - j0 + 1, i1 - i0 + 1)
    wlat = _array(url, "lat", (j0, j1), (i0, i1)).reshape(shape)
    wlon = _array(url, "lon", (j0, j1), (i0, i1)).reshape(shape)
    zeta = _array(url, "zeta", (0, 0), (j0, j1), (i0, i1)).reshape(shape)
    d = np.where(zeta != FILL, distance_km(lat, lon, wlat, wlon), np.inf)
    j, i = np.unravel_index(int(np.argmin(d)), d.shape)
    return j0 + int(j), i0 + int(i), float(wlat[j, i]), float(wlon[j, i]), float(d[j, i])


def nearest_wet_cell(lat, lon, t, max_km=2.0):
    """The nearest wet NorKyst cell to (lat, lon): 160 m Oslofjord if it has one
    within max_km, else 800 m -> (domain, j, i, cell lat, cell lon, km)."""
    day = _day(t)
    best = None
    for domain in DOMAINS:
        cell = (domain,) + _nearest_in(domain, lat, lon, day)
        if cell[5] <= max_km:
            return cell
        if best is None or cell[5] < best[5]:
            best = cell
    return best


def series(cell, t0, t1):
    """Hourly surface current, wind and zeta at cell over [t0, t1] (epoch s, UTC),
    from the daily files the window touches -> dict, or None when a file is missing.

      time, u, v (current, 0 m) [m/s], uw, vw (wind) [m/s], zeta [m]
      cur_speed, cur_dir   mean current vector, direction it flows TOWARDS [deg]
      wind_speed, wind_dir mean wind vector, direction it blows FROM [deg]
      zeta_mean, zeta_min, zeta_max
    """
    domain, j, i = cell[0], cell[1], cell[2]
    cols = {k: [] for k in ("time", "u", "v", "uw", "vw", "zeta")}
    day = _day(t0)
    # One hour either side, so the window can be interpolated.
    while day.timestamp() <= t1 + 3600.0:
        url = day_url(domain, day)
        try:
            times = _array(url, "time")
        except OSError:
            return None
        sel = np.where((times >= t0 - 3600.0) & (times <= t1 + 3600.0))[0]
        if len(sel):
            k0, k1 = int(sel[0]), int(sel[-1])
            cols["time"].extend(times[k0:k1 + 1])
            for key, var, depth in (("u", "u_eastward", True), ("v", "v_northward", True),
                                    ("uw", "Uwind_eastward", False),
                                    ("vw", "Vwind_northward", False),
                                    ("zeta", "zeta", False)):
                rng = [(k0, k1)] + ([(0, 0)] if depth else []) + [(j, j), (i, i)]
                v = _array(url, var, *rng)
                cols[key].extend(np.where(v == FILL, np.nan, v * SCALE))
        day += timedelta(days=1)
    out = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    if len(out["time"]) < 2:
        return None

    tt = np.arange(t0, t1 + 1.0, 60.0)

    def mean(key):
        return float(np.nanmean(np.interp(tt, out["time"], out[key])))

    u, v, uw, vw = mean("u"), mean("v"), mean("uw"), mean("vw")
    z = np.interp(tt, out["time"], out["zeta"])
    out.update(cur_speed=math.hypot(u, v),
               cur_dir=math.degrees(math.atan2(u, v)) % 360.0,
               wind_speed=math.hypot(uw, vw),
               wind_dir=math.degrees(math.atan2(-uw, -vw)) % 360.0,
               zeta_mean=float(np.nanmean(z)), zeta_min=float(np.nanmin(z)),
               zeta_max=float(np.nanmax(z)))
    return out
