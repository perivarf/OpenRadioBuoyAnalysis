"""OPeNDAP access to thredds.met.no with the standard library only."""

import urllib.request

import numpy as np

TIMEOUT_S = 120


def fetch_ascii(url):
    """An OPeNDAP .ascii response -> list of floats in the first data block (for a
    Grid: the array itself, not its maps)."""
    url = url.replace("[", "%5B").replace("]", "%5D")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        text = r.read().decode("ascii", "replace")
    body = text.split("---------------------------------------------", 1)[-1]
    lines = [ln.strip() for ln in body.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    values = []
    for line in lines[1:]:                    # lines[0] is "var[n]..." (a header)
        if not line:
            break
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].startswith("["):
            parts = parts[1:]                 # "[j], v, v, ..." row index
        values.extend(float(p) for p in parts if p)
    return values



def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance [km] (haversine), broadcasting over arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
