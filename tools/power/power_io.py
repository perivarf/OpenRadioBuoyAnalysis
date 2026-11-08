"""Files of the power measurements: <prefix>_raw.csv (samples) and <prefix>_sum.csv (summary)."""

import csv
import glob
import os

import numpy as np

# power_measurement/ in the repo root, the default input and output of the power scripts
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "power_measurement")

RAW_HEADER = ["t_s", "voltage_V", "current_A", "power_W"]


def strip_prefix(path):
    """test1, test1.csv, test1_raw.csv and test1_sum.csv all give test1."""
    if path.endswith(".csv"):
        path = path[:-4]
    for suffix in ("_raw", "_sum"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return path


def unique_prefix(prefix):
    """A prefix whose _raw.csv does not exist yet: batch1 -> batch1_2 -> batch1_3 ..."""
    base, n = prefix, 1
    while os.path.exists(f"{prefix}_raw.csv"):
        n += 1
        prefix = f"{base}_{n}"
    return prefix


def raw_row(t, v, i, p):
    return [f"{t:.3f}", v, i, f"{p:.6f}"]


def read_raw(path):
    """(t [s], V, A, W) as arrays; tolerates a half-written last line."""
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    t, v, i, p = (np.atleast_1d(d[k]) for k in RAW_HEADER)
    ok = np.isfinite(t) & np.isfinite(i)
    return t[ok], v[ok], i[ok], p[ok]


def write_raw(path, t, v, i, p):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_HEADER)
        for row in zip(t, v, i, p):
            w.writerow(raw_row(*row))


def read_sum(path):
    """The summary as {quantity: value} ({} if the file is missing)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["quantity"]: row["value"] for row in csv.DictReader(f)}


def write_sum(path, items):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerows(items)


def read_legend_names(path):
    """name,legendname per line (# comments; "text, with commas" in quotes)."""
    if not os.path.exists(path):
        return {}
    names = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, skipinitialspace=True):
            if len(row) >= 2 and not row[0].strip().startswith("#"):
                names[row[0].strip()] = row[1].strip()
    return names


def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None
