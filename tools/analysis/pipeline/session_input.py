"""Steps 1-2, sensing and on-sensor configuration: one logged session -> Session.

Reads <stamp>_imu.csv (the logged rows), _gps.csv (vUp), _cfg.csv (the logging
parameters) and _ses.csv. Everything downstream works on the Session arrays.

    session = read_session(path)      # directory or *_imu.csv
"""

import os
import sys
from collections import namedtuple
import dataclasses
from dataclasses import dataclass
from glob import glob

import numpy as np


@dataclass
class GpsVup:
    """GPS vertical velocity on the imu.csv time axis."""
    t_s: np.ndarray              # [s]
    vup: np.ndarray              # [m/s]


@dataclass
class Session:
    """One logged session. Row arrays have one entry per imu.csv row."""
    imu_path: str
    stamp: str
    directory: str
    cfg: dict                    # <stamp>_cfg.csv (build and logging constants)
    ses: dict                    # <stamp>_ses.csv (per-run values)
    row_dt_ms: float             # cfg window_ms, or None
    t_ms: np.ndarray             # win_start_ms, int
    acc_mg: np.ndarray           # (N, 3) ax, ay, az [mg]
    gyro_mdps: np.ndarray        # (N, 3) gx, gy, gz [mdps]
    az_ned_sflp: np.ndarray      # the chip's vertical accel [mg]
    braking: np.ndarray          # 0/1
    ovf: np.ndarray              # 0/1 fifo overflow (0 without the column)
    vacc_raw: np.ndarray         # AHRS on the raw stream [m/s²], or None
    vacc_col: str                # the column vacc_raw came from, or None
    n_raw: np.ndarray            # raw samples behind each row, or None
    gps: GpsVup                  # or None

    @property
    def orientation_name(self):
        """The capture's own AHRS as cfg.csv names it."""
        return self.cfg.get("orientation_name", "madgwick")

    def actual_odr(self):
        """The sensor's actual sample rate [Hz] from the n column (sum of raw samples
        over the span plus one row period), or None."""
        if self.n_raw is None or len(self.n_raw) < 2:
            return None
        t = self.t_ms.astype(np.float64)
        dt = np.diff(t)
        dt = dt[dt > 0]
        if not len(dt):
            return None
        span_ms = (t[-1] - t[0]) + float(np.median(dt))
        return float(sum(self.n_raw.tolist())) / (span_ms / 1000.0) if span_ms > 0 else None

    def cfg_number(self, key):
        """A positive number from cfg.csv, or None."""
        try:
            v = float(self.cfg.get(key, ""))
            return v if v > 0 else None
        except ValueError:
            return None


def resolve_paths(path):
    """Find imu.csv + stamp + session directory from a directory or file path."""
    if os.path.isdir(path):
        cand = sorted(glob(os.path.join(path, "*_imu.csv")))
        if not cand:
            sys.exit(f"Found no *_imu.csv in {path}")
        imu = cand[0]
    elif os.path.isfile(path):
        imu = path
    else:
        sys.exit(f"Did not find {path}")
    directory = os.path.dirname(os.path.abspath(imu))
    stamp = os.path.basename(imu)[:-len("_imu.csv")]
    return imu, stamp, directory


def imu_col_names(idx):
    """Column names for this imu.csv version. Before build_seq 3 the SFLP columns had
    no _sflp suffix (recognised by the column 'mqw')."""
    legacy = "mqw" in idx
    return dict(
        azn="az_ned" if legacy else "az_ned_sflp",
        vacc="vacc_madgwick" if legacy else "vacc",
    )


# How much of a stream file was read; reason is None when all of it was.
CsvTruncation = namedtuple("CsvTruncation", "rows used total reason")


def truncation_message(path, k):
    return (f"  {os.path.basename(path)}: {k.rows} whole rows, "
            f"{k.total - k.used} B tail discarded ({k.reason})")


def read_csv_stream(path, time_col):
    """Read a streamed CSV (imu.csv / gps.csv) -> (idx, rows as string lists,
    CsvTruncation). Used by every stream reader, including mapplot.

    An interrupted capture leaves old card content after the last row, so the file
    is read as bytes and the data ends at the first row with the wrong number of
    fields, a non-numeric field, or time_col that does not increase."""
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    header = [c.decode("ascii", "replace") for c in lines[0].rstrip(b"\r").split(b",")]
    idx = {nm: i for i, nm in enumerate(header)}
    if time_col not in idx:
        return idx, [], CsvTruncation(0, len(lines[0]) + 1, len(data),
                                      f"missing the column '{time_col}'")

    rows, used, prev, reason = [], len(lines[0]) + 1, None, None
    for ln in lines[1:]:
        if not ln.strip():
            continue                       # blank line (the final \n) - not a tail
        fld = ln.rstrip(b"\r").split(b",")
        if len(fld) != len(header):
            reason = "incomplete row"
            break
        try:
            t = float(fld[idx[time_col]])
            row = [v.decode("ascii") for v in fld]
        except (ValueError, UnicodeDecodeError):
            reason = "non-numeric field"
            break
        if prev is not None and t <= prev:
            reason = f"time did not advance ({t:.0f} after {prev:.0f})"
            break
        prev = t
        rows.append(row)
        used += len(ln) + 1
    return idx, rows, CsvTruncation(len(rows), used, len(data), reason)


def read_kv(path):
    """Read a simple key,value file (ses.csv / cfg.csv) into a dict."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split(",", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def read_gps_vup(gps_path):
    """gps.csv -> GpsVup on the imu.csv time axis, or None."""
    if not os.path.isfile(gps_path):
        return None
    idx, rows, trunc = read_csv_stream(gps_path, "rel_ms")
    if "vUp" not in idx:
        return None
    if trunc.reason:
        print(truncation_message(gps_path, trunc))
    ts = np.array([float(r[idx["rel_ms"]]) / 1000.0 for r in rows])
    vs = np.array([float(r[idx["vUp"]]) for r in rows])
    if len(ts) < 2:
        return None
    return GpsVup(ts, vs)


def _read_imu(imu_path):
    """imu.csv -> the Session row arrays as a dict."""
    idx, rows, trunc = read_csv_stream(imu_path, "win_start_ms")
    cols = imu_col_names(idx)
    need = ["win_start_ms", "ax_mg", "ay_mg", "az_mg", cols["azn"],
            "gx_mdps", "gy_mdps", "gz_mdps", "braking"]
    for nm in need:
        if nm not in idx:
            sys.exit(f"Missing column '{nm}' in {imu_path}")
    if trunc.reason:
        print(truncation_message(imu_path, trunc))

    def col(name, conv=float):
        i = idx[name]
        return [conv(r[i]) for r in rows]

    def flag(name):
        i = idx[name]
        return [1 if r[i].strip() == "1" else 0 for r in rows]

    # vacc_fir (FIR decimated), else the unfiltered centre-tap column.
    vacc_col = ("vacc_fir" if "vacc_fir" in idx
                else cols["vacc"] if cols["vacc"] in idx else None)
    return dict(
        t_ms=np.array(col("win_start_ms", int), dtype=np.int64),
        acc_mg=np.array([col("ax_mg"), col("ay_mg"), col("az_mg")],
                        dtype=np.float64).T.reshape(-1, 3),
        gyro_mdps=np.array([col("gx_mdps"), col("gy_mdps"), col("gz_mdps")],
                           dtype=np.float64).T.reshape(-1, 3),
        az_ned_sflp=np.array(col(cols["azn"]), dtype=np.float64),
        braking=np.array(flag("braking"), dtype=np.int64),
        # fifo_ovf from build 55; 0 without the column.
        ovf=(np.array(flag("fifo_ovf"), dtype=np.int64) if "fifo_ovf" in idx
             else np.zeros(len(rows), dtype=np.int64)),
        vacc_raw=(np.array(col(vacc_col), dtype=np.float64) if vacc_col else None),
        vacc_col=vacc_col,
        n_raw=(np.array(col("n", int), dtype=np.int64) if "n" in idx else None),
    )


def truncate_session(session, minutes):
    """The session as if the capture had stopped after `minutes`: rows and GPS from
    then on are dropped."""
    end_ms = minutes * 60000.0
    keep = session.t_ms < end_ms
    rows = {name: (getattr(session, name)[keep] if getattr(session, name) is not None
                   else None)
            for name in ("t_ms", "acc_mg", "gyro_mdps", "az_ned_sflp", "braking", "ovf",
                         "vacc_raw", "n_raw")}
    gps = session.gps
    if gps is not None:
        g = gps.t_s < end_ms / 1000.0
        gps = GpsVup(t_s=gps.t_s[g], vup=gps.vup[g])
    return dataclasses.replace(session, gps=gps, **rows)


def read_session(path):
    """The session at path (directory or *_imu.csv) -> Session. Prints what it read."""
    imu, stamp, directory = resolve_paths(path)
    print(f"=== POST-PROCESSING {stamp} ===")
    print(f"  imu:   {imu}")

    ses = read_kv(os.path.join(directory, f"{stamp}_ses.csv"))
    if ses:
        # Only the keys present (an interrupted session lacks some).
        shown = [f"{k}={ses[k]}" for k in ("start_utc_iso", "gps_rows", "duration_ms")
                 if k in ses]
        if shown:
            print("  ses:   " + "  ".join(shown))

    # Build identity from cfg.csv (older sessions: build_seq in ses.csv). Only the
    # logging parameters are used from cfg.csv.
    cfg = read_kv(os.path.join(directory, f"{stamp}_cfg.csv"))
    build = [f"{k}={cfg[k]}" for k in ("build_date", "build_commit", "build_branch")
             if cfg and k in cfg]
    if not build and ses and "build_seq" in ses:
        build = [f"build_seq={ses['build_seq']}"]
    if build:
        print("  cfg:   " + "  ".join(build))
    row_dt_ms = None
    if cfg:
        shown = [f"{k}={cfg[k]}" for k in
                 ("imu_odr_hz", "output_rate_hz", "window_ms", "gps_rate_hz",
                  "log_duration_sec")
                 if k in cfg]
        if shown:
            print("         " + "  ".join(shown))
        try:
            w = float(cfg["window_ms"])
            row_dt_ms = w if w > 0.0 else None
        except (KeyError, ValueError):
            row_dt_ms = None

    arrays = _read_imu(imu)
    print(f"  read {len(arrays['t_ms'])} imu rows")

    gps = read_gps_vup(os.path.join(directory, f"{stamp}_gps.csv"))
    if gps is None:
        print("  gps:   no usable *_gps.csv (vUp) - GPS check omitted")
    else:
        print(f"  gps:   {len(gps.t_s)} fixes, "
              f"{len(gps.t_s) / max(gps.t_s[-1] - gps.t_s[0], 1e-9):.2f} Hz")

    return Session(imu_path=imu, stamp=stamp, directory=directory, cfg=cfg, ses=ses,
                   row_dt_ms=row_dt_ms, gps=gps, **arrays)
