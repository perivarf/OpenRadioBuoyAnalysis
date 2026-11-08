"""Step 5, post-AHRS decimation: VerticalAccel (row rate) -> AnalysisSeries.

The rows are grouped in buckets of bucket_ms (t // bucket_ms); "mean" takes the
bucket mean (firmware), "fir" a FIR low-pass + decimation on the same bucket axis
(fir.py, as the sfy buoy). Missing buckets are filled by linear interpolation so
the time axis is uniform, and flagged in the gap mask. GPS vUp is interpolated to
the same bucket centres.

    series = decimate(vacc, session.gps, st.decimation, session.row_dt_ms)
"""

import sys
from dataclasses import dataclass

import numpy as np

import _paths                         # noqa: F401
import fir
from .ahrs import KEYS
from .settings import GAP_MAX_FILL


@dataclass
class AnalysisSeries:
    """Step 5 output: every series on one uniform bucket axis at fs."""
    fs: float                    # the analysis rate [Hz]
    bucket_ms: int
    idx: np.ndarray              # bucket index per sample (time = idx·bucket_ms)
    acc: dict                    # {key: vertical accel [m/s²]}
    gps_vup: np.ndarray          # GPS vUp [m/s] (mean removed), or None
    brake: np.ndarray            # 1 where any row in the bucket was braking
    ovf: np.ndarray              # 1 where any row had a fifo overflow
    gap: np.ndarray              # 1 where the bucket is filled (or FIR edge/smear)
    gapstats: dict               # n_gaps, n_filled, longest, n_long, span
    firstats: dict               # FIR details, or None in mean mode
    mode: str                    # "mean" | "fir"
    n_rows: int
    n_brake: int                 # braking rows
    n_ovf: int                   # fifo overflow rows


def fill_bucket_gaps(bidx, series, masks, max_fill=GAP_MAX_FILL):
    """Fill the missing buckets between bidx[0] and bidx[-1] so the series is time
    uniform.

      bidx   = bucket indices (strictly increasing, may have gaps)
      series = {name: values}, interpolated linearly
      masks  = {name: 0/1 flags}; a filled bucket is 1 if either neighbour is

    Returns (full_idx, series_out, masks_out, gap, stats); gap is 1 for every
    filled bucket. max_fill only sets which gaps count as long in stats."""
    b = np.asarray(bidx, dtype=np.int64)
    stats = dict(n_gaps=0, n_filled=0, longest=0, n_long=0, span=len(b))
    if len(b) < 2:
        return (b,
                {k: np.asarray(v, dtype=np.float64) for k, v in series.items()},
                {k: np.asarray(v, dtype=np.float64) for k, v in masks.items()},
                np.zeros(len(b), dtype=np.float64), stats)

    d = np.diff(b)
    holes = d[d > 1] - 1
    stats["n_gaps"] = int(len(holes))
    stats["n_filled"] = int(holes.sum()) if len(holes) else 0
    stats["longest"] = int(holes.max()) if len(holes) else 0
    stats["n_long"] = int(np.count_nonzero(holes > max_fill))
    if stats["n_gaps"] == 0:
        return (b,
                {k: np.asarray(v, dtype=np.float64) for k, v in series.items()},
                {k: np.asarray(v, dtype=np.float64) for k, v in masks.items()},
                np.zeros(len(b), dtype=np.float64), stats)

    full = np.arange(b[0], b[-1] + 1, dtype=np.int64)
    xf, xb = full.astype(np.float64), b.astype(np.float64)
    out = {k: np.interp(xf, xb, np.asarray(v, dtype=np.float64))
           for k, v in series.items()}
    out_m = {k: (np.interp(xf, xb, np.asarray(v, dtype=np.float64)) > 0.0
                 ).astype(np.float64) for k, v in masks.items()}
    gap = np.ones(len(full), dtype=np.float64)
    gap[b - b[0]] = 0.0
    stats["span"] = int(len(full))
    return full, out, out_m, gap, stats


def bucket_means(t_ms, acc, braking, ovf, bucket_ms):
    """Bucket means of every series, and 0/1 flags per bucket ->
    (bucket indices, {key: means}, brake flags, ovf flags)."""
    t_list = t_ms.tolist()
    cols = {k: acc[k].tolist() for k in KEYS}
    brk, ofl = braking.tolist(), ovf.tolist()
    bidx, means = [], {k: [] for k in KEYS}
    brake_b, ovf_b = [], []
    cur, start = None, 0
    for i in range(len(t_list) + 1):
        bucket = t_list[i] // bucket_ms if i < len(t_list) else None
        if bucket == cur:
            continue
        if i > start:
            n = i - start
            for k in KEYS:
                s = 0.0
                for v in cols[k][start:i]:
                    s += v
                means[k].append(s / n)
            brake_b.append(1 if any(brk[start:i]) else 0)
            ovf_b.append(1 if any(ofl[start:i]) else 0)
            bidx.append(cur)
        cur, start = bucket, i
    return bidx, means, brake_b, ovf_b


def decimate(vacc, gps, dec_st, row_dt_ms=None):
    """VerticalAccel (+ GpsVup or None) -> AnalysisSeries at dec_st.fs.
    row_dt_ms is the logged row period from cfg.csv (None = derive from the rows)."""
    bucket_ms, fs = dec_st.bucket_ms, dec_st.fs
    if row_dt_ms and bucket_ms < row_dt_ms:
        sys.exit(f"--decimate-hz {fs:g} ({bucket_ms} ms) is higher than the logging "
                 f"rate {1000.0 / row_dt_ms:g} Hz ({row_dt_ms:g} ms) - "
                 "there is no data to decimate from")

    bidx, means, brake_b, ovf_b = bucket_means(vacc.t_ms, vacc.acc, vacc.braking,
                                               vacc.ovf, bucket_ms)
    # A uniform time axis.
    full_idx, ser, msk, gap, gapstats = fill_bucket_gaps(
        bidx, means, {"brake": brake_b, "ovf": ovf_b})

    # FIR mode: replace the bucket means with FIR decimation on the same axis.
    firstats = None
    if dec_st.decimate == "fir":
        fser, fgap, firstats = fir.fir_decimate_series(
            vacc.t_ms.astype(np.float64), {k: vacc.acc[k] for k in KEYS}, full_idx,
            bucket_ms, fs, ntap=dec_st.fir_ntap, cutoff=dec_st.fir_cutoff, unwrap=(),
            compensate=dec_st.fir_compensate, raw_dt_ms=row_dt_ms)
        ser.update(fser)
        gap = np.maximum(gap, fgap)            # add the FIR edges and gap smearing

    # GPS vUp at the bucket centres.
    vg = None
    if gps is not None and len(full_idx):
        bt = (np.asarray(full_idx, dtype=np.float64) + 0.5) * bucket_ms / 1000.0
        vg = np.interp(bt, gps.t_s, gps.vup - float(np.mean(gps.vup)))

    return AnalysisSeries(
        fs=fs, bucket_ms=bucket_ms, idx=np.asarray(full_idx), acc=ser, gps_vup=vg,
        brake=msk["brake"], ovf=msk["ovf"], gap=gap, gapstats=gapstats,
        firstats=firstats, mode=dec_st.decimate, n_rows=len(vacc.t_ms),
        n_brake=int(np.count_nonzero(vacc.braking)),
        n_ovf=int(np.count_nonzero(vacc.ovf)))
