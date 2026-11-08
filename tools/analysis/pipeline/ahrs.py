"""Step 4, orientation (AHRS): Session rows -> vertical linear accel per method.

Each method in METHODS gives one vertical-accel series at the row rate. The AHRS
filters are replayed here on the logged rows; where the raw stream has already
been through a filter (vacc_raw in imu.csv, or RawSeries from raw_stream.py), that
value replaces the replay. SFLP is the chip's own fusion (az_ned_sflp).

    vacc = vertical_accel_series(session, raw, st.ahrs, st.source)
"""

import math
import sys
from dataclasses import dataclass

import numpy as np

import _paths                         # noqa: F401
from rotation import world_z
from madgwick import Madgwick
from kalman import Kalman
from kalman_nxp import KalmanNxp

GRAVITY = 9.80665            # kGravity (m/s²)
MG2MS2 = GRAVITY / 1000.0    # mg -> m/s²
MDPS2RADS = 1.0e-3 * math.pi / 180.0  # mdps -> rad/s


@dataclass(frozen=True)
class Method:
    """One vertical-accel method. name: file/CSV keys (Hs_<name>); key: internal key
    (dict keys in every step); label: figure legend; short: table header.
    make(beta, row_fs) -> an AHRS filter (init_from_accel/update/.q), or None for
    the chip's own SFLP (the az_ned_sflp column)."""
    name: str
    key: str
    label: str
    short: str
    make: object = None


# The methods in table order. Add a filter here and every table, file and figure
# follows.
METHODS = (
    Method("madgwick", "m", "Madgwick", "Madgwick",
           lambda beta, fs: Madgwick(beta=beta)),
    Method("sflp", "s", "SFLP", "SFLP"),
    Method("kalman", "k", "Kalman", "Kalman", lambda beta, fs: Kalman()),
    Method("nxp", "n", "NXP", "NXP", lambda beta, fs: KalmanNxp(fs=fs)),
)
# The methods run here (all but the on-chip SFLP).
AHRS_METHODS = tuple(m for m in METHODS if m.make is not None)
KEYS = tuple(m.key for m in METHODS)
# GPS shares this method's taper and f_c.
REF_KEY = METHODS[0].key


@dataclass
class VerticalAccel:
    """Step 4 output: vertical linear accel per method at the row rate."""
    t_ms: np.ndarray             # row times (win_start_ms)
    acc: dict                    # {key: [m/s²] per row}, non-finite -> 0
    braking: np.ndarray          # 0/1 per row
    ovf: np.ndarray              # 0/1 per row
    source: str                  # "raw" (vacc_raw used) or "csv" (replay only)
    source_method: str           # the method vacc_raw went to, or None
    source_col: str              # the imu.csv column it came from, or None
    fixed_fs: dict               # {name: Hz} for filters with a fixed time step


def vertical_accel(q, ax, ay, az):
    """Vertical linear accel (m/s²): rotation.world_z minus g."""
    return world_z(q, ax, ay, az) - GRAVITY


def ahrs_from_cfg(orientation_name):
    """cfg.csv orientation_name ("Madgwick", "Kalman", "SFLP") -> "kalman" or
    "madgwick". Also used by raw_to_imu_csv."""
    n = (orientation_name or "").strip().lower()
    return "kalman" if n.startswith("kalman") else "madgwick"


def row_period_ms(session):
    """The row period [ms] for filters with a fixed time step: cfg.csv window_ms,
    else the median row spacing. Warns when the two disagree."""
    dts = np.diff(session.t_ms.astype(np.float64))
    dts = dts[dts > 0.0]
    if not len(dts):
        raise ValueError("cannot derive the row rate for the filters: "
                         "too few rows with increasing time")
    obs_dt_ms = float(np.median(dts))
    cfg_dt = session.row_dt_ms
    if cfg_dt and abs(cfg_dt - obs_dt_ms) > 0.05 * obs_dt_ms:
        print(f"  WARNING: cfg.csv states a row period of {cfg_dt:g} ms, but the "
              f"rows lie {obs_dt_ms:g} ms apart - filters with a fixed time "
              f"step will be wrong if cfg.csv is not right")
    return cfg_dt or obs_dt_ms


def vertical_accel_series(session, raw, ahrs_st, source_st):
    """Session (+ RawSeries from raw_stream.py) -> VerticalAccel.

    Every AHRS filter is replayed over all rows; vacc_raw (when source_st allows)
    replaces the capture's own method, and raw.series the others."""
    row_dt_ms = row_period_ms(session)
    filters = {m.key: m.make(ahrs_st.beta, 1000.0 / row_dt_ms) for m in AHRS_METHODS}
    onchip = tuple(m.key for m in METHODS if m.make is None)

    vacc_source = source_st.vacc_source
    has_raw = session.vacc_raw is not None and len(session.t_ms) > 0
    if vacc_source == "raw" and not has_raw:
        sys.exit("--vacc-source raw: imu.csv has neither vacc_fir nor vacc. That "
                 "column exists only in sessions logged with a raw log or "
                 "WaveLogMode::Both - use --vacc-source csv (or auto).")
    use_raw = has_raw and vacc_source in ("auto", "raw")
    # vacc_raw goes in the row of the filter the capture ran.
    raw_ahrs = session.orientation_name
    raw_slot = {m.name: m.key for m in AHRS_METHODS}.get(str(raw_ahrs).lower())
    if use_raw and raw_slot is None:
        print(f"  WARNING: cfg.csv states AHRS '{raw_ahrs}', which has no row of "
              f"its own here - vacc_fir is put in the {AHRS_METHODS[0].label} row.")
        raw_slot = AHRS_METHODS[0].key
    if not use_raw:
        raw_slot = None

    n = len(session.t_ms)
    out = {k: np.empty(n, dtype=np.float64) for k in KEYS}
    t_list = session.t_ms.tolist()
    acc = session.acc_mg.tolist()
    gyro = session.gyro_mdps.tolist()
    azn = session.az_ned_sflp.tolist()
    vacc_raw = session.vacc_raw.tolist() if raw_slot is not None else None
    raw_series = raw.series if raw is not None else {}

    have_q = False
    prev_t = 0
    for i in range(n):
        t = t_list[i]
        # The filters take m/s² (the Kalman R compares |a| with g).
        axm = acc[i][0] * MG2MS2
        aym = acc[i][1] * MG2MS2
        azm = acc[i][2] * MG2MS2
        gx = gyro[i][0] * MDPS2RADS
        gy = gyro[i][1] * MDPS2RADS
        gz = gyro[i][2] * MDPS2RADS
        dt = (t - prev_t) / 1000.0 if have_q else 0.0

        if not have_q:
            for flt in filters.values():
                flt.init_from_accel(axm, aym, azm)
            have_q = True
        elif dt > 0.0:
            for flt in filters.values():
                flt.update(gx, gy, gz, axm, aym, azm, dt)
        prev_t = t

        v = {k: vertical_accel(flt.q, axm, aym, azm) for k, flt in filters.items()}
        for k in onchip:
            v[k] = azn[i] * MG2MS2
        # Raw-stream values replace the replay: the capture's filter from vacc_raw,
        # the others from raw_series (which wins).
        if raw_slot is not None:
            v[raw_slot] = vacc_raw[i]
        # NaN = no matching row; the replay stands.
        for k, arr in raw_series.items():
            if math.isfinite(arr[i]):
                v[k] = arr[i]
        for k in KEYS:
            out[k][i] = v[k] if math.isfinite(v[k]) else 0.0

    return VerticalAccel(
        t_ms=session.t_ms, acc=out, braking=session.braking, ovf=session.ovf,
        source="raw" if use_raw else "csv",
        source_method={m.key: m.name for m in METHODS}.get(raw_slot),
        source_col=session.vacc_col if use_raw else None,
        fixed_fs={m.name: filters[m.key].fs for m in AHRS_METHODS
                  if hasattr(filters[m.key], "fs")})
