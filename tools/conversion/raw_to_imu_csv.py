"""raw.bin -> CSV: the raw samples, or a reconstructed imu.csv.

  samples   one row per accel sample at kImuOdrHz, gyro and SFLP paired to the
            nearest sample
  imu       the device format (kImuCsvHeader columns): FIR decimation to the row
            rate, AHRS on the raw stream, vertical accel - what the firmware computes
            on board. postprocess reads it like the device's imu.csv.

convert(..., compare_device=True) compares the reconstruction with the device's own
imu.csv from the same capture (float32 vs float64: ~1e-6).

--mode imu warns that the bucket does not divide the row period when the raw -> row
ratio is not whole (480 Hz -> 100 Hz is 4.8, kFirS1DecimX10): expected. The row
period need not be whole ms either (80 Hz is 12.5 ms).

Run with script_raw_to_imu_csv.py.
"""

import contextlib
import io
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

import _paths                                   # noqa: E402,F401
from pipeline import ahrs as pipe_ahrs           # noqa: E402
from pipeline.session_input import read_kv       # noqa: E402
import rawlog                                   # noqa: E402
import fir                                      # noqa: E402
from madgwick import Madgwick                   # noqa: E402
from kalman import Kalman                       # noqa: E402
from kalman_nxp import KalmanNxp                # noqa: E402

# Filters that can run on the raw stream, in postprocess's method order.
AHRS_NAMES = ("madgwick", "kalman", "nxp")

# Units from the pipeline's AHRS step, so vacc is defined in one place.
GRAVITY = pipe_ahrs.GRAVITY
MG2MS2 = pipe_ahrs.MG2MS2
MDPS2RADS = pipe_ahrs.MDPS2RADS

# Fallbacks used only when the capture's cfg.csv lacks the key.
ROW_ODR_HZ = 100          # kRowOdrHz
MADGWICK_BETA = 0.05      # kMadgwickBeta
BRAKE_G_THRESHOLD = 0.5   # kBrakeGThreshold
BRAKE_MIN_SAMPLES = 3     # kBrakeMinSamples (derived on board from kBrakeMinMs * ODR)


class Params:
    """Reconstruction parameters from the capture's cfg.csv, with the fallbacks
    above. describe() lists which came from the file."""

    def __init__(self, cfg):
        self.from_cfg, self.missing = [], []

        def get(key, default, cast=float):
            try:
                v = cast(cfg[key])
                self.from_cfg.append(f"{key}={v:g}")
                return v
            except (KeyError, ValueError):
                self.missing.append(key)
                return default

        self.row_odr_hz = get("output_rate_hz", ROW_ODR_HZ, lambda s: int(float(s)))
        self.madgwick_beta = get("madgwick_beta", MADGWICK_BETA)
        self.brake_g = get("brake_g_thresh", BRAKE_G_THRESHOLD)
        # The value derived on board (brake_min_ms * ODR).
        self.brake_min_samples = get("brake_min_samples", BRAKE_MIN_SAMPLES,
                                     lambda s: int(float(s)))
        self.gravity = get("gravity", pipe_ahrs.GRAVITY)
        # The filter the capture ran.
        self.orientation = cfg.get("orientation_name", "")
        # Stage 1 (raw -> rows): the FIR with the capture's taps (kFirNtap).
        # $FIR_STAGE1 (set by plot.sh) tries another setting: "average" (block mean
        # per row window) or a tap count.
        self.stage1 = "fir"
        self.fir_ntap = get("fir_ntap", fir.NTAP, lambda s: int(float(s)))
        override = os.environ.get("FIR_STAGE1", "").strip().lower()
        if override:
            self.set_stage1(override)
            self.from_cfg.append(f"stage1={self.stage1_desc()} ($FIR_STAGE1)")

    def set_stage1(self, spec):
        """"average" -> block mean, a number -> the FIR with that many taps."""
        if spec in ("average", "mean"):
            self.stage1 = "mean"
        else:
            self.stage1, self.fir_ntap = "fir", int(spec)

    def stage1_desc(self):
        """"mean" or "fir <taps>" - also the reconstruction's cache marker."""
        return "mean" if self.stage1 == "mean" else f"fir {self.fir_ntap}"

    def ahrs_name(self, override=None):
        """override, or the filter the capture ran (pipeline.ahrs.ahrs_from_cfg)."""
        if override:
            return override
        return pipe_ahrs.ahrs_from_cfg(self.orientation)

    def describe(self):
        s = "  cfg: " + (", ".join(self.from_cfg) if self.from_cfg else "none")
        if self.orientation:
            s += f", orientation_name={self.orientation}"
        if self.missing:
            s += ("\n  MISSING from cfg.csv, using built-in defaults: "
                  + ", ".join(self.missing))
        return s


def read_params(binpath):
    return Params(read_kv(binpath.replace("_raw.bin", "_cfg.csv")))


def check_integrity(cap, allow_damaged=False):
    """Stop if the raw log lost blocks (WRITE_FAIL), unless allow_damaged."""
    if not cap.n_write_fail:
        return
    msg = (f"the raw log lost {cap.n_write_fail} blocks and is desynchronised from "
           f"accel sample {cap.first_write_fail_n} of {len(cap.acc)}")
    if not allow_damaged:
        sys.exit(f"ABORTING: {msg}.\n"
                 f"  See also raw_write_failures in ana.csv. Use --allow-damaged "
                 f"if you want to look at the intact part.")
    print(f"  [!] --allow-damaged: {msg}")

IMU_COLUMNS = ("win_start_ms,n,ax_mg,ay_mg,az_mg,ax_ned_sflp,ay_ned_sflp,az_ned_sflp,"
               "gx_mdps,gy_mdps,gz_mdps,qw_sflp,qx_sflp,qy_sflp,qz_sflp,braking,"
               "qw,qx,qy,qz,vacc,vacc_sflp,sflp_nan,fifo_ovf,vacc_fir,"
               "vacc_sflp_fir").split(",")


def resolve(path):
    """Accept a .bin, or a session directory containing one."""
    if os.path.isdir(path):
        cand = sorted(glob(os.path.join(path, "*_raw.bin")))
        if not cand:
            sys.exit(f"found no *_raw.bin in {path}")
        return cand[0]
    if not os.path.isfile(path):
        sys.exit(f"did not find {path}")
    return path


def align(cap):
    """Gyro and SFLP onto the accel time axis, holding the last value as the
    firmware does."""
    n = len(cap.acc_t)
    gyr = np.zeros((n, 3))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    if len(cap.gyr_t):
        j = np.searchsorted(cap.gyr_t, cap.acc_t, side="right") - 1
        gyr = cap.gyr[np.clip(j, 0, len(cap.gyr) - 1)]
        gyr[j < 0] = 0.0
    if len(cap.quat_t):
        j = np.searchsorted(cap.quat_t, cap.acc_t, side="right") - 1
        quat = cap.quat[np.clip(j, 0, len(cap.quat) - 1)]
        quat[j < 0] = np.array([1.0, 0.0, 0.0, 0.0])
    return gyr, quat


def samples_frame(cap):
    gyr, quat = align(cap)
    return pd.DataFrame({
        "t_s": cap.acc_t,
        "ax_mg": cap.acc[:, 0], "ay_mg": cap.acc[:, 1], "az_mg": cap.acc[:, 2],
        "gx_mdps": gyr[:, 0], "gy_mdps": gyr[:, 1], "gz_mdps": gyr[:, 2],
        "qw_sflp": quat[:, 0], "qx_sflp": quat[:, 1],
        "qy_sflp": quat[:, 2], "qz_sflp": quat[:, 3],
    })


def world_ned(quat, acc_mg):
    """R(q)·a for the whole series, gravity subtracted from Z. Vectorised version of
    the inlined rotation in imu_sampler.cpp."""
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    ax, ay, az = acc_mg[:, 0], acc_mg[:, 1], acc_mg[:, 2]
    wx = ((1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay
          + 2 * (qx * qz + qw * qy) * az)
    wy = (2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay
          + 2 * (qy * qz - qw * qx) * az)
    wz = (2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay
          + (1 - 2 * (qx * qx + qy * qy)) * az)
    return wx, wy, wz - 1000.0


def make_ahrs(name, params, odr_hz):
    """The filter to run on the raw stream, with the capture's constants. NXP is
    constructed with odr_hz (it has a fixed time step): the measured rate, not
    the nominal ODR."""
    if name == "madgwick":
        return Madgwick(beta=params.madgwick_beta)
    if name == "kalman":
        return Kalman()
    if name == "nxp":
        return KalmanNxp(fs=float(odr_hz))
    sys.exit(f"unknown AHRS '{name}' - know {', '.join(AHRS_NAMES)}")


def run_ahrs(t_s, acc_mg, gyr_mdps, filt):
    """AHRS on the RAW STREAM - one update per sample, as in imu_sampler.cpp.
    Returns the quaternions and vertical linear accel (m/s²) per sample."""
    n = len(t_s)
    q = np.empty((n, 4))
    vacc = np.empty(n)
    a_si = acc_mg * MG2MS2
    g_si = gyr_mdps * MDPS2RADS
    for i in range(n):
        if i == 0:
            filt.init_from_accel(*a_si[0])
        else:
            dt = t_s[i] - t_s[i - 1]
            if dt > 0:
                filt.update(*g_si[i], *a_si[i], dt)
        q[i] = filt.q
        vacc[i] = pipe_ahrs.vertical_accel(filt.q, *a_si[i])
    return q, vacc


def braking_flags(wx, wy, wz, brake_g, min_samples):
    """The braking flag: |linear a| above threshold for min_samples consecutive
    samples."""
    thr2 = (brake_g * 1000.0) ** 2
    over = (wx * wx + wy * wy + wz * wz) > thr2
    need = max(1, min_samples)
    run = 0
    out = np.zeros(len(over), dtype=bool)
    for i, o in enumerate(over):
        run = run + 1 if o else 0
        out[i] = run >= need
    return out


def average_samples(t_s, acc, gyr, quat, f):
    """Mean of every f consecutive samples (time, accel, gyro): rate / f. SFLP keeps
    the last quaternion of each block, as the hold on board does. The tail that
    does not fill a block is dropped."""
    n = len(t_s) // f * f

    def mean(x):
        x = np.asarray(x)[:n]
        return x.reshape(n // f, f, *x.shape[1:]).mean(axis=1)

    return mean(t_s), mean(acc), mean(gyr), np.asarray(quat)[f - 1:n:f]


def block_mean_series(t_ms, series, win, row_index):
    """Stage 1 as a plain mean of the raw samples in each row window, instead of the
    FIR -> ({key: per row}, gap per row: 1 where a window has no sample)."""
    idx = row_index(t_ms) - win[0]
    ok = (idx >= 0) & (idx < len(win))
    cnt = np.bincount(idx[ok], minlength=len(win)).astype(np.float64)
    dec = {}
    for k, x in series.items():
        s = np.bincount(idx[ok], weights=np.asarray(x, np.float64)[ok], minlength=len(win))
        with np.errstate(invalid="ignore", divide="ignore"):
            dec[k] = np.where(cnt > 0, s / cnt, np.nan)
    return dec, (cnt == 0).astype(np.float64)


def imu_frame(cap, params, ahrs=None, average=1, stage1="fir"):
    """Reconstruct the device format from the raw samples, with the capture's own
    constants. average > 1 first takes the mean of that many raw samples
    (average_samples). stage1 = "fir" (as on board) or "mean" (block mean per row
    window) for the raw -> row decimation."""
    gyr, quat = align(cap)
    acc = cap.acc
    t_s = cap.acc_t
    odr = float(cap.imu_odr_hz)
    if average > 1:
        t_s, acc, gyr, quat = average_samples(t_s, acc, gyr, quat, average)
        odr /= average
    row_odr_hz = params.row_odr_hz
    # The row grid as on board (rowIndexAt/rowStartMs in analysis_config.h): defined by
    # the RATE, so 80 Hz is 12.5 ms - not 12 ms accumulated into an 83.3 Hz grid.
    row_ms = 1000.0 / row_odr_hz

    def row_index(tms):
        """Window index of a time [ms], from the whole-ms time as on board."""
        return (np.floor(np.asarray(tms)).astype(np.int64) * row_odr_hz) // 1000

    wx, wy, wz = world_ned(quat, acc)
    name = params.ahrs_name(ahrs)
    # The sensor's clock is off by a few percent (460.8 Hz at a nominal 480 on
    # 2026-08-14), so the fixed-step NXP gets the mean rate of the time axis.
    fs = (len(t_s) - 1) / (t_s[-1] - t_s[0]) if len(t_s) > 1 else odr
    q_ahrs, vacc = run_ahrs(t_s, acc, gyr, make_ahrs(name, params, fs))
    brake = braking_flags(wx, wy, wz, params.brake_g, params.brake_min_samples)

    # FIR stage 1 on all ten series onto the row grid, as FirRowBank on board.
    t_ms = t_s * 1000.0
    series = {"ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
              "nx": wx, "ny": wy, "nz": wz,
              "gx": gyr[:, 0], "gy": gyr[:, 1], "gz": gyr[:, 2],
              "vacc": vacc}
    win = np.arange(row_index(t_ms[0]), row_index(t_ms[-1]) + 1)
    if stage1 == "mean":
        dec, gap = block_mean_series(t_ms, series, win, row_index)
    else:
        dec, gap, _ = fir.fir_decimate_series(t_ms, series, win, row_ms, row_odr_hz,
                                              ntap=params.fir_ntap,
                                              raw_dt_ms=1000.0 / odr)

    # Unfiltered values: the raw sample nearest the row centre (kFirS1CenterMs).
    centre_t = (win * row_ms + row_ms / 2.0) / 1000.0
    ci = np.clip(np.searchsorted(t_s, centre_t), 0, len(t_s) - 1)

    out = pd.DataFrame({
        "win_start_ms": (win * 1000) // row_odr_hz,     # rounded down, as on board
        "n": np.bincount(np.clip(row_index(t_ms) - win[0], 0, len(win) - 1),
                         minlength=len(win)),
        "ax_mg": dec["ax"], "ay_mg": dec["ay"], "az_mg": dec["az"],
        "ax_ned_sflp": dec["nx"], "ay_ned_sflp": dec["ny"], "az_ned_sflp": dec["nz"],
        "gx_mdps": dec["gx"], "gy_mdps": dec["gy"], "gz_mdps": dec["gz"],
        "qw_sflp": quat[ci, 0], "qx_sflp": quat[ci, 1],
        "qy_sflp": quat[ci, 2], "qz_sflp": quat[ci, 3],
        "braking": brake[ci].astype(int),
        "qw": q_ahrs[ci, 0], "qx": q_ahrs[ci, 1],
        "qy": q_ahrs[ci, 2], "qz": q_ahrs[ci, 3],
        "vacc": vacc[ci],
        "vacc_sflp": wz[ci] * MG2MS2,
        "sflp_nan": 0,
        "fifo_ovf": 0,
        "vacc_fir": dec["vacc"],
        "vacc_sflp_fir": dec["nz"] * MG2MS2,
    })
    # fifo_ovf from the sync records. As on board, the flag goes on the window of
    # the previous drain's last sample (accel_n - 1), not the sync's t_us.
    for s in cap.sync:
        if s.get("fifo_ovf", bool(s["flags"] & rawlog.FLAG_FIFO_OVF)):
            i = max(0, int(s["accel_n"]) - 1) // average   # accel_n == 0: first window
            if i < len(t_ms):
                w = int(row_index(t_ms[i])) - win[0]
                if 0 <= w < len(out):
                    out.loc[w, "fifo_ovf"] = 1
    return out.reindex(columns=IMU_COLUMNS)


def write_imu_csv(binpath, out_file, ahrs=None, allow_damaged=False, verbose=True,
                  average=1, stage1=None, ntap=None, row_odr_hz=None):
    """<stamp>_raw.bin -> reconstructed imu.csv at out_file (written to a
    temporary name, then renamed). Returns the DataFrame. ahrs overrides the
    capture's filter; average > 1 averages that many raw samples first; stage1
    ("fir" | "mean") and ntap override stage 1 from cfg.csv / $FIR_STAGE1;
    row_odr_hz the row rate (e.g. the Welch rate itself, for a single stage)."""
    cap = rawlog.read(binpath)
    if verbose:
        print(rawlog.summary(cap))
    check_integrity(cap, allow_damaged)

    params = read_params(binpath)
    if stage1:
        params.stage1 = stage1
    if ntap:
        params.fir_ntap = ntap
    if row_odr_hz:
        params.row_odr_hz = int(row_odr_hz)
    if verbose:
        print(params.describe())
        print(f"  AHRS: {params.ahrs_name(ahrs)}"
              f"{' (chosen)' if ahrs else ' (from cfg.csv)'}")

    if verbose and average > 1:
        print(f"  averaging {average} raw samples: {cap.imu_odr_hz:g} -> "
              f"{cap.imu_odr_hz / average:g} Hz nominal")
    df = imu_frame(cap, params, ahrs, average, params.stage1)
    tmp = f"{out_file}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False, float_format="%.5f")
    os.replace(tmp, out_file)
    return df


def compare(recon, device_csv):
    """Reconstruction against the device's own imu.csv, column by column."""
    dev = pd.read_csv(device_csv)
    key = "win_start_ms"
    m = recon.merge(dev, on=key, suffixes=("_r", "_d"))
    if m.empty:
        return "  no shared win_start_ms - is this the same recording?"
    lines = [f"  {len(m)} shared rows of {len(recon)} reconstructed / {len(dev)} device",
             f"  {'column':16s} {'max |dev|':>12s} {'rms':>12s} {'device rms':>12s}"]
    for c in IMU_COLUMNS:
        if c == key or f"{c}_r" not in m:
            continue
        a, b = m[f"{c}_r"].to_numpy(float), m[f"{c}_d"].to_numpy(float)
        d = np.abs(a - b)
        lines.append(f"  {c:16s} {d.max():12.3e} {np.sqrt((d**2).mean()):12.3e} "
                     f"{np.sqrt((b**2).mean()):12.3e}")
    return "\n".join(lines)


def write_gps_from_raw(binpath):
    """From format v3 the GPS track is in the raw log and the firmware writes no
    gps.csv. Write <stamp>_gps.csv from it, so the readers of gps.csv find it.
    Older captures (no GPS records) are left alone - their gps.csv is the device's."""
    with open(binpath, "rb") as f:
        if rawlog.read_header(f.read(rawlog.HEADER_BYTES))["version"] < 3:
            return None
    with contextlib.redirect_stdout(io.StringIO()):
        cap = rawlog.read(binpath)
    if not cap.gps:
        return None
    out = binpath.replace("_raw.bin", "_gps.csv")
    n = rawlog.write_gps_csv(cap, out)
    print(f"wrote {out}  ({n} rows, from the raw log's GPS records)")
    return out


def convert(path, mode="samples", ahrs=None, out=None, compare_device=False,
            allow_damaged=False):
    """<stamp>_raw.bin (or its session directory) -> CSV. mode "samples" writes the
    raw samples, "imu" an imu.csv like the device's. Returns the output path."""
    binpath = resolve(path)
    out = out or binpath.replace("_raw.bin", f"_{mode}_python.csv")
    if mode == "imu":
        df = write_imu_csv(binpath, out, ahrs=ahrs, allow_damaged=allow_damaged)
    else:
        cap = rawlog.read(binpath)
        print(rawlog.summary(cap))
        check_integrity(cap, allow_damaged)
        df = samples_frame(cap)
        df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nwrote {out}  ({len(df)} rows, {len(df.columns)} columns)")

    write_gps_from_raw(binpath)

    if compare_device and mode == "imu":
        dev = binpath.replace("_raw.bin", "_imu.csv")
        if not os.path.isfile(dev):
            print(f"  --compare: did not find {dev} (requires WaveLogMode::Both)")
        else:
            print(f"\nagainst {os.path.basename(dev)}:")
            print(compare(df, dev))
    return out
