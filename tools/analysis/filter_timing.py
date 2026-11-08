#!/usr/bin/env python3
"""CPU time per orientation filter (Madgwick, Kalman, Kalman NXP) on one session.

Each filter runs alone over the same input as in the pipeline: every raw sample
from <stamp>_raw.bin when there is one (the measured rate), else the rows of
<stamp>_imu.csv (the row rate). Only init/update is timed (time.perf_counter);
reading the data and the vertical accel are left out.

    rows = filter_timing.time_session(sess_dir)
    filter_timing.write_csv(rows, path)
"""

import contextlib
import csv
import io
import os
import platform
import time
from glob import glob

import numpy as np

import _paths                                               # noqa: F401
import raw_to_imu_csv as rti
import rawlog
from pipeline.ahrs import AHRS_METHODS
from pipeline.session_input import read_session
from pipeline.settings import MADGWICK_BETA

# (name in pipeline.ahrs / raw_to_imu_csv, label)
FILTERS = (("madgwick", "Madgwick"), ("kalman", "Kalman"), ("nxp", "Kalman NXP"))
FIELDS = ["session", "filter", "input", "rate_hz", "n_samples", "data_s", "cpu_s",
          "us_per_sample", "x_realtime", "python", "machine"]
MG2MS2 = 9.80665e-3
MDPS2RADS = np.pi / 180.0e3


def _run(filt, t_s, acc_mg, gyr_mdps):
    """init + one update per sample -> CPU seconds (wall clock, perf_counter)."""
    a = (np.asarray(acc_mg) * MG2MS2).tolist()
    g = (np.asarray(gyr_mdps) * MDPS2RADS).tolist()
    t = np.asarray(t_s, dtype=float).tolist()
    start = time.perf_counter()
    filt.init_from_accel(*a[0])
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            filt.update(*g[i], *a[i], dt)
    return time.perf_counter() - start


def _raw_input(binpath):
    """(t_s, acc_mg, gyr_mdps, rate_hz, make) from the raw log; make(name) builds
    the filter as raw_to_imu_csv does."""
    with contextlib.redirect_stdout(io.StringIO()):
        cap = rawlog.read(binpath)
    gyr, _ = rti.align(cap)
    t_s = cap.acc_t
    rate = (len(t_s) - 1) / (t_s[-1] - t_s[0])
    params = rti.read_params(binpath)
    return t_s, cap.acc, gyr, rate, lambda name: rti.make_ahrs(name, params, rate)


def _csv_input(sess_dir):
    """(t_s, acc_mg, gyr_mdps, rate_hz, make) from imu.csv; make(name) builds the
    filter as the pipeline's replay does."""
    with contextlib.redirect_stdout(io.StringIO()):
        s = read_session(sess_dir)
    t_s = s.t_ms.astype(float) / 1000.0
    d = np.diff(t_s)
    rate = 1.0 / float(np.median(d[d > 0]))
    makers = {m.name: m.make for m in AHRS_METHODS}
    return t_s, s.acc_mg, s.gyro_mdps, rate, lambda name: makers[name](MADGWICK_BETA,
                                                                       rate)


def time_session(sess_dir):
    """One row per filter (dicts with FIELDS) for the session in sess_dir."""
    raw = sorted(glob(os.path.join(sess_dir, "*_raw.bin")))
    if raw:
        t_s, acc, gyr, rate, make = _raw_input(raw[0])
        source = "raw.bin"
    else:
        t_s, acc, gyr, rate, make = _csv_input(sess_dir)
        source = "imu.csv"
    data_s = float(t_s[-1] - t_s[0])
    rows = []
    for name, label in FILTERS:
        cpu = _run(make(name), t_s, acc, gyr)
        rows.append(dict(session=os.path.basename(os.path.normpath(sess_dir)),
                         filter=label, input=source, rate_hz=f"{rate:.1f}",
                         n_samples=len(t_s), data_s=f"{data_s:.1f}",
                         cpu_s=f"{cpu:.2f}",
                         us_per_sample=f"{cpu / len(t_s) * 1e6:.2f}",
                         x_realtime=f"{data_s / cpu:.0f}",
                         python=platform.python_version(),
                         machine=platform.processor() or platform.machine()))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
