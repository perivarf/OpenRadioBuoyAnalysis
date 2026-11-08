#!/usr/bin/env python3
"""Which log a session is analysed from, and a directory postprocess can read.

The raw log (<stamp>_raw.bin) wins over the device's <stamp>_imu.csv when both
exist. ensure_imu(sess_dir, config) returns a directory with an imu.csv in it:

  with a config  <measurement>/<config>/<stamp>/: imu.csv reconstructed from the
                 raw log, or a symlink to the device's; sidecars copied.
  without        <session>/<stamp>_raw/ for a raw log, else the session itself.

The session directory itself is never written. A reconstructed imu.csv is reused
until deleted, or until the averaging asked for or the capture's FIR taps
stage 1 (the FIR with fir_ntap from cfg.csv, or $FIR_STAGE1; kept in
<stamp>_stage1.txt) changes.

RAW_AVERAGE in the environment ("<session folder>=<f> ...", set by plot.sh) makes
the reconstruction average f consecutive raw samples (rate / f) for those folders;
the f used is kept in <stamp>_raw_average.txt beside the imu.csv.

    from session_source import ensure_imu
    d = ensure_imu(sess_dir, config)
"""

import csv
import os
import shutil
import sys
from glob import glob

import _paths                                               # noqa: E402,F401
import raw_to_imu_csv as rti                                # noqa: E402
from configs import config_dir                              # noqa: E402

# Copied beside the imu.csv. _cfg.csv gives the row period and the capture's filter;
# mapplot takes the heading and description from _notat.txt.
SIDECAR_FILES = ["_cfg.csv", "_ses.csv", "_gps.csv", "_notat.txt"]


# _gps.csv from the 2026-08-13/14 firmware is in integer units; the copy is scaled
# to the decoded format the tools read. old name -> (new name, scale)
GPS_SCALE = {
    "lat_e7": ("lat", 1e-7),
    "lng_e7": ("lon", 1e-7),
    "gspeed_mms": ("gspeed", 1e-3),
    "vup_mms": ("vUp", 1e-3),
    "hacc_mm": ("hAccuracy", 1e-3),
    "head_e5": ("head", 1e-5),
}


def _copy(src, dst):
    """Copy src to dst; a _gps.csv in integer units is scaled on the way."""
    if not src.endswith("_gps.csv"):
        shutil.copy2(src, dst)
        return
    with open(src, newline="", encoding="latin-1") as f:
        rows = list(csv.reader(f))
    head = rows[0] if rows else []
    if not any(c in GPS_SCALE for c in head):
        shutil.copy2(src, dst)
        return
    # The 2026-08-14 files end in raw-log bytes: keep the rows up to the first bad one.
    good = 1
    while good < len(rows) and len(rows[good]) == len(head) and all(
            v.lstrip("-").isdigit() for v in rows[good]):
        good += 1
    if good < len(rows):
        print(f"  {os.path.basename(src)}: kept {good - 1} of {len(rows) - 1} rows "
              f"(the rest is not GPS data)", flush=True)
    rows = rows[:good]
    scale = [GPS_SCALE.get(c, (c, None))[1] for c in head]
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([GPS_SCALE.get(c, (c, None))[0] for c in head])
        for row in rows[1:]:
            w.writerow([f"{float(v) * k:.7f}".rstrip("0").rstrip(".")
                        if k is not None and v != "" else v
                        for v, k in zip(row, scale)])
    shutil.copystat(src, dst)


def _refresh(src, dst):
    """Copy src over dst when dst is missing or older."""
    if os.path.isfile(src) and (not os.path.isfile(dst)
                                or os.path.getmtime(src) > os.path.getmtime(dst)):
        _copy(src, dst)


def _link(src, dst):
    """Make dst a relative symlink to src, replacing whatever is there."""
    target = os.path.relpath(src, os.path.dirname(dst))
    if os.path.islink(dst) and os.readlink(dst) == target:
        return
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(target, dst)


def _raw_average(sess_dir):
    """f for this session folder from $RAW_AVERAGE, 1 when it is not named there."""
    name = os.path.basename(os.path.normpath(sess_dir))
    for item in os.environ.get("RAW_AVERAGE", "").split():
        folder, _, f = item.partition("=")
        if folder == name:
            return int(f)
    return 1


def _read_marker(path, default):
    if os.path.isfile(path):
        with open(path) as fh:
            return int(fh.read().strip() or default)
    return default


def _reconstruct(binpath, imu_out, sess_dir, where):
    """Write imu_out from the raw log unless it is there with the f asked for and
    the stage 1 asked for (the capture's FIR taps, or $FIR_STAGE1)."""
    f = _raw_average(sess_dir)
    marker = imu_out.replace("_imu.csv", "_raw_average.txt")
    done = _read_marker(marker, 1)
    # Reconstructions from before the stage was recorded used the FIR with fir.NTAP.
    stage1 = rti.read_params(binpath).stage1_desc()
    stage1_marker = imu_out.replace("_imu.csv", "_stage1.txt")
    made = f"fir {rti.fir.NTAP}"
    if os.path.isfile(stage1_marker):
        with open(stage1_marker) as fh:
            made = fh.read().strip()
    if os.path.isfile(imu_out) and done == f and made == stage1:
        return
    stamp = os.path.basename(binpath)[:-len("_raw.bin")]
    avg = f" averaging {f} samples" if f > 1 else ""
    print(f"  reconstructing {stamp}_imu.csv from the raw log{where}{avg} ...",
          flush=True)
    rti.write_imu_csv(binpath, imu_out, verbose=False, average=f)
    # The other filters' runs on the raw stream were made with the old f.
    for old in glob(imu_out.replace("_imu.csv", "_imu_raw_*.csv")):
        os.remove(old)
    if f > 1:
        with open(marker, "w") as fh:
            fh.write(f"{f}\n")
    elif os.path.isfile(marker):
        os.remove(marker)
    with open(stage1_marker, "w") as fh:
        fh.write(f"{stage1}\n")


def _config_workdir(sess_dir, config):
    """<measurement>/<config>/<stamp>/ with the inputs in place."""
    sess_dir = os.path.abspath(sess_dir)
    out = os.path.join(config_dir(os.path.dirname(sess_dir), config),
                       os.path.basename(sess_dir))
    os.makedirs(out, exist_ok=True)

    raw = sorted(glob(os.path.join(sess_dir, "*_raw.bin")))
    if raw:
        stamp = os.path.basename(raw[0])[:-len("_raw.bin")]
    else:
        imu = sorted(glob(os.path.join(sess_dir, "*_imu.csv")))
        if not imu:
            sys.exit(f"found neither *_raw.bin nor *_imu.csv in {sess_dir}")
        stamp = os.path.basename(imu[0])[:-len("_imu.csv")]
        _link(imu[0], os.path.join(out, f"{stamp}_imu.csv"))

    # Sidecars follow the originals (the config directory is output only).
    for suf in SIDECAR_FILES:
        _refresh(os.path.join(sess_dir, stamp + suf), os.path.join(out, stamp + suf))

    imu_out = os.path.join(out, f"{stamp}_imu.csv")
    if raw and os.path.islink(imu_out):
        # A leftover link must go, or the reconstruction would write through it.
        os.remove(imu_out)
    if raw:
        _reconstruct(raw[0], imu_out, sess_dir, f" into {config}/")
    return out


def workdir(sess_dir, config=""):
    """(directory, stamp) that ensure_imu would return for this session, without
    creating or reconstructing anything - for redrawing figures from data/."""
    raw = sorted(glob(os.path.join(sess_dir, "*_raw.bin")))
    imu = sorted(glob(os.path.join(sess_dir, "*_imu.csv")))
    if raw:
        stamp = os.path.basename(raw[0])[:-len("_raw.bin")]
    elif imu:
        stamp = os.path.basename(imu[0])[:-len("_imu.csv")]
    else:
        sys.exit(f"found neither *_raw.bin nor *_imu.csv in {sess_dir}")
    if config:
        sess_dir = os.path.abspath(sess_dir)
        return (os.path.join(os.path.dirname(sess_dir), config,
                             os.path.basename(sess_dir)), stamp)
    return (os.path.join(sess_dir, f"{stamp}_raw") if raw else sess_dir), stamp


def ensure_imu(sess_dir, config=""):
    """A directory postprocess can analyse for this session (see the module docstring).

    _gps.csv from the 2026-08-13/14 firmware (integer units) is scaled on the copy."""
    if config:
        return _config_workdir(sess_dir, config)
    raw = sorted(glob(os.path.join(sess_dir, "*_raw.bin")))
    if not raw:
        if glob(os.path.join(sess_dir, "*_imu.csv")):
            return sess_dir
        sys.exit(f"found neither *_raw.bin nor *_imu.csv in {sess_dir}")

    binpath = raw[0]
    stamp = os.path.basename(binpath)[:-len("_raw.bin")]
    out = os.path.join(sess_dir, f"{stamp}_raw")
    imu_out = os.path.join(out, f"{stamp}_imu.csv")
    os.makedirs(out, exist_ok=True)
    for suf in SIDECAR_FILES:
        src, dst = os.path.join(sess_dir, stamp + suf), os.path.join(out, stamp + suf)
        # Never overwrite: _notat.txt here may be hand-edited.
        if os.path.isfile(src) and not os.path.isfile(dst):
            _copy(src, dst)

    _reconstruct(binpath, imu_out, sess_dir, "")
    return out
