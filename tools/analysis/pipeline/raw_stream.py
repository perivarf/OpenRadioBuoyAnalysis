"""Step 3, the raw stream: AHRS per method on <stamp>_raw.bin -> RawSeries.

The capture's own filter already ran on the raw stream (vacc_fir in imu.csv). The
other AHRS methods are run on the raw log here, via raw_to_imu_csv, into
<stamp>_imu_raw_<method>.csv (cached), and matched to the rows. Without a raw log
they are replayed on the rows in step 4.

    raw = raw_series(session, st.source)
"""

import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from .ahrs import AHRS_METHODS, METHODS, ahrs_from_cfg
from .session_input import read_csv_stream, truncation_message


@dataclass
class RawSeries:
    """Step 3 output."""
    series: dict                 # {key: vacc [m/s²] on the row axis, NaN = no match}
    sources: dict                # {name: "onchip" | "csv" | "onboard:imu.csv" | "raw:<file>"}
    rates: dict                  # {name: Hz the method ran at} (+ "gps")


def find_raw_log(directory, stamp, override=None):
    """<stamp>_raw.bin, or None ("off" disables the lookup). Looked for in directory,
    one level up (<session>/<stamp>_raw/), and in <measurement>/<stamp>/ for a
    config work directory <measurement>/<config>/<stamp>/."""
    if override == "off":
        return None
    if override:
        if not os.path.isfile(override):
            sys.exit(f"--raw {override}: did not find the file")
        return override
    up = os.path.dirname(directory)
    for candidate in (os.path.join(directory, f"{stamp}_raw.bin"),
                      os.path.join(up, f"{stamp}_raw.bin"),
                      os.path.join(os.path.dirname(up), os.path.basename(directory),
                                   f"{stamp}_raw.bin")):
        if os.path.isfile(candidate):
            return candidate
    return None


def raw_average(directory, stamp):
    """f when imu.csv here was made from f-sample averages of the raw log
    (<stamp>_raw_average.txt, written by session_source), else 1."""
    path = os.path.join(directory, f"{stamp}_raw_average.txt")
    if not os.path.isfile(path):
        return 1
    with open(path) as f:
        return int(f.read().strip() or 1)


def make_raw_csv(raw_bin, out_file, method, average=1):
    """Reconstruct imu.csv with `method` on the raw stream (averaged in blocks of
    `average`) -> out_file (<stamp>_imu_raw_<method>.csv). Output is shown only on
    failure."""
    import raw_to_imu_csv as rti          # local: it pulls in pandas
    print(f"  raw:   running {method} on the raw stream -> "
          f"{os.path.basename(out_file)} ...", flush=True)
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rti.write_imu_csv(raw_bin, out_file, ahrs=method, verbose=False,
                              average=average)
    except BaseException:
        tail = "\n".join(buf.getvalue().strip().splitlines()[-15:])
        print(f"the reconstruction failed for --ahrs {method}:\n{tail}",
              file=sys.stderr)
        raise
    print(f"         done in {time.time() - t0:.0f} s")


def read_raw_vacc(path, t_ms):
    """vacc_fir from a sibling file, matched to the rows on win_start_ms ->
    (array, n_missing); unmatched rows are NaN."""
    idx, raw_rows, trunc = read_csv_stream(path, "win_start_ms")
    if "vacc_fir" not in idx:
        sys.exit(f"{os.path.basename(path)} lacks vacc_fir - was it made by an "
                 f"older raw_to_csv?")
    if trunc.reason:
        print(truncation_message(path, trunc))
    table = {int(r[idx["win_start_ms"]]): float(r[idx["vacc_fir"]]) for r in raw_rows}
    out = np.array([table.get(t, np.nan) for t in t_ms.tolist()], dtype=np.float64)
    return out, int(np.isnan(out).sum())


def method_rates(session, sources):
    """The rate each method ran at -> {name: Hz} (raw ODR, SFLP ODR, row rate, or the
    GPS rate set in cfg.csv). "onboard" (the capture's filter on the rows) and
    "csv" (replayed here) both ran at the row rate."""
    num = session.cfg_number
    imu_odr = num("imu_odr_hz")
    if imu_odr:
        imu_odr /= raw_average(session.directory, session.stamp)
    sflp_odr = num("sflp_odr_hz")
    # Row rate from cfg.csv, else from the rows.
    row_hz = num("output_rate_hz")
    if row_hz is None and len(session.t_ms) > 1:
        d = np.diff(session.t_ms.astype(np.float64))
        d = d[d > 0]
        row_hz = 1000.0 / float(np.median(d)) if len(d) else None

    rates = {}
    for m in METHODS:
        if m.make is None:
            rates[m.name] = sflp_odr      # the on-chip fusion, not the raw rate
        elif str(sources.get(m.name, "csv")).startswith("raw"):
            rates[m.name] = imu_odr       # the filter ran per raw sample
        else:
            rates[m.name] = row_hz
    rates["gps"] = num("gps_rate_hz")
    return rates


def raw_series(session, source_st):
    """Vertical accel from the raw stream for each AHRS method -> RawSeries."""
    directory, stamp = session.directory, session.stamp
    vacc_source = source_st.vacc_source
    cfg_ahrs = ahrs_from_cfg(session.cfg.get("orientation_name"))
    raw_bin = find_raw_log(directory, stamp, source_st.raw)
    if raw_bin and vacc_source != "csv":
        print(f"  raw:   {os.path.basename(raw_bin)} "
              f"({os.path.getsize(raw_bin) / 1e6:.1f} MB), the capture's filter is "
              f"{cfg_ahrs}")
    has_column = session.vacc_raw is not None and len(session.t_ms) > 0

    series, sources = {}, {"sflp": "onchip"}
    for name, key in ((m.name, m.key) for m in AHRS_METHODS):
        if name == cfg_ahrs and has_column and vacc_source != "csv":
            # Step 4 takes it from vacc_raw. Before vacc/vacc_fir (build_seq < 3,
            # the column vacc_madgwick) the capture ran its filter on board once per
            # averaged row, not per raw sample.
            sources[name] = ("onboard:imu.csv" if session.vacc_col == "vacc_madgwick"
                             else "raw:imu.csv")
            continue
        path = os.path.join(directory, f"{stamp}_imu_raw_{name}.csv")
        if vacc_source == "csv":
            sources[name] = "csv"
            continue
        if source_st.regen_raw or not os.path.isfile(path):
            if raw_bin is None:
                if vacc_source == "raw":
                    sys.exit(f"--vacc-source raw: found no {stamp}_raw.bin, and "
                             f"{os.path.basename(path)} does not exist. Without the "
                             f"raw log {name} can only be replayed - use auto or csv.")
                sources[name] = "csv"
                continue
            make_raw_csv(raw_bin, path, name, raw_average(directory, stamp))
        arr, missing = read_raw_vacc(path, session.t_ms)
        if missing:
            print(f"  raw:   {missing} of {len(session.t_ms)} rows without a match in "
                  f"{os.path.basename(path)} - they keep the replay value")
        series[key] = arr
        sources[name] = f"raw:{os.path.basename(path)}"
    return RawSeries(series, sources, method_rates(session, sources))
