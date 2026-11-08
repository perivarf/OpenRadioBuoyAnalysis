#!/usr/bin/env python3
"""CPU time per orientation filter (Madgwick, Kalman, Kalman NXP) for every session
in a set (analysis/filter_timing.py). Writes <measurement>/<prefix>_filter_timing.csv,
one row per session and filter, and prints it.

    python3 tools/script_filter_timing.py SKJAERHALDEN
"""

import argparse
import os
import sys

import _paths                                               # noqa: F401
import filter_timing                                        # noqa: E402
import session_sets                                         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", help="a set in session_sets.SETS or a measurement directory")
    ap.add_argument("--out", help="CSV path (default "
                                  "<measurement>/<prefix>_filter_timing.csv)")
    args = ap.parse_args(argv)

    sset = session_sets.get_set(args.set, quiet=True)
    rows = []
    for sess_dir in session_sets.check(sset):
        print(f"  timing {os.path.basename(sess_dir)} ...", flush=True)
        for r in filter_timing.time_session(sess_dir):
            print(f"    {r['filter']:11s} {r['input']:8s} {r['rate_hz']:>6s} Hz "
                  f"{r['n_samples']:>7d} samples  {r['cpu_s']:>7s} s  "
                  f"{r['us_per_sample']:>6s} µs/sample  x{r['x_realtime']} realtime")
            rows.append(r)
    out = args.out or os.path.join(sset.directory, f"{sset.filename}_filter_timing.csv")
    filter_timing.write_csv(rows, out)
    print(f"  wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
