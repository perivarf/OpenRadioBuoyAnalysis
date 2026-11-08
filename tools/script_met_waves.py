#!/usr/bin/env python3
"""Modelled significant wave height (Kystvarslingssenteret, KVS) at the time
and place of each session, next to the buoy's own Hs for one taper
(norce_kystverket/kystvarsling.py).

The IMU columns are read from existing <stamp>_ana_python.csv files (from plot.sh);
sessions without one are left out unless --all.

  --config NAME   IMU results from <measurement>/NAME/ (whatever taper that run
                  used); writes <measurement>/NAME/<prefix>_met_waves_NAME.csv (one
                  row per session) and <stamp>_met_waves_NAME.png (one table per
                  session) beside each session's results (as plot.sh). The rows
                  go to NAME/data/<prefix>_met_waves.csv first and every table is
                  drawn from there; --plot-only redraws without fetching anything
  otherwise      IMU results with --taper, from any directory; console only
                  unless --csv/--png

    python3 tools/script_met_waves.py SKJAERHALDEN --config 0.03-0.05
    python3 tools/script_met_waves.py "20260731 Skjærhalden" --taper 0.05 0.1
"""

import argparse
import os
import sys
from glob import glob

import _paths                                               # noqa: F401
import kystvarsling                                         # noqa: E402
import session_sets                                         # noqa: E402
from configs import (check_config, config_dir, config_suffix,  # noqa: E402
                     data_path)

# The taper of the IMU results shown when neither --taper nor --config is given [Hz].
DEFAULT_TAPER = (0.05, 0.1)


def resolve(name):
    """A directory, or a set in session_sets -> (directory, filename prefix)."""
    path = name if os.path.isabs(name) else os.path.join(kystvarsling.ROOT, name)
    if os.path.isdir(path) and name not in session_sets.SETS:
        return path, os.path.basename(os.path.normpath(path))
    sset = session_sets.get_set(name, quiet=True)
    return sset.directory, sset.filename


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", metavar="SET_OR_DIR",
                    help="sets in session_sets.SETS, measurement or session "
                         "directories (default: every 20*/ directory)")
    ap.add_argument("--config", help="read the IMU results from and write the table "
                                     "to <measurement>/<config>/")
    ap.add_argument("--taper", nargs=2, type=float, default=DEFAULT_TAPER,
                    metavar=("F1", "F2"),
                    help="without --config: the fixed taper of the IMU results [Hz] "
                         f"(default {DEFAULT_TAPER[0]:g} {DEFAULT_TAPER[1]:g})")
    ap.add_argument("--all", action="store_true",
                    help="also sessions without IMU results")
    ap.add_argument("--csv", help="also write the table here (without --config)")
    ap.add_argument("--png", metavar="DIR",
                    help="also write <stamp>_met_waves.png per session to DIR "
                         "(without --config)")
    ap.add_argument("--sessions", nargs="+", metavar="STAMP",
                    help="with --config: one table with only these sessions side by "
                         "side, in this order, instead of the usual files")
    ap.add_argument("--labels", nargs="+",
                    help="column labels for --sessions (default: the stamps)")
    ap.add_argument("--name", default="selected",
                    help="with --sessions: the file is <prefix>_met_waves_<name>_<config>.png")
    ap.add_argument("--ab", nargs="+", metavar="STAMP",
                    help="with --config: Hs per filter for buoy A and B at each "
                         "location, given as pairs A1 B1 A2 B2 ...; writes "
                         "<prefix>_hs_ab_<config>.png instead of the usual files")
    ap.add_argument("--ab-labels", nargs="+",
                    help="one label per --ab pair (default 1, 2, ...)")
    ap.add_argument("--plot-only", action="store_true",
                    help="with --config: draw from data/<prefix>_met_waves.csv, fetch "
                         "and compute nothing")
    args = ap.parse_args(argv)
    if args.plot_only and not args.config:
        ap.error("--plot-only needs --config")
    if (args.sessions or args.ab) and not args.config:
        ap.error("--sessions and --ab need --config")
    if args.ab and len(args.ab) % 2:
        ap.error("--ab takes pairs: A1 B1 A2 B2 ...")
    if args.ab_labels and len(args.ab_labels) != len(args.ab or []) // 2:
        ap.error("--ab-labels needs one label per --ab pair")
    if args.labels and len(args.labels) != len(args.sessions or []):
        ap.error("--labels needs one label per --sessions entry")
    names = args.paths or sorted(glob(os.path.join(kystvarsling.ROOT, "20*/")))

    if args.config:
        config = check_config(args.config)
        for name in names:
            directory, prefix = resolve(name)
            # The rows go to data/ first, and every table is drawn from there.
            data = data_path(os.path.join(config_dir(directory, config),
                                          f"{prefix}_met_waves.png"))
            if not args.plot_only:
                kystvarsling.write_rows(kystvarsling.compare(
                    [directory], all_sessions=args.all, config=config), data)
            rows = kystvarsling.read_rows(data)
            # Sessions by stamp or folder name (the folder wins on a shared stamp).
            by_stamp = {r["stamp"]: r for r in rows}
            by_stamp.update({r["folder"]: r for r in rows})
            wanted = (args.sessions or []) + (args.ab or [])
            missing = [s for s in wanted if s not in by_stamp]
            if missing:
                sys.exit(f"no IMU results in {config}/ for {', '.join(missing)}")
            if args.ab:
                pairs = [(by_stamp[a], by_stamp[b])
                         for a, b in zip(args.ab[::2], args.ab[1::2])]
                labels = args.ab_labels or [str(n + 1) for n in range(len(pairs))]
                kystvarsling.print_ab(pairs, labels)
                png = config_suffix(f"{prefix}_hs_ab.png", config)
                kystvarsling.save_ab_png(pairs, labels, os.path.join(
                    config_dir(directory, config), png))
                print(f"  wrote {png}")
                continue
            if args.sessions:
                picked = [by_stamp[s] for s in args.sessions]
                labels = args.labels or args.sessions
                kystvarsling.print_combined(picked, labels)
                png = config_suffix(f"{prefix}_met_waves_{args.name}.png", config)
                kystvarsling.save_combined_png(picked, labels, os.path.join(
                    config_dir(directory, config), png))
                print(f"  wrote {png}")
                continue
            kystvarsling.print_table(rows)
            out = config_dir(directory, config)
            csv_path = os.path.join(out, f"{prefix}_met_waves_{config}.csv")
            kystvarsling.write_csv(rows, csv_path)
            print(f"  wrote {os.path.basename(csv_path)}")
            for row in rows:
                if not row["ana_file"]:
                    continue
                ses_out = os.path.dirname(os.path.join(kystvarsling.ROOT, row["ana_file"]))
                png = config_suffix(f"{row['stamp']}_met_waves.png", config)
                kystvarsling.save_png(row, os.path.join(ses_out, png))
                print(f"  wrote {png}")
        return

    taper = tuple(args.taper)
    rows = kystvarsling.compare([resolve(n)[0] for n in names], taper, args.all)
    kystvarsling.print_table(rows)
    if args.csv:
        kystvarsling.write_csv(rows, args.csv)
        print(f"wrote {args.csv}", file=sys.stderr)
    if args.png:
        os.makedirs(args.png, exist_ok=True)
        for row in rows:
            kystvarsling.save_png(row, os.path.join(args.png,
                                                    f"{row['session']}_met_waves.png"))
        print(f"wrote {len(rows)} PNG files to {args.png}", file=sys.stderr)


if __name__ == "__main__":
    main()
