#!/usr/bin/env python3
"""NorKyst v3 surface current, wind and sea level at the time and place of each
session, next to the buoy's own drift (models/norkyst160.py).

NorKyst 160 m (Oslofjord) is used where it reaches, else NorKyst 800 m. The model
has no waves - see script_met_waves.py for Hs.

Directions: current and drift TOWARDS, wind FROM, degrees from north.

    python3 tools/script_norkyst160.py                      # every measurement
    python3 tools/script_norkyst160.py "20260830 Skjærhalden" --csv norkyst.csv
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from glob import glob

import _paths                                               # noqa: F401
import norkyst160                                           # noqa: E402
from pipeline.session_input import read_kv                  # noqa: E402
from session_points import drift, find_sessions, position, window  # noqa: E402

ROOT = os.path.dirname(_paths.TOOLS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*",
                    help="measurement or session directories (default: every "
                         "20*/ directory in the repository)")
    ap.add_argument("--max-km", type=float, default=2.0,
                    help="no model values when the nearest wet cell is further away")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args(argv)
    paths = args.paths or sorted(glob(os.path.join(ROOT, "20*/")))

    cols = [("session", "Session", 16), ("start_utc", "Start UTC", 17),
            ("minutes", "min", 4), ("model", "Model", 8), ("cell_km", "km", 4),
            ("drift", "Drift m/s->", 12), ("current", "Current m/s->", 14),
            ("wind", "Wind m/s<-", 11), ("zeta", "Sea level m", 18), ("note", "", 0)]
    rows = []
    for ses_path in find_sessions(paths):
        stamp = os.path.basename(ses_path)[:-len("_ses.csv")]
        ses = read_kv(ses_path)
        win, pos = window(ses_path, ses), position(ses_path, ses)
        row = {k: "" for k, _, _ in cols}
        row["session"] = stamp
        if win is None or pos is None:
            row["note"] = "no GPS time" if win is None else "no position"
            rows.append(row)
            continue
        t0, t1 = win
        row.update(start_utc=datetime.fromtimestamp(t0, tz=timezone.utc)
                   .strftime("%Y-%m-%d %H:%M"), minutes=f"{(t1 - t0) / 60.0:.0f}")
        dr = drift(ses_path, ses)
        if dr:
            row["drift"] = f"{dr[0]:.2f} {dr[1]:3.0f}°"
        try:
            cell = norkyst160.nearest_wet_cell(*pos, t0, max_km=args.max_km)
        except OSError as e:
            row["note"] = f"no model file ({e})"
            rows.append(row)
            continue
        if cell[5] > args.max_km:          # inf: no wet cell in the search window
            row["note"] = "no model sea nearby"
            rows.append(row)
            continue
        row["model"] = cell[0][0][len("norkyst"):] + " m"
        row["cell_km"] = f"{cell[5]:.1f}"
        print(f"  {stamp}: {cell[0][2]} cell {cell[3]:.4f}N {cell[4]:.4f}E "
              f"({cell[5]:.2f} km) ...", file=sys.stderr, flush=True)
        m = norkyst160.series(cell, t0, t1)
        if m is None:
            row["note"] = "no model file for the window"
        else:
            row.update(current=f"{m['cur_speed']:.2f} {m['cur_dir']:3.0f}°",
                       wind=f"{m['wind_speed']:.1f} {m['wind_dir']:3.0f}°",
                       zeta=f"{m['zeta_mean']:+.2f} ({m['zeta_min']:+.2f}..{m['zeta_max']:+.2f})")
        rows.append(row)

    print("  ".join(f"{h:<{w}}" for _, h, w in cols))
    for r in rows:
        print("  ".join(f"{r[k]:<{w}}" for k, _, w in cols))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k, _, _ in cols])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
