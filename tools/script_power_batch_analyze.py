#!/usr/bin/env python3
"""Splits a power log from the batch board (firmware: drifter/src/power_batch.cpp) into tests.

Finds the ID and capture-start markers (power/batch_markers.py), measures each test over the
analysis window (--window after capture start / marker end) and over the whole segment, and
writes <prefix>_tests.csv.

    python3 tools/script_power_batch_analyze.py batch1_raw.csv
    python3 tools/script_power_batch_analyze.py batch1_raw.csv --plot
    python3 tools/script_power_batch_analyze.py batch1_raw.csv --split   # one _raw.csv per test
    python3 tools/script_power_batch_analyze.py batch2_raw.csv --window 20 100
"""

import argparse

import _paths                                               # noqa: F401
import batch_analysis as ba                                 # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="<prefiks>_raw.csv fra script_power_log.py / script_power_batch_log.py")
    ap.add_argument("--window", nargs=2, type=float, default=list(ba.WINDOW), metavar=("FRA", "TIL"),
                    help="Analysevindu [s] etter capture-start / markørslutt (standard 30 90)")
    ap.add_argument("--delta", type=float, default=ba.DELTA_MA, help="Minste strømsprang for en kant [mA] (standard 3)")
    ap.add_argument("--capture-s", type=float, default=ba.CAPTURE_S,
                    help="Capture-lengden i firmware (power_batch_test_ms) [s], for gyldighetssjekken")
    ap.add_argument("--min-score", type=float, default=ba.MIN_SCORE,
                    help="Laveste score for en capture-start som ikke er streng (standard 1.3)")
    ap.add_argument("--capture-tail", type=float, default=ba.CAPTURE_TAIL_S,
                    help="Tid fra capture-slutt til neste markør [s], for anslått capture-start når "
                         "markøren drukner i GPS-strømmen. Måles fra sikre capture-starter i loggen "
                         "når det finnes noen (standard 7)")
    ap.add_argument("--edge-side", type=float, default=ba.EDGE_SIDE_S, help="Medianvindu på hver side av en kant [s]")
    ap.add_argument("--split", action="store_true",
                    help="Skriv <prefiks>_<id>_<navn>_raw.csv per måling, t=0 ved vindusankeret")
    ap.add_argument("--plot", action="store_true", help="Vis plott med markører og vinduer")
    ap.add_argument("--out", help="Lagre plottet til fil i stedet for å vise det")
    args = ap.parse_args(argv)

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
    ba.analyze_log(args.raw, tuple(args.window), args.delta, args.capture_s, args.min_score, args.capture_tail,
                   args.edge_side, split=args.split, plot=args.plot, out=args.out)


if __name__ == "__main__":
    main()
