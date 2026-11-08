#!/usr/bin/env python3
"""Current and power for one or more measurements from script_power_log.py.

Reads <prefix>_raw.csv (and <prefix>_sum.csv if it exists) and writes <prefix>_plot.png.

    python3 tools/script_power_plot.py --csv test1
    python3 tools/script_power_plot.py --csv test1 test2 test3 --start 180 --slutt 600
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import _paths                                               # noqa: E402,F401
import power_plots                                          # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True, help="One or more prefixes (without _raw.csv)")
    p.add_argument("--start", type=float, help="Start time [s] (default: first measurement)")
    p.add_argument("--slutt", type=float, help="End time [s] (default: last measurement)")
    args = p.parse_args()

    for prefix in args.csv:
        power_plots.plot_raw_file(prefix, args.start, args.slutt)


if __name__ == "__main__":
    main()
