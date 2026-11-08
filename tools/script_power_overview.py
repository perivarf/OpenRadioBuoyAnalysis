#!/usr/bin/env python3
"""Overview of the single (non-batch) power measurements (<name>_raw.csv) in a directory.

Current over time, smoothed with a moving average (boxcar), a component table
(power/single_components.py) and a table of mean current, power and energy per measurement in the
terminal. Power and energy are P = I * --voltage (default 5 V), from the raw files (sum files may
come from an older run with the same name).

Legend names can be overridden in <dir>/legend_names.csv, one line per measurement: name,legendname
(lines starting with # are ignored; measurements missing from the file keep their filename).

    python3 tools/script_power_overview.py
    python3 tools/script_power_overview.py --smooth 5000 --start 10 --slutt 300
    python3 tools/script_power_overview.py --voltage 4
    python3 tools/script_power_overview.py --dir . --out overview.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np                                          # noqa: E402

import _paths                                              # noqa: E402,F401
import power_io                                             # noqa: E402
import power_plots                                          # noqa: E402
import single_components as sc                              # noqa: E402
from power_stats import mean_std                            # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=power_io.DATA_DIR,
                   help="Directory with *_raw.csv (default: power_measurement/ in the repo root)")
    p.add_argument("--smooth", type=float, default=1000.0, help="Smoothing window [ms] (default: 1000; 0 = none)")
    p.add_argument("--start", type=float, help="Start time [s]")
    p.add_argument("--slutt", type=float, help="End time [s]")
    p.add_argument("--out", default=None, help="Output file (default: <dir>/oversikt_plot.png)")
    p.add_argument("--names", default=None, help="CSV with name,legendname (default: <dir>/legend_names.csv)")
    p.add_argument("--voltage", type=float, default=5.0,
                   help="Voltage [V] for power in the component table, P = I * V (default: 5)")
    args = p.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.dir, "*_raw.csv")))
    if not raw_files:
        print(f"No *_raw.csv files found in {args.dir}")
        return

    runs = []
    for raw in raw_files:
        prefix = raw[: -len("_raw.csv")]
        name = os.path.basename(prefix)
        t, _, i, _ = power_io.read_raw(raw)
        m = np.ones_like(t, dtype=bool)
        if args.start is not None:
            m &= t >= args.start
        if args.slutt is not None:
            m &= t <= args.slutt
        t, i_mA = t[m], i[m] * 1e3
        if len(t) < 2:
            print(f"{name}: too few samples in the selected time range, skipping")
            continue
        s = power_io.read_sum(f"{prefix}_sum.csv")
        mean_i, std_i = mean_std(t, i_mA)
        # Power and energy from current and --voltage, not from the measured voltage (power_W in the raw file)
        T = t[-1] - t[0]
        runs.append(dict(
            name=name, t=t, i_mA=i_mA, mean_i=mean_i, std_i=std_i,
            mean_p=mean_i * args.voltage, energy_mWh=mean_i * args.voltage * T / 3600,
            T=T, n=len(t), start=s.get("start_time", ""),
        ))

    # Sorted by mean current, so colors and legend get a natural order
    runs.sort(key=lambda r: r["mean_i"])
    legend_names = power_io.read_legend_names(args.names or os.path.join(args.dir, "legend_names.csv"))
    curves = [(r["t"], r["i_mA"], r["mean_i"], f"{legend_names.get(r['name'], r['name'])} ({r['mean_i']:.1f} mA)",
               f"C{k % 10}") for k, r in enumerate(runs)]

    comp = sc.component_rows(runs)
    power_plots.plot_currents(curves, args.out or os.path.join(args.dir, "oversikt_plot.png"), args.voltage,
                              args.smooth)
    for path in sc.write_component_tables(comp, os.path.join(args.dir, "komponent_tabell"), args.voltage):
        print(f"Lagret {path}")

    print(f"Effekt og energi ved {args.voltage:g} V")
    print(f"{'måling':<16}{'tid':>8}{'n':>8}{'I [mA]':>9}{'P [mW]':>9}{'E [mWh]':>10}")
    for r in runs:
        print(f"{r['name']:<16}{r['T']:>7.0f}s{r['n']:>8}{r['mean_i']:>9.2f}{r['mean_p']:>9.1f}"
              f"{r['energy_mWh']:>10.3f}")
    print()
    print(f"{'komponent':<16}{'I [mA]':>9}{'std':>8}{'P [mW]':>9}")
    for label, m, s in comp:
        print(f"{label:<16}{m:>9.2f}{s:>8.2f}{m * args.voltage:>9.1f}")


if __name__ == "__main__":
    main()
