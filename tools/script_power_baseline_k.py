#!/usr/bin/env python3
"""Batch results with the deep-sleep floor from the Keithley 2000. This is the thesis version
(energy_overview.png, table tab:power_components).

As script_power_batch_results.py, but the deep-sleep floor (test 1) is set to the Keithley reading.
The other components are differences between GPP-4323 readbacks (GW Δ), where a fixed readback
offset cancels, so they are unchanged; the total is shifted by δ = K1 - GW1. The time series in
energy_overview.png is the raw GW readback.

Writes to <out-dir> (default batch_results_baseline_k/): energy_overview.png,
komponent_tabell.{csv,tex,png}, batch_tests.csv.

    python3 tools/script_power_baseline_k.py
    python3 tools/script_power_baseline_k.py --keithley-floor 4.8
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import _paths                                               # noqa: E402,F401
import batch_components as bc                               # noqa: E402
import power_plots                                          # noqa: E402
from power_io import DATA_DIR                               # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch1", default=os.path.join(DATA_DIR, "batch1_4_raw.csv"), help="Log with tests 1-5, 8")
    p.add_argument("--batch2", default=os.path.join(DATA_DIR, "batch2_raw.csv"), help="Log with tests 6, 7")
    p.add_argument("--keithley-floor", type=float, default=4.8,
                   help="Keithley reading of the deep-sleep floor (test 1) [mA] (default 4.8)")
    p.add_argument("--out-dir", default=os.path.join(DATA_DIR, "batch_results_baseline_k"))
    p.add_argument("--window", nargs=2, type=float, default=[30.0, 90.0], metavar=("FROM", "TO"),
                   help="Window [s] after capture start / marker end (default 30 90)")
    p.add_argument("--capture-s", type=float, default=120.0, help="Capture length in firmware [s]")
    p.add_argument("--smooth", type=float, default=1000.0, help="Smoothing window [ms] (default 1000; 0 = none)")
    p.add_argument("--voltage", type=float, default=3.7, help="Voltage [V] for P = I * V (default 3.7)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w0, w1 = window = tuple(args.window)
    runs = {}
    for raw in (args.batch1, args.batch2):
        runs.update(bc.window_runs(raw, args.out_dir, window, args.capture_s))

    power_plots.plot_currents(bc.current_curves(runs), os.path.join(args.out_dir, "energy_overview.png"),
                              args.voltage, args.smooth, "Time in analysis window [s]", (0, w1 - w0))

    # One fixed offset for all tests: the differences are unchanged, the floor becomes K
    delta = args.keithley_floor - runs[1]["mean_i"]
    print(f"\nδ = K1 - GW1 = {args.keithley_floor:.2f} - {runs[1]['mean_i']:.2f} = {delta:+.2f} mA")
    for r in runs.values():
        r["gw_mean"] = r["mean_i"]
        r["mean_i"] = r["mean_i"] + delta

    bc.write_batch_tests(os.path.join(args.out_dir, "batch_tests.csv"), runs, {
        "log": lambda r: r["src"],
        "gw_mean_current_mA": lambda r: f"{r['gw_mean']:.3f}",
        "delta_mA": lambda r: f"{delta:.3f}",
    }, args.voltage)

    comp = bc.component_rows(runs)
    sources = {label: "GW Δ" for label, _, _ in comp}
    sources["Deep-sleep floor*"] = "K"
    sources["Total"] = "K + GW Δ"
    note_extra = (f" K: Keithley 2000 reading. GW Δ: difference between GPP-4323 readbacks, where a "
                  f"constant readback offset cancels. The floor reads {delta:.1f} mA higher on the Keithley "
                  f"than on the GPP-4323, so the Total is {delta:.1f} mA above the GPP-4323 readback of the "
                  f"full configuration.",
                  rf" K: Keithley 2000 reading. GW $\Delta$: difference between GPP-4323 readbacks, where a "
                  rf"constant readback offset cancels. The floor reads \SI{{{delta:.1f}}}{{\milli\ampere}} "
                  rf"higher on the Keithley than on the GPP-4323, so the Total is "
                  rf"\SI{{{delta:.1f}}}{{\milli\ampere}} above the GPP-4323 readback of the full configuration "
                  rf"(figure~\ref{{fig:energy_overview}}).")
    for path in bc.write_component_tables(comp, os.path.join(args.out_dir, "komponent_tabell"), args.voltage,
                                          w1 - w0, sources=sources, note_extra=note_extra):
        print(f"Lagret {path}")

    print(f"\n{'komponent':<44}{'I [mA]':>9}{'std':>7}{'P [mW]':>9}  kilde")
    for label, m, s in comp:
        print(f"{label:<44}{m:>9.2f}{s:>7.2f}{m * args.voltage:>9.1f}  {sources[label]}")


if __name__ == "__main__":
    main()
