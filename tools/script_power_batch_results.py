#!/usr/bin/env python3
"""Figure and component table for the batch board, straight from the GPP-4323 readback.

Analyzes the log with all tests (batch_all*, tests 1-10 with card swaps), or the older batch 1
(tests 1-5, 8) and batch 2 (tests 6-7), and uses the analysis window of each test (default
30-90 s after capture start / marker end). Writes to <out-dir> (default batch_results/):

    energy_overview.png            current over time in the window, one curve per test
    komponent_tabell.{csv,tex,png} per-component contributions as differences between tests,
                                   with typical datasheet values
    batch_tests.csv                mean/std/power per test
    <log>_analyse.png              analysis plot for each log

Power is P = I * --voltage (default 3.7 V, the VBAT voltage on the board).

    python3 tools/script_power_batch_results.py
    python3 tools/script_power_batch_results.py --log batch_all_raw.csv
    python3 tools/script_power_batch_results.py --batch1 batch1_3_raw.csv --batch2 batch2_raw.csv
    python3 tools/script_power_batch_results.py --window 40 100 --smooth 500
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import _paths                                               # noqa: E402,F401
import batch_components as bc                               # noqa: E402
import power_plots                                          # noqa: E402
from power_io import DATA_DIR, newest                       # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", help="Logg med alle testene (standard: nyeste batch_all*_raw.csv, "
                                 "ellers batch1 + batch2)")
    p.add_argument("--batch1", help="Logg for batch 1 (standard: nyeste batch1*_raw.csv)")
    p.add_argument("--batch2", help="Logg for batch 2 (standard: nyeste batch2*_raw.csv)")
    p.add_argument("--out-dir", default=os.path.join(DATA_DIR, "batch_results"))
    p.add_argument("--window", nargs=2, type=float, default=[30.0, 90.0], metavar=("FRA", "TIL"),
                   help="Vindu [s] etter capture-start / markørslutt (standard 30 90)")
    p.add_argument("--capture-s", type=float, default=120.0, help="Capture-lengden i firmware [s]")
    p.add_argument("--smooth", type=float, default=1000.0, help="Glattevindu [ms] (standard 1000; 0 = ingen)")
    p.add_argument("--voltage", type=float, default=3.7, help="Spenning [V] for P = I * V (standard 3.7)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w0, w1 = window = tuple(args.window)
    log = args.log or (None if args.batch1 or args.batch2
                       else newest(os.path.join(DATA_DIR, "batch_all*_raw.csv")))
    raws = [log] if log else [args.batch1 or newest(os.path.join(DATA_DIR, "batch1*_raw.csv")),
                              args.batch2 or newest(os.path.join(DATA_DIR, "batch2*_raw.csv"))]
    runs = {}
    for raw in raws:
        if raw:
            runs.update(bc.window_runs(raw, args.out_dir, window, args.capture_s))
    if not runs:
        print("Ingen gyldige tester funnet")
        return
    print(f"\nTester med: {sorted(runs)}")

    power_plots.plot_currents(bc.current_curves(runs), os.path.join(args.out_dir, "energy_overview.png"),
                              args.voltage, args.smooth, "Time in analysis window [s]", (0, w1 - w0))
    bc.write_batch_tests(os.path.join(args.out_dir, "batch_tests.csv"), runs, {
        "source": lambda r: r["src"],
        "window_start_s": lambda r: f"{r['start']:.2f}",
        "capture_estimated": lambda r: int(r["estimated"]),
    }, args.voltage)

    comp = bc.component_rows(runs)
    for path in bc.write_component_tables(comp, os.path.join(args.out_dir, "komponent_tabell"), args.voltage,
                                          w1 - w0):
        print(f"Lagret {path}")

    print(f"\n{'test':<44}{'I [mA]':>9}{'std':>7}{'P [mW]':>9}")
    for tid in sorted(runs):
        r = runs[tid]
        est = " (capture-start anslått)" if r["estimated"] else ""
        print(f"{tid}: {bc.LABELS[tid]:<41}{r['mean_i']:>9.2f}{r['std_i']:>7.2f}"
              f"{r['mean_i'] * args.voltage:>9.1f}{est}")
    if all(k in runs for k in (1, 2, 3, 8)):
        awake, asleep = runs[8]["mean_i"] - runs[3]["mean_i"], runs[2]["mean_i"] - runs[1]["mean_i"]
        print(f"\nKontroll: GNSS standby - deep sleep er {awake:.2f} mA med MCU våken (8 - 3) og "
              f"{asleep:.2f} mA i Stop2 (2 - 1)")
    if all(k in runs for k in (1, 4, 9, 10)):
        floor, capture = runs[9]["mean_i"] - runs[1]["mean_i"], runs[10]["mean_i"] - runs[4]["mean_i"]
        print(f"Kontroll: SD-kortet i hvile er {floor:.2f} mA i deep sleep (9 - 1) og "
              f"{capture:.2f} mA under IMU-capture (10 - 4)")
    print(f"\n{'komponent':<44}{'I [mA]':>9}{'std':>7}{'P [mW]':>9}{'datablad':>10}")
    for label, m, s in comp:
        print(f"{label:<44}{m:>9.2f}{s:>7.2f}{m * args.voltage:>9.1f}{bc.DATASHEET.get(label, ''):>10}")


if __name__ == "__main__":
    main()
