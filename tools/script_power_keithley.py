#!/usr/bin/env python3
"""Batch results corrected to the Keithley 2000 (the GPP-4323 reads lower; see the fit it prints).

Like script_power_batch_results.py, but each test value comes from:
    test 1, 2, 3, 8   Keithley reading (K)
    test 4            K8 + a * (GW4 - GW8) from --imu-log
    test 5            --new-log, corrected as a * GW + b
    test 6, 7         --old-log, corrected as a * GW + b
where a and b are a linear fit to --keithley.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np                                          # noqa: E402

import _paths                                               # noqa: E402,F401
import batch_components as bc                               # noqa: E402
import power_plots                                          # noqa: E402
from power_io import DATA_DIR                               # noqa: E402
from power_stats import mean_std                            # noqa: E402

STABLE_IDS = [1, 2, 3, 8]
NEW_IDS = [5]
OLD_IDS = [6, 7]

IMU_NOTE = ("IMU from an earlier run (test 4 - 8); the latest run gave 0.35-0.4 mA. The difference "
            "also includes the capture loop against busy spin on the MCU.")
IMU_NOTE_TEX = (r"\acs{IMU} from an earlier run (test 4 $-$ 8); the latest run gave "
                r"\SIrange{0.35}{0.4}{\milli\ampere}. The difference also includes the capture loop "
                r"against busy spin on the \acs{MCU}.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new-log", default=os.path.join(DATA_DIR, "batch_all_2_raw.csv"),
                   help="Log with Keithley readings (tests 1, 2, 3, 8, 5)")
    p.add_argument("--old-log", default=os.path.join(DATA_DIR, "batch2_raw.csv"), help="Log with tests 6 and 7")
    p.add_argument("--imu-log", default=os.path.join(DATA_DIR, "batch1_4_raw.csv"),
                   help="Log with tests 4 and 8 for the IMU difference")
    p.add_argument("--keithley", default=os.path.join(DATA_DIR, "keithley_readings.csv"))
    p.add_argument("--out-dir", default=os.path.join(DATA_DIR, "batch_results_keithley"))
    p.add_argument("--window", nargs=2, type=float, default=[30.0, 90.0], metavar=("FROM", "TO"),
                   help="Window [s] after capture start / marker end (default 30 90)")
    p.add_argument("--capture-s", type=float, default=120.0, help="Capture length in firmware [s]")
    p.add_argument("--smooth", type=float, default=1000.0, help="Smoothing window [ms] (default 1000; 0 = none)")
    p.add_argument("--voltage", type=float, default=3.7, help="Voltage [V] for P = I * V (default 3.7)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    w0, w1 = window = tuple(args.window)
    new = bc.window_runs(args.new_log, args.out_dir, window, args.capture_s)
    old = bc.window_runs(args.old_log, args.out_dir, window, args.capture_s)
    imu = bc.window_runs(args.imu_log, args.out_dir, window, args.capture_s)

    kei = bc.read_keithley(args.keithley)
    a, b, gw, k = bc.keithley_fit(kei, new)
    resid = k - (a * gw + b)
    print(f"\nKorreksjon: K = {a:.4f} * GW + {b:.3f}  ({len(k)} avlesninger, største restavvik "
          f"{np.max(np.abs(resid)):.3f} mA)")
    for g, kk, r in zip(gw, k, resid):
        print(f"    GW {g:7.2f}  K {kk:7.2f}  K - GW {kk - g:+5.2f}  rest {r:+6.3f}")
    kmap = {int(row["id"]): float(row["keithley_mA"]) for row in kei if row["id"].strip()}

    runs = {}
    for tid in STABLE_IDS:
        r = new[tid]
        shift = kmap[tid] - r["mean_i"]
        runs[tid] = dict(r, i_mA=r["i_mA"] + shift, mean_i=kmap[tid], gw_mean=r["mean_i"], offset=shift,
                         source="K")

    # Test 4 = K8 + IMU from the earlier run (same log for tests 4 and 8, so the offset cancels)
    imu_mA = a * (imu[4]["mean_i"] - imu[8]["mean_i"])
    r4 = imu[4]
    val4 = kmap[8] + imu_mA
    runs[4] = dict(r4, i_mA=r4["i_mA"] + (val4 - r4["mean_i"]), mean_i=val4, gw_mean=r4["mean_i"],
                   offset=val4 - r4["mean_i"], source="GW")
    print(f"\nIMU = {a:.4f} * ({imu[4]['mean_i']:.3f} - {imu[8]['mean_i']:.3f}) = {imu_mA:.3f} mA "
          f"({os.path.basename(args.imu_log)}); test 4 = K8 + IMU = {val4:.2f} mA "
          f"(Keithley i ny kjøring: {kmap.get(4, float('nan')):.2f} mA, IMU der "
          f"{kmap.get(4, float('nan')) - kmap[8]:.2f} mA)")

    for src, ids in ((new, NEW_IDS), (old, OLD_IDS)):
        for tid in ids:
            r = src[tid]
            ii = a * r["i_mA"] + b
            mean, std = mean_std(r["t"], ii)
            runs[tid] = dict(r, i_mA=ii, mean_i=mean, std_i=std, gw_mean=r["mean_i"], offset=mean - r["mean_i"],
                             source="GW")

    print()
    curves = bc.current_curves(runs, lambda r: f"{r['id']}: {bc.LABELS[r['id']]} ({r['mean_i']:.1f} mA, {r['source']})")
    power_plots.plot_currents(curves, os.path.join(args.out_dir, "energy_overview.png"), args.voltage, args.smooth,
                              "Time in analysis window [s]", (0, w1 - w0),
                              title_extra=". K: Keithley 2000, GW: GPP-4323 corrected", title_size=11)
    bc.write_batch_tests(os.path.join(args.out_dir, "batch_tests.csv"), runs, {
        "log": lambda r: r["src"],
        "source": lambda r: r["source"],
        "gw_mean_current_mA": lambda r: f"{r['gw_mean']:.3f}",
        "offset_mA": lambda r: f"{r['offset']:.3f}",
    }, args.voltage)

    comp = bc.component_rows(runs)
    sources = {}
    for label, _, _ in comp:
        terms = bc.component_terms(label, runs)
        sources[label] = "K" if all(runs[t]["source"] == "K" for t in terms) else "GW"
    note_extra = (f" K: Keithley 2000 reading. GW: GPP-4323 readback corrected as K = {a:.3f}·GW + {b:.2f} "
                  f"(fit to {len(k)} readings).\n†{IMU_NOTE}",
                  rf" K: Keithley 2000 reading. GW: GPP-4323 readback corrected as "
                  rf"$K = {a:.3f}\,\mathrm{{GW}} + \SI{{{b:.2f}}}{{\milli\ampere}}$ (fit to {len(k)} readings)."
                  rf"\newline $^{{\dagger}}${IMU_NOTE_TEX}")
    for path in bc.write_component_tables(comp, os.path.join(args.out_dir, "komponent_tabell"), args.voltage,
                                          w1 - w0, sources=sources, marks={"IMU": ("†", r"$^{\dagger}$")},
                                          note_extra=note_extra):
        print(f"Lagret {path}")

    print(f"\n{'test':<44}{'GW':>8}{'I [mA]':>9}{'std':>7}  kilde")
    for tid in sorted(runs):
        r = runs[tid]
        print(f"{tid}: {bc.LABELS[tid]:<41}{r['gw_mean']:>8.2f}{r['mean_i']:>9.2f}{r['std_i']:>7.2f}  "
              f"{r['source']} ({r['src']})")
    print(f"\n{'komponent':<44}{'I [mA]':>9}{'std':>7}{'P [mW]':>9}  kilde")
    for label, m, s in comp:
        print(f"{label:<44}{m:>9.2f}{s:>7.2f}{m * args.voltage:>9.1f}  {sources[label]}")
    total = dict((label, m) for label, m, _ in comp)
    parts = sum(m for label, m, _ in comp if label != "Total")
    print(f"\nSum av komponentene {parts:.2f} mA, Total {total.get('Total', float('nan')):.2f} mA")


if __name__ == "__main__":
    main()
