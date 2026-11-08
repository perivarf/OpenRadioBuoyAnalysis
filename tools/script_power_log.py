#!/usr/bin/env python3
"""Logs voltage and current of a load (the ORB board) from the GW Instek GPP-4323 over LAN.

Writes <prefix>_raw.csv (every sample) and <prefix>_sum.csv (mean current, power, energy).

    # Default: log CH3 for 5 min without touching the supply, CSV with timestamp
    python3 tools/script_power_log.py

    # Set 3.3 V / 200 mA current limit, turn output on, log until Ctrl+C, then switch off
    python3 tools/script_power_log.py --voltage 3.3 --current-limit 0.2 --output-on --output-off-at-end

    # Save to test1_raw.csv and test1_sum.csv, and show the plot
    python3 tools/script_power_log.py --duration 120 --csv test1 --plot
"""

import argparse
import bisect
import csv
import time

import _paths                                               # noqa: F401
import gpp4323                                              # noqa: E402
import power_io                                             # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.2.124")
    p.add_argument("--port", type=int, default=1026)
    p.add_argument("--channel", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--voltage", type=float, help="Set output voltage [V]")
    p.add_argument("--current-limit", type=float, help="Set current limit [A]")
    p.add_argument("--output-on", action="store_true", help="Turn the output on before measurement")
    p.add_argument("--output-off-at-end", action="store_true", help="Turn the output off after measurement")
    p.add_argument("--power-cycle", action="store_true",
                   help="Power-cycle the output before logging (t=0 is set at switch-on)")
    p.add_argument("--off-time", type=float, default=3.0, help="How long the output stays off in --power-cycle [s]")
    p.add_argument("--settle", type=float, default=1.0, help="Wait time after switch-on before logging [s]")
    p.add_argument("--skip", type=float, default=0.0,
                   help="Ignore the first SKIP seconds in the result (still logged to CSV)")
    p.add_argument("--duration", type=float, default=300.0, help="Measurement time [s] (default: 300; 0 = until Ctrl+C)")
    p.add_argument("--interval", type=float, default=0.1, help="Desired time between measurements [s]")
    p.add_argument("--report-every", type=float, default=10.0, help="Interval for printing average values [s]")
    p.add_argument("--csv", help="Prefix for files: <prefix>_raw.csv and <prefix>_sum.csv "
                                 "(default: maaling_CH<n>_<timestamp>)")
    p.add_argument("--plot", action="store_true", help="Show a plot of current and power")
    args = p.parse_args()

    ch = args.channel
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = power_io.strip_prefix(args.csv or time.strftime(f"maaling_CH{ch}_%Y%m%d_%H%M%S"))
    raw_file, sum_file = f"{prefix}_raw.csv", f"{prefix}_sum.csv"
    psu = gpp4323.GPP4323(args.host, args.port)
    idn = psu.idn()
    print("Connected:", idn)

    gpp4323.configure(psu, ch, args.voltage, args.current_limit)
    if args.power_cycle:
        print(f"Turning off CH{ch} for {args.off_time} s ...")
        t0 = gpp4323.power_cycle(psu, ch, args.off_time)   # t=0 at switch-on, so the startup is logged
        print(f"CH{ch} switched back on")
    elif args.output_on:
        psu.output(ch, True)
        time.sleep(args.settle)
        t0 = time.perf_counter()
    else:
        t0 = time.perf_counter()

    t_list, v_list, i_list, p_list = [], [], [], []
    # Cumulative energy [J] and charge [C] from t=0, per measurement (for moving average)
    e_cum, q_cum = [], []
    energy_j = 0.0
    charge_c = 0.0
    skip_idx = None  # index of the first measurement with t >= skip
    next_report = args.report_every

    def window_avg(t_from, k_end):
        """Average current [A], power [W] between t_from and measurement k_end, based on the integrals."""
        k0 = bisect.bisect_left(t_list, t_from, hi=k_end)
        dt = t_list[k_end] - t_list[k0]
        if dt <= 0:
            return float("nan"), float("nan")
        return (q_cum[k_end] - q_cum[k0]) / dt, (e_cum[k_end] - e_cum[k0]) / dt

    print(f"Logging CH{ch} ... (Ctrl+C to stop)")
    if args.skip > 0:
        print(f"Cumulative values start at t = {args.skip:.0f} s (before that: from t = 0, marked *)")
    print(f"{'t [s]':>7} | {'last 10 s':^17} | {'last 30 s':^17} | {'cumulative':^27}")
    print(f"{'':>7} | {'mA':>7} {'mW':>9} | {'mA':>7} {'mW':>9} | {'mA':>7} {'mW':>9} {'mWh':>9}")
    print("-" * 80)

    raw_f = open(raw_file, "w", newline="")
    raw_w = csv.writer(raw_f)
    raw_w.writerow(power_io.RAW_HEADER)
    raw_f.flush()

    try:
        for t, v, i in gpp4323.sample(psu, ch, t0, args.interval):
            if args.duration and t >= args.duration:
                break
            pw = v * i

            # Trapezoidal integration of power -> energy and current -> charge
            if t_list:
                energy_j += 0.5 * (pw + p_list[-1]) * (t - t_list[-1])
                charge_c += 0.5 * (i + i_list[-1]) * (t - t_list[-1])

            t_list.append(t)
            v_list.append(v)
            i_list.append(i)
            p_list.append(pw)
            e_cum.append(energy_j)
            q_cum.append(charge_c)
            raw_w.writerow(power_io.raw_row(t, v, i, pw))
            raw_f.flush()
            k = len(t_list) - 1
            if skip_idx is None and t >= args.skip:
                skip_idx = k

            print(f"{t:7.1f} | nå: {i * 1e3:7.2f} mA {pw * 1e3:9.2f} mW", end="\r")

            if t >= next_report:
                next_report += args.report_every
                i10, p10 = window_avg(t - 10, k)
                i30, p30 = window_avg(t - 30, k)
                in_window = skip_idx is not None and k > skip_idx
                k0 = skip_idx if in_window else 0
                mark = " " if in_window else "*"
                q_acc = q_cum[k] - q_cum[k0]
                e_acc = e_cum[k] - e_cum[k0]
                dt_acc = t_list[k] - t_list[k0]
                ia = q_acc / dt_acc if dt_acc > 0 else float("nan")
                pa = e_acc / dt_acc if dt_acc > 0 else float("nan")
                print(f"{t:7.1f} | {i10 * 1e3:7.2f} {p10 * 1e3:9.2f} | {i30 * 1e3:7.2f} {p30 * 1e3:9.2f} | "
                      f"{ia * 1e3:7.2f} {pa * 1e3:9.2f} {e_acc / 3.6:9.4f}{mark}")
    except KeyboardInterrupt:
        pass
    finally:
        raw_f.close()
        if args.output_off_at_end:
            psu.output(ch, False)
        psu.close()

    print()
    print(f"Lagret {raw_file}")

    # Results are computed only for t >= skip
    idx = [k for k, t in enumerate(t_list) if t >= args.skip]
    n = len(idx)
    if n < 2:
        print("For få målinger.")
        return

    ts = [t_list[k] for k in idx]
    vs = [v_list[k] for k in idx]
    is_ = [i_list[k] for k in idx]
    ps = [p_list[k] for k in idx]
    T = ts[-1] - ts[0]
    energy_win = sum(0.5 * (ps[k] + ps[k - 1]) * (ts[k] - ts[k - 1]) for k in range(1, n))
    charge_win = sum(0.5 * (is_[k] + is_[k - 1]) * (ts[k] - ts[k - 1]) for k in range(1, n))
    avg_p = energy_win / T
    avg_i = charge_win / T
    avg_v = sum(vs) / n
    print("\n--- Resultat ---")
    print(f"Analysevindu    : {ts[0]:.1f} s – {ts[-1]:.1f} s (første {args.skip:.0f} s utelatt)")
    print(f"Antall målinger : {n}  (snitt {T / (n - 1) * 1e3:.1f} ms mellom)")
    print(f"Måletid         : {T:.2f} s")
    print(f"Snitt spenning  : {avg_v:.4f} V")
    print(f"Snitt strøm     : {avg_i * 1e3:.2f} mA  (min {min(is_) * 1e3:.2f}, maks {max(is_) * 1e3:.2f})")
    print(f"Snitt effekt    : {avg_p * 1e3:.2f} mW")
    print(f"Energi          : {energy_win:.4f} J = {energy_win / 3.6:.4f} mWh")

    power_io.write_sum(sum_file, [
        ("start_time", start_time),
        ("instrument", idn),
        ("channel", ch),
        ("skip_s", args.skip),
        ("window_start_s", round(ts[0], 3)),
        ("window_end_s", round(ts[-1], 3)),
        ("duration_s", round(T, 3)),
        ("n_samples", n),
        ("mean_interval_ms", round(T / (n - 1) * 1e3, 1)),
        ("mean_voltage_V", round(avg_v, 5)),
        ("mean_current_mA", round(avg_i * 1e3, 3)),
        ("min_current_mA", round(min(is_) * 1e3, 3)),
        ("max_current_mA", round(max(is_) * 1e3, 3)),
        ("mean_power_mW", round(avg_p * 1e3, 3)),
        ("energy_J", round(energy_win, 5)),
        ("energy_mWh", round(energy_win / 3.6, 5)),
    ])
    print(f"Lagret {sum_file}")

    if args.plot:
        import power_plots
        power_plots.show_log(t_list, i_list, p_list, avg_p)


if __name__ == "__main__":
    main()
