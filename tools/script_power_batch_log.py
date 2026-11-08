#!/usr/bin/env python3
"""Logs a batch from the power board (firmware: drifter/src/power_batch.cpp) and follows it live.

Logs like script_power_log.py (same GPP-4323, same _raw.csv format), but also looks for markers
in the current and reports when a new test starts, when the capture loop starts, and when the
analysis window is in place. Stops itself when the last test in the batch is complete, and then
analyzes the log (as script_power_batch_analyze.py).

Tests with the SD card (CARD_IDS) run with the card inserted, the others without it. When the next
test needs the card swapped, the firmware waits after the marker until it is done, and the script
asks for the swap. The firmware then sends the marker again, and the test starts after it.

    # Power-cycle the channel (the board starts the batch), log until the batch is done
    python3 tools/script_power_batch_log.py --csv batch_all

    # Different voltage (default 3.7 V) and 0.5 A current limit, switch the output off at the end
    python3 tools/script_power_batch_log.py --csv batch2 --voltage 4.0 --current-limit 0.5 --output-off-at-end

    # Older batch (1 = without SD card, 2 = with; default 3 = all with card swaps)
    python3 tools/script_power_batch_log.py --csv batch2 --batch 2

    # Replay an existing log (without instrument) to test the detection
    python3 tools/script_power_batch_log.py --replay batch1_raw.csv --csv replay1
"""

import argparse
import csv
import time

import numpy as np

import _paths                                               # noqa: F401
import batch_analysis                                       # noqa: E402
import gpp4323                                              # noqa: E402
import power_io                                             # noqa: E402
from batch_live import BatchTracker                         # noqa: E402
from batch_markers import BATCHES, CARD_IDS                 # noqa: E402


def psu_source(args):
    psu = gpp4323.GPP4323(args.host, args.port)
    ch = args.channel
    print("Tilkoblet:", psu.idn())
    gpp4323.configure(psu, ch, args.voltage, args.current_limit)
    try:
        if args.no_power_cycle:
            psu.output(ch, True)
            t0 = time.perf_counter()
        else:
            print(f"Slår av CH{ch} i {args.off_time} s ...")
            t0 = gpp4323.power_cycle(psu, ch, args.off_time)   # t=0 at switch-on, so the whole batch is logged
        yield from gpp4323.sample(psu, ch, t0, args.interval)
    finally:
        if args.output_off_at_end:
            psu.output(ch, False)
        psu.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Prefix: <prefix>_raw.csv (and analysis files)")
    ap.add_argument("--host", default="192.168.2.148")
    ap.add_argument("--port", type=int, default=1026)
    ap.add_argument("--channel", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--voltage", type=float, default=3.7,
                    help="Output voltage [V] (default 3.7, nominal Li-ion voltage)")
    ap.add_argument("--current-limit", type=float, help="Set current limit [A]")
    ap.add_argument("--no-power-cycle", action="store_true",
                    help="Do not power-cycle the channel first (the batch then must be started another way)")
    ap.add_argument("--off-time", type=float, default=3.0, help="How long the output stays off at startup [s]")
    ap.add_argument("--output-off-at-end", action="store_true", help="Switch the output off after measurement")
    ap.add_argument("--interval", type=float, default=0.02, help="Desired time between measurements [s]")
    ap.add_argument("--check-every", type=float, default=1.0, help="How often to look for markers [s]")
    ap.add_argument("--max-duration", type=float, default=3600.0,
                    help="Stop after this many seconds regardless (default 3600)")
    ap.add_argument("--no-auto-stop", action="store_true", help="Do not stop when the batch is finished")
    ap.add_argument("--window", nargs=2, type=float, default=list(batch_analysis.WINDOW), metavar=("FRA", "TIL"))
    ap.add_argument("--delta", type=float, default=batch_analysis.DELTA_MA, help="Minste strømsprang for en kant [mA]")
    ap.add_argument("--batch", type=int, choices=sorted(BATCHES), default=3,
                    help="Forventet batch (standard 3 = alle tester, SD-kortet byttes underveis; "
                         "eldre: 1 = uten SD-kort, 2 = med)")
    ap.add_argument("--replay", help="Spill av en _raw.csv i stedet for å måle")
    args = ap.parse_args()

    base = power_io.strip_prefix(args.csv)
    prefix = power_io.unique_prefix(base)      # never overwrite an existing log
    if prefix != base:
        print(f"{base}_raw.csv finnes allerede - logger til {prefix}_raw.csv")
    raw_file = f"{prefix}_raw.csv"
    window = tuple(args.window)
    source = gpp4323.replay(args.replay) if args.replay else psu_source(args)

    tracker = BatchTracker(args.batch, window, args.delta, auto_stop=not args.no_auto_stop)
    print(f"Forventer batch {args.batch}: tester {BATCHES[args.batch]}")
    print(f"Start {'MED' if BATCHES[args.batch][0] in CARD_IDS else 'UTEN'} SD-kort i kortet")

    t_l, i_l = [], []
    last_check = 0.0
    print("Logger ... (Ctrl+C for å stoppe)")
    with open(raw_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(power_io.RAW_HEADER)
        try:
            for t, v, i in source:
                w.writerow(power_io.raw_row(t, v, i, v * i))
                t_l.append(t)
                i_l.append(i * 1e3)
                if not args.replay:
                    f.flush()
                    print(f"{t:7.1f} s | {i * 1e3:7.2f} mA", end="\r")

                if t >= args.max_duration:
                    print(f"\nNådde --max-duration ({args.max_duration:.0f} s)")
                    break
                if tracker.stop_at is not None and t >= tracker.stop_at:
                    print(f"\n[{t:7.1f} s] Batch {args.batch} ferdig")
                    break
                if t - last_check < args.check_every or len(t_l) < 100:
                    continue
                last_check = t
                tracker.update(t, np.asarray(t_l), np.asarray(i_l))
        except KeyboardInterrupt:
            print("\nStoppet")
        finally:
            source.close()

    print(f"Lagret {raw_file}\n")
    batch_analysis.analyze_log(raw_file, window, args.delta, out=f"{prefix}.png")


if __name__ == "__main__":
    main()
