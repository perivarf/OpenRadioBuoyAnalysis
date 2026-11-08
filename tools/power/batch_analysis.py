"""Splits a batch log into tests from the markers (batch_markers.py) and measures each one.

The analysis window is measured from the capture start for the wave tests (WAVE_IDS),
otherwise from the end of the ID marker. In addition, the full segment (marker end to the
next marker) is measured, i.e. the total cost including GPS waiting etc.

analyze_log() does it all for one <prefix>_raw.csv and writes <prefix>_tests.csv.
"""

import csv
import os

import numpy as np

import power_io
import power_plots
from batch_markers import (NAMES, WAVE_IDS, capture_problem, detect, drop_repeated, match_batch,
                           pick_capture, BATCHES)
from power_stats import window_stats

WINDOW = (30.0, 90.0)   # analysis window [s] after capture start / marker end
DELTA_MA = 3.0          # smallest current step for an edge [mA]
CAPTURE_S = 120.0       # capture length in firmware (power_batch_test_ms) [s]
MIN_SCORE = 1.3         # lowest score for a capture start that is not strict
CAPTURE_TAIL_S = 7.0    # capture end to next marker [s] when no strict capture start gives it
EDGE_SIDE_S = 0.3       # median window on each side of an edge [s]

WINDOW_KEYS = ("mean_current_mA", "std_current_mA", "min_current_mA", "max_current_mA",
               "mean_voltage_V", "mean_power_mW", "energy_J")


def analyze(t, v, i, p, markers, caps, window=WINDOW, capture_s=CAPTURE_S, min_score=MIN_SCORE,
            capture_tail=CAPTURE_TAIL_S):
    """One row per marker (as in <prefix>_tests.csv), plus the private keys _anchor and _seg_end."""
    w0, w1 = window

    def segment(k):
        seg_end = markers[k + 1]["group_start"] if k + 1 < len(markers) else t[-1]
        return markers[k]["test_start"], seg_end, k + 1 < len(markers)

    # Time from capture end to the next marker (end position, processing), measured on the
    # strict capture starts. With GPS tracking the capture marker's ~5 mA drowns in the
    # GPS current, and the capture start is then estimated backwards from the next marker.
    tails = []
    for k, m in enumerate(markers):
        seg_start, seg_end, closed = segment(k)
        c = pick_capture(caps, seg_start, seg_end, full=True)
        if m["id"] in WAVE_IDS and closed and c and capture_problem(c, seg_end, capture_s, min_score) is None:
            tails.append(seg_end - c["t"] - capture_s)
    tail = float(np.median(tails)) if tails else capture_tail

    rows = []
    for k, m in enumerate(markers):
        seg_start, seg_end, closed = segment(k)
        cap_c = pick_capture(caps, seg_start, seg_end, full=True) if m["id"] in WAVE_IDS else None
        problem = capture_problem(cap_c, seg_end, capture_s, min_score) if m["id"] in WAVE_IDS else None
        estimated = False
        if problem and closed and seg_end - seg_start >= capture_s + tail:
            print(f"Måling {m['id']}: {problem}; capture-start anslått fra neste markør "
                  f"({capture_s:g} s capture + {tail:.1f} s etter)")
            cap_c = {"t": seg_end - capture_s - tail, "strict": False, "score": float("nan")}
            problem, estimated = None, True
        cap = cap_c["t"] if cap_c else None
        anchor = cap if (cap is not None and m["id"] in WAVE_IDS) else seg_start
        if problem:
            print(f"UGYLDIG: måling {m['id']}: {problem}")

        win = window_stats(t, v, i, p, anchor + w0, anchor + w1)
        seg = window_stats(t, v, i, p, seg_start, seg_end)
        if anchor + w1 > seg_end:
            print(f"ADVARSEL: vinduet i måling {m['id']} går forbi segmentets slutt")

        row = {
            "id": m["id"],
            "name": NAMES.get(m["id"], f"test{m['id']}"),
            "test_start_s": round(seg_start, 2),
            "capture_start_s": round(cap, 2) if cap is not None else "",
            "window_start_s": round(anchor + w0, 2),
            "window_end_s": round(anchor + w1, 2),
        }
        for key in WINDOW_KEYS:
            row[key] = round(win[key], 4) if win else ""
        row["segment_end_s"] = round(seg_end, 2)
        row["segment_duration_s"] = round(seg["duration_s"], 2) if seg else ""
        row["segment_mean_current_mA"] = round(seg["mean_current_mA"], 3) if seg else ""
        row["segment_energy_J"] = round(seg["energy_J"], 4) if seg else ""
        row["segment_open_end"] = int(k + 1 == len(markers))
        row["capture_estimated"] = int(estimated)
        row["valid"] = int(problem is None)
        row["problem"] = problem or ""
        row["_anchor"], row["_seg_end"] = anchor, seg_end
        rows.append(row)
    return rows


def write_split(prefix, t, v, i, p, rows):
    """<prefix>_<id>_<name>_raw.csv per test, from 10 s before the window anchor (t = 0 there)."""
    for row in rows:
        anchor = row["_anchor"]
        ms = (t >= anchor - 10) & (t <= row["_seg_end"])
        name = f"{prefix}_{row['id']}_{row['name']}_raw.csv"
        power_io.write_raw(name, t[ms] - anchor, v[ms], i[ms], p[ms])
        print(f"Lagret {name}")


def write_tests_csv(path, rows):
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def print_tests(rows, window):
    w0, w1 = window
    print()
    print(f"{'ID':>3} {'navn':<20} {'start':>8} {'capture':>8} | {'vindu ' + f'{w0:g}-{w1:g} s':^30} | "
          f"{'hele segmentet':^24}")
    print(f"{'':>3} {'':<20} {'[s]':>8} {'[s]':>8} | {'mA':>8} {'±mA':>6} {'mW':>8} {'J':>6} | "
          f"{'s':>7} {'mA':>7} {'J':>8}")
    print("-" * 100)
    for r in rows:
        cap = (f"{'~' if r['capture_estimated'] else ''}{r['capture_start_s']:.1f}".rjust(8)
               if r["capture_start_s"] != "" else f"{'-':>8}")
        win = (f"{r['mean_current_mA']:8.2f} {r['std_current_mA']:6.2f} {r['mean_power_mW']:8.2f} "
               f"{r['energy_J']:6.2f}") if r["mean_current_mA"] != "" else f"{'(ingen data)':>31}"
        seg = (f"{r['segment_duration_s']:7.1f} {r['segment_mean_current_mA']:7.2f} {r['segment_energy_J']:8.2f}"
               if r["segment_duration_s"] != "" else f"{'(ingen data)':>24}")
        mark = ("*" if r["segment_open_end"] else " ") + ("  UGYLDIG" if not r["valid"] else "")
        print(f"{r['id']:>3} {r['name']:<20} {r['test_start_s']:8.1f} {cap} | {win} | {seg}{mark}")
    print("* segmentet går til slutten av loggen (ingen markør etter)")
    if any(r["capture_estimated"] for r in rows):
        print("~ capture-start anslått bakover fra neste markør (capture-markøren ble ikke sett)")


def analyze_log(raw, window=WINDOW, delta=DELTA_MA, capture_s=CAPTURE_S, min_score=MIN_SCORE,
                capture_tail=CAPTURE_TAIL_S, edge_side=EDGE_SIDE_S, split=False, plot=False, out=None):
    """Analyze <prefix>_raw.csv: table to the console, <prefix>_tests.csv, optionally split files and
    a plot (shown with plot=True, saved to out). Returns the rows ([] if no markers were found)."""
    prefix = raw[:-len("_raw.csv")] if raw.endswith("_raw.csv") else os.path.splitext(raw)[0]
    t, v, i, p = power_io.read_raw(raw)
    i_ma = i * 1e3

    markers, caps = detect(t, i_ma, delta, edge_side)
    if not markers:
        print("Fant ingen ID-markører. Prøv lavere --delta, eller sjekk loggen med --plot.")
        return []

    markers = drop_repeated(markers)
    ids = [m["id"] for m in markers]
    batch = match_batch(ids)
    if batch is None:
        print(f"ADVARSEL: ID-rekkefølgen {ids} matcher ingen batch {BATCHES}")
    else:
        print(f"Batch {batch}: ID-er {ids}")

    rows = analyze(t, v, i, p, markers, caps, window, capture_s, min_score, capture_tail)
    if split:
        write_split(prefix, t, v, i, p, rows)
    print_tests(rows, window)

    out_csv = f"{prefix}_tests.csv"
    write_tests_csv(out_csv, rows)
    print(f"\nLagret {out_csv}")

    if plot or out:
        power_plots.plot_batch_log(t, i_ma, rows, markers, os.path.basename(raw), out)
    return rows
