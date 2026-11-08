"""The batch tests as components: labels, the component differences and the component table.

Each component is a signed sum of tests (COMPONENTS), e.g. IMU = test 4 - test 8. window_runs()
gives the current of each test over the analysis window, component_rows() the components, and
write_component_tables() the table (CSV, TeX and PNG) used in the thesis.
"""

import csv
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np

import batch_analysis
import power_io
from power_stats import mean_std

LABELS = {
    1: "Deep-sleep floor",
    2: "Deep-sleep MCU, GNSS standby",
    3: "Busy spin, GNSS deep sleep",
    4: "IMU capture, GNSS standby",
    5: "IMU capture, GNSS tracking",
    6: "IMU capture, GNSS, SD card idle",
    7: "IMU capture, GNSS, SD logging",
    8: "Busy spin, GNSS standby",
    9: "Deep-sleep floor, SD card idle",
    10: "IMU capture, GNSS standby, SD card idle",
}

# Contributions per component: (name in the table, {test: sign}). The terms sum to test 7,
# so the sum of the components equals "Total". The SD card idle current is taken between two
# quiet IMU captures (10 - 4), and GNSS tracking with the card inserted in both (6 - 10).
# Without test 10 (older log), both are taken via test 5, with GNSS noise in the SD-card term.
# Without test 8, IMU and GNSS standby are taken via test 2: then standby - deep sleep is assumed
# to equal the MCU in Stop2 and awake.
COMPONENTS = [
    ("SD card logging", {7: 1, 6: -1}),
    ("SD card inserted, idle", {10: 1, 4: -1}),
    ("GNSS tracking (10 Hz)", {6: 1, 10: -1}),
    ("IMU", {4: 1, 8: -1}),
    ("GNSS standby vs. deep sleep", {8: 1, 3: -1}),
    ("MCU awake (48 MHz)", {3: 1, 1: -1}),
    ("Deep-sleep floor*", {1: 1}),
    ("Total", {7: 1}),
]
COMPONENTS_NO_8 = {
    "IMU": {4: 1, 3: -1, 2: -1, 1: 1},
    "GNSS standby vs. deep sleep": {2: 1, 1: -1},
}
COMPONENTS_NO_10 = {
    "SD card inserted, idle": {6: 1, 5: -1},
    "GNSS tracking (10 Hz)": {5: 1, 4: -1},
}

# Names in the TeX table (acro/siunitx as in the report) and typical datasheet values [mA]:
# GNSS in continuous tracking (u-blox MAX-M8), IMU with accelerometer and gyro in high-performance mode
# (LSM6DSV), MCU in run mode at 48 MHz (STM32WLE5).
TEX_LABELS = {
    "SD card logging": "SD card logging",
    "SD card inserted, idle": "SD card inserted, idle",
    "GNSS tracking (10 Hz)": r"\acs{GNSS} tracking (\SI{10}{\hertz})",
    "IMU": r"\acs{IMU}",
    "GNSS standby vs. deep sleep": r"\acs{GNSS} standby vs.\ deep sleep",
    "MCU awake (48 MHz)": r"\acs{MCU} awake (\SI{48}{\mega\hertz})",
    "Deep-sleep floor*": r"Deep-sleep floor$^{*}$",
    "Total": "Total",
}
DATASHEET = {"GNSS tracking (10 Hz)": "23", "IMU": "0.65", "MCU awake (48 MHz)": "5.55"}
# Explanation for the asterisk on Deep-sleep floor
# I2C is released (Wire.end() in backupGPS) and SPI is parked high, so the pull-up resistors do not draw current
FLOOR_NOTE = ("MCU in Stop2, GNSS in software backup, radio and IMU asleep, I2C and SPI lines released or "
              "parked high. The rest is drawn by the board: voltage regulator, bus buffer and LEDs.")
FLOOR_NOTE_TEX = (r"\acs{MCU} in Stop2, \acs{GNSS} in software backup, radio and \acs{IMU} asleep, \acs{I2C} "
                  r"and \acs{SPI} lines released or parked high. The rest is drawn by the board: voltage "
                  r"regulator, bus buffer and LEDs.")


def write_component_tables(rows, prefix, volts, span, sources=None, marks=None, note_extra=("", "")):
    """The component table as <prefix>.csv, <prefix>.tex (booktabs, siunitx, acro) and <prefix>.png.

    Optional: sources = {name: "K"/"GW"} adds a Source column, marks = {name: (symbol, tex-symbol)}
    adds a footnote mark after the name, and note_extra = (text, tex) is appended to the explanation.
    """
    sources = sources or {}
    marks = marks or {}
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "mean_current_mA", "std_current_mA", "mean_power_mW", "std_power_mW",
                    "datasheet_mA"] + (["source"] if sources else []))
        for label, m, s in rows:
            w.writerow([label, f"{m:.3f}", f"{s:.3f}", f"{m * volts:.2f}", f"{s * volts:.2f}",
                        DATASHEET.get(label, "")] + ([sources.get(label, "")] if sources else []))

    note_tex = (rf"Mean $\pm$ standard deviation of the raw samples over \SI{{{span:g}}}{{\second}} per "
                rf"configuration. Differences: $\sigma = \sqrt{{\sum \sigma_i^2}}$."
                rf"{note_extra[1]}\newline $^{{*}}${FLOOR_NOTE_TEX}")
    cols, src_head = ("l r r r", r" & \textbf{Source}") if sources else ("l r r", "")
    tex = [
        rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
        r"  \toprule",
        r"  \textbf{Component} & \textbf{Current (\si{\milli\ampere})}"
        rf" & \textbf{{Datasheet (\si{{\milli\ampere}})}}{src_head} \\",
        r"  \midrule",
    ]
    for label, m, s in rows:
        if label == "Total":
            tex.append(r"  \midrule")
            ds = ""
        else:
            ds = DATASHEET.get(label, "--")
        name = TEX_LABELS.get(label, label) + (marks[label][1] if label in marks else "")
        src = f" & {sources.get(label, '').replace('Δ', r'$\Delta$')}" if sources else ""
        tex.append(f"  {name} & ${m:.1f} \\pm {s:.1f}$ & {ds}{src} \\\\")
    ncol = 4 if sources else 3
    tex += [r"  \bottomrule",
            rf"  \multicolumn{{{ncol}}}{{@{{}}p{{0.95\linewidth}}@{{}}}}{{\footnotesize {note_tex}}}",
            r"\end{tabular}", ""]
    with open(f"{prefix}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    # PNG in the same style as the TeX table (booktabs rules), drawn row by row
    note = (f"Mean ± std. dev. of raw samples over {span:g} s per configuration. "
            f"Differences: σ = √(Σσ$_i^2$).{note_extra[0]}\n*{FLOOR_NOTE}")
    lines = [("Component", "Current [mA]", "Datasheet [mA]", "Source" if sources else "", "bold")]
    for label, m, s in rows:
        ds = "" if label == "Total" else DATASHEET.get(label, "–")
        name = label + (marks[label][0] if label in marks else "")
        lines.append((name, f"{m:.1f} ± {s:.1f}", ds, sources.get(label, ""), "normal"))
    n = len(lines)
    x_cur = 0.1 + 0.085 * max(len(line[0]) for line in lines) + 1.2
    x_ds = x_cur + 1.45
    x_src = x_ds + (0.75 if sources else 0.0)
    width = x_src + 0.08
    # Wrap the note to the table width (~17 characters per inch at 8 pt)
    note = "\n".join(textwrap.fill(par, int(width * 17)) for par in note.split("\n"))
    extra = 0.5 * note.count("\n") + 1.0
    fig, ax = plt.subplots(figsize=(width, 0.3 * (n + extra) + 0.2))
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(n + extra + 0.2, -0.2)  # row 0 at the top, note below
    ax.text(0.08, n + 0.3, note, ha="left", va="top", fontsize=8, color="0.3")
    for k, (label, cur, ds, src, weight) in enumerate(lines):
        y = k + 0.5
        ax.text(0.08, y, label, ha="left", va="center", fontsize=10, weight=weight)
        for x, txt in ((x_cur, cur), (x_ds, ds), (x_src, src)):
            ax.text(x, y, txt, ha="right", va="center", fontsize=10, weight=weight)
    rules = [(0, 1.2), (1, 0.6), (n, 1.2)]  # (rad, linjetykkelse): topp, under header, bunn
    rules += [(k, 0.6) for k, line in enumerate(lines) if line[0] == "Total"]
    for y, lw in rules:
        ax.axhline(y, c="k", lw=lw)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f"{prefix}.png", dpi=200)
    plt.close(fig)
    return [f"{prefix}.csv", f"{prefix}.tex", f"{prefix}.png"]


def window_runs(raw, out_dir, window=batch_analysis.WINDOW, capture_s=batch_analysis.CAPTURE_S):
    """Analyze a log and return {id: run} for the valid tests, with the current over the window
    (t from the window start, i_mA, mean_i, std_i)."""
    name = os.path.basename(raw)[: -len("_raw.csv")]
    print(f"--- {raw}")
    rows = batch_analysis.analyze_log(raw, window, capture_s=capture_s,
                                      out=os.path.join(out_dir, f"{name}_analyse.png"))
    t, _, i, _ = power_io.read_raw(raw)
    runs = {}
    for row in rows:
        tid = row["id"]
        if not row["valid"]:
            print(f"Test {tid} in {name} is invalid ({row['problem']}) - skipped")
            continue
        a, b = row["window_start_s"], row["window_end_s"]
        m = (t >= a) & (t <= b)
        if m.sum() < 2 or t[m][-1] - t[m][0] < 0.9 * (b - a):
            print(f"Test {tid} in {name}: window {a:.0f}-{b:.0f} s was not fully logged - skipped")
            continue
        tt, ii = t[m] - a, i[m] * 1e3
        mean, std = mean_std(tt, ii)
        runs[tid] = dict(id=tid, src=name, start=a, t=tt, i_mA=ii, mean_i=mean, std_i=std,
                         estimated=bool(row["capture_estimated"]))
    return runs


def component_terms(label, runs, verbose=False):
    """The tests in a component ({test: sign}), with the fallbacks when test 8 or 10 is missing."""
    terms = dict(COMPONENTS)[label]
    if 8 not in runs and label in COMPONENTS_NO_8:
        terms = COMPONENTS_NO_8[label]
        if verbose:
            print(f"{label}: test 8 is missing, treating as {terms} "
                  f"(assuming GNSS standby - deep sleep independent of MCU)")
    if 10 not in runs and label in COMPONENTS_NO_10:
        terms = COMPONENTS_NO_10[label]
        if verbose:
            print(f"{label}: test 10 is missing, treating as {terms}")
    return terms


def component_rows(runs):
    """(label, mean, std) per component; std of a sum is sqrt(sum of std²) (independent tests)."""
    rows = []
    for label, _ in COMPONENTS:
        terms = component_terms(label, runs, verbose=True)
        missing = [k for k in terms if k not in runs]
        if missing:
            print(f"{label}: mangler test {missing}, hopper over")
            continue
        mean = sum(s * runs[k]["mean_i"] for k, s in terms.items())
        std = np.sqrt(sum(runs[k]["std_i"] ** 2 for k in terms))
        rows.append((label, mean, std))
    return rows


def current_curves(runs, label=None):
    """The runs as curves for power_plots.plot_currents: lowest mean first, color per test ID."""
    label = label or (lambda r: f"{r['id']}: {LABELS[r['id']]} ({r['mean_i']:.1f} mA)")
    return [(r["t"], r["i_mA"], r["mean_i"], label(r), f"C{(r['id'] - 1) % 10}")
            for r in sorted(runs.values(), key=lambda r: r["mean_i"])]


def write_batch_tests(path, runs, columns, volts):
    """batch_tests.csv: id, label, the given extra columns {header: fn(run)}, mean/std current and power."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + list(columns) + ["mean_current_mA", "std_current_mA", "mean_power_mW"])
        for tid in sorted(runs):
            r = runs[tid]
            w.writerow([tid, LABELS[tid]] + [fn(r) for fn in columns.values()]
                       + [f"{r['mean_i']:.3f}", f"{r['std_i']:.3f}", f"{r['mean_i'] * volts:.2f}"])
    print(f"Lagret {path}")


def read_keithley(path):
    """The Keithley 2000 readings: id (test in the log, or empty), keithley_mA, gw_mA (optional), note."""
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["keithley_mA"].strip()]


def keithley_fit(readings, runs):
    """Linear fit K = a * GW + b. GW is gw_mA if given, else the window mean of test id in runs.

    Returns (a, b, gw, k): the coefficients and the points as arrays.
    """
    gw = np.array([float(r["gw_mA"]) if r["gw_mA"].strip() else runs[int(r["id"])]["mean_i"] for r in readings])
    k = np.array([float(r["keithley_mA"]) for r in readings])
    a, b = np.polyfit(gw, k, 1)
    return a, b, gw, k
