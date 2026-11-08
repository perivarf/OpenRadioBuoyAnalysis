"""Component table for the single (non-batch) measurements in power_measurement/ (<name>_raw.csv)."""

import csv

import matplotlib.pyplot as plt
import numpy as np


# Contributions per component: (name in the table, measurement, measurement subtracted or None).
# The differences form a chain from gps_sd_imu down to the baseline, so the sum equals "Total".
COMPONENTS = [
    ("GPS", "gps_sd_imu", "sd_imu"),
    ("SD card", "sd_imu", "imu_only"),
    ("IMU", "imu_only", "busy_spin"),
    ("MCU awake", "busy_spin", "deep_sleep"),
    ("GPS idle", "deep_sleep", "deep_sleep_gps_sleep"),
    ("PC connection", "deep_sleep_gps_sleep", "deep_sleep_gps_sleep_no_pc"),
    ("Baseline (MCU, GPS deep sleep)", "deep_sleep_gps_sleep_no_pc", None),
    ("Total", "gps_sd_imu", None),
]


def write_component_tables(rows, prefix, volts):
    """Write the component table as <prefix>.csv, <prefix>.tex (LaTeX, requires booktabs) and <prefix>.png.

    Power is calculated as I * volts.
    """
    note = f"Mean ± std. dev. of raw samples. Differences: σ = √(σ$_a^2$ + σ$_b^2$). P = I · {volts:g} V."
    note_tex = (
        rf"Mean $\pm$ std.\ dev.\ of raw samples. Differences: $\sigma = \sqrt{{\sigma_a^2 + \sigma_b^2}}$. "
        rf"$P = I \cdot {volts:g}$\,V."
    )

    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["component", "mean_current_mA", "std_current_mA", "mean_power_mW", "std_power_mW"])
        for label, m, s in rows:
            w.writerow([label, f"{m:.3f}", f"{s:.3f}", f"{m * volts:.2f}", f"{s * volts:.2f}"])

    tex = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Component & Current [mA] & Power [mW] \\",
        r"\midrule",
    ]
    for label, m, s in rows:
        if label == "Total":
            tex.append(r"\midrule")
        tex.append(f"{label} & ${m:.1f} \\pm {s:.1f}$ & ${m * volts:.0f} \\pm {s * volts:.0f}$ \\\\")
    tex += [r"\bottomrule", f"\\multicolumn{{3}}{{l}}{{\\footnotesize {note_tex}}} \\\\",
            r"\end{tabular}", ""]
    with open(f"{prefix}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    # PNG in the same style as the TeX table (booktabs rules), drawn row by row
    lines = [("Component", "Current [mA]", "Power [mW]", "bold")]
    for label, m, s in rows:
        lines.append((label, f"{m:.1f} ± {s:.1f}", f"{m * volts:.0f} ± {s * volts:.0f}", "normal"))
    n = len(lines)
    # Column positions in inches; the first column scales to the longest name (~0.085 in per character)
    x_cur = 0.1 + 0.085 * max(len(line[0]) for line in lines) + 1.2
    x_pw = x_cur + 1.15
    width = x_pw + 0.08
    fig, ax = plt.subplots(figsize=(width, 0.3 * (n + 1) + 0.2))
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(n + 1.2, -0.2)  # row 0 at the top, one extra row for the note
    ax.text(0.08, n + 0.5, note, ha="left", va="center", fontsize=8, color="0.3")
    for k, (label, cur, pw, weight) in enumerate(lines):
        y = k + 0.5
        ax.text(0.08, y, label, ha="left", va="center", fontsize=10, weight=weight)
        ax.text(x_cur, y, cur, ha="right", va="center", fontsize=10, weight=weight)
        ax.text(x_pw, y, pw, ha="right", va="center", fontsize=10, weight=weight)
    rules = [(0, 1.2), (1, 0.6), (n, 1.2)]  # (rad, linjetykkelse): topp, under header, bunn
    rules += [(k, 0.6) for k, line in enumerate(lines) if line[0] == "Total"]
    for y, lw in rules:
        ax.axhline(y, c="k", lw=lw)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f"{prefix}.png", dpi=200)
    plt.close(fig)
    return [f"{prefix}.csv", f"{prefix}.tex", f"{prefix}.png"]


def component_rows(runs):
    """Mean and standard deviation of current per component, as a difference between measurements.

    The standard deviation of a difference is sqrt(s_a² + s_b²) (independent measurements).
    """
    by_name = {r["name"]: r for r in runs}
    rows = []
    for label, a, b in COMPONENTS:
        if a not in by_name or (b is not None and b not in by_name):
            print(f"{label}: missing measurement ({a}, {b}), skipping")
            continue
        ra, rb = by_name[a], by_name.get(b)
        mean = ra["mean_i"] - (rb["mean_i"] if rb else 0.0)
        std = np.hypot(ra["std_i"], rb["std_i"] if rb else 0.0)
        rows.append((label, mean, std))
    return rows
