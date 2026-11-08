#!/usr/bin/env python3
"""All sessions of a measurement series in one elevation spectrum and one
acceleration spectrum.

The PSDs are computed through psd/spectra.py (postprocess.analyse_session, cached),
every session with the same parameters. Per run:

  <filename>_eta_<method>_<f1>_<f2>.png   elevation, with this taper (Hs in legend)
  <filename>_acc_<method>.png             acceleration, untapered

in <measurement>/taper_sweep/, whatever the config: the figures sweep the taper
themselves. sweep() writes acc once and eta per taper. Run with script_felles_psd.py.

The numbers go to taper_sweep/data/ first, one column per session folder:
<filename>_acc_<method>.csv, <filename>_eta_<method>_<f1>_<f2>.csv and beside it
..._wp.csv (Hs/Tz/Tc per session). The figures are always drawn from those files,
with labels and styles from the set - plot_only=True redraws without computing.
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))      # the repo root

import _paths                                               # noqa: E402,F401
import configs                                              # noqa: E402
from spectra import (ACC_COL, ACC_TAPER_FC, DECIMATE_HZ,     # noqa: E402
                     ETA_COL, psd_frame, save_figure, set_hz_axis)

# --- Defaults for main() ----------------------------------------------------------
METHOD = "madgwick"          # madgwick | sflp | kalman | nxp | gps

# "both" | "eta" | "acc"
FIGURES = "both"

DESCRIPTION = ""             # first in the figure title (from the set)
FILENAME = ""                # prefix on the PNGs (from the set)

# Plot window [Hz]. The eta axis starts at max(taper f1, F_LO).
F_LO, F_HI = 0.01, 2

# Figure size (inches) of the _acc_ and _eta_ figures.
FIGSIZE = (10, 3.6)

# Labelled y decades every N decades; the range follows the data (y_decades).
YTICK_DECADE_STEP = 2

# Add each curve's geometric mean, 10^(mean log10 S), to its legend entry.
LEGEND_AVG = False

# Default taper (Kohout/Tucker-Pitt): weight 0 below f1, 1 above f2.
TAPER_F1, TAPER_F2 = 0.03, 0.05                 # Kohout/Tucker-Pitt

# The tapers sweep() runs; f1 == f2 is a hard cut.
TAPERS = [(0.03, 0.05),      # Kohout/Tucker-Pitt
          (0.05, 0.05),      # hard cut at Kohout's f2
          (0.10, 0.20),      # Inshore - no swell below ~10 s
          (0.03, 0.03),      # hard cut, no ramp
          (0.30, 0.30),      # hard cut, no ramp
          ]

# [[session dir, label, line style, colour], ...] relative to `directory`.
SESSIONS = []


def _positive(values):
    """The finite values > 0 (those that can go on a log axis) as floats."""
    return [float(x) for x in values if x > 0.0 and math.isfinite(x)]


def geometric_mean(values):
    """10^(mean of log10 v) over the positive values, or None if there are none."""
    v = _positive(values)
    if not v:
        return None
    return 10.0 ** (sum(math.log10(x) for x in v) / len(v))


def _mathtext_sci(v):
    """1.3e-07 -> "$1.3\\cdot10^{-7}$", matching the log axis labels."""
    e = math.floor(math.log10(abs(v)))
    m = round(v / 10.0 ** e, 1)
    # Rounding can give a mantissa of 10.0: carry it into the exponent.
    if abs(m) >= 10.0:
        m /= 10.0
        e += 1
    return rf"${m:.1f}\cdot10^{{{e}}}$"


def y_decades(values, step=YTICK_DECADE_STEP):
    """Y ticks as powers of 10 covering the data, with exponents on multiples of
    `step` (so figures with different ranges share tick values)."""
    v = _positive(values)
    if not v:
        return [10.0 ** e for e in range(-12, 1, step)]      # empty figure: show something
    lo = math.floor(math.log10(min(v)))
    hi = math.ceil(math.log10(max(v)))
    lo -= lo % step                  # down to a multiple (Python: -7 % 2 == 1)
    hi += (-hi) % step               # up to a multiple
    return [10.0 ** e for e in range(lo, hi + 1, step)]


def write_spectra(frames, col, path):
    """[(session, psd_frame)] -> one CSV: f + the column col per session."""
    wide = None
    for session, df in frames:
        part = df[["f", col]].rename(columns={col: session})
        wide = part if wide is None else wide.merge(part, on="f", how="outer")
    configs.write_data(wide.sort_values("f"), path)


def read_spectra(path, sessions, col):
    """write_spectra's CSV -> {session: DataFrame with f and col}, NaN rows dropped
    (a session that lacks a bin the others have)."""
    wide = configs.read_data(path, "script_felles_psd.py")
    out = {}
    for session in sessions:
        if session in wide:
            out[session] = (wide[["f", session]].rename(columns={session: col})
                            .dropna().reset_index(drop=True))
    return out


def write_wp(frames, method, path):
    """[(session, psd_frame)] -> CSV of Hs/Tz/Tc per session (empty when undefined)."""
    rows = []
    for session, df in frames:
        wp = df.attrs["wp"].get(method) or {}
        rows.append({"session": session, "hs": wp.get("hs"), "tz": wp.get("tz"),
                     "tc": wp.get("tc")})
    configs.write_data(pd.DataFrame(rows, columns=["session", "hs", "tz", "tc"]), path)


def read_wp(path):
    """write_wp's CSV -> {session: Hs or None}."""
    wp = configs.read_data(path, "script_felles_psd.py")
    return {r.session: (None if pd.isna(r.hs) else float(r.hs))
            for r in wp.itertuples()}


def draw_figure(series, col, ylabel, xlim, title, out_file, taper=None,
               ytick_step=YTICK_DECADE_STEP, legend_avg=LEGEND_AVG):
    """One log-log figure of computed spectra -> PNG.

    series     = [(df, legend text, style, colour)]
    col        = the column plotted (eta_m, acc_m, ...)
    xlim       = (f_start, f_hi)
    out_file   = path of the PNG
    taper      = (f1, f2) to shade, or None
    legend_avg = append the geometric mean of the drawn bins to each legend entry"""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    drawn = []                       # all plotted y values -> the y axis range
    for df, name, style, colour in series:
        if col not in df:
            # Only gps: the column exists when the session has usable GPS.
            print(f"  WARNING: {name} has no '{col}' - skipped")
            continue
        band = df[(df["f"] >= xlim[0]) & (df["f"] <= xlim[1])]

        label = name
        if legend_avg:
            g = geometric_mean(band[col])
            if g is not None:
                label = f"{name} (geo. avg = {_mathtext_sci(g)})"

        ax.plot(band["f"], band[col], style, color=colour, label=label)
        drawn.extend(band[col].tolist())

    # The taper band, behind the curves (before the axis setup).
    if taper and taper[1] > taper[0]:
        ax.axvspan(taper[0], taper[1], color="yellow", alpha=0.2, zorder=0)

    set_hz_axis(ax, xlim)

    yticks = y_decades(drawn, ytick_step)
    ax.set_yscale("log")
    ax.set_yticks(yticks)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    # No unlabelled minor decades.
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    print(f"  y axis: 10^{round(math.log10(yticks[0]))} .. "
          f"10^{round(math.log10(yticks[-1]))}")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    save_figure(fig, out_file)


def main(method=METHOD, f_lo=F_LO, f_hi=F_HI, decimate_hz=DECIMATE_HZ,
         taper=(TAPER_F1, TAPER_F2), sessions=SESSIONS, filename=FILENAME,
         description=DESCRIPTION, figures=FIGURES, directory=_ROOT,
         legend_avg=LEGEND_AVG, config="", plot_only=False):
    """Compute the spectra, write them to data/ and draw the figures from there.
    Returns the elevation series.

    sessions    = [[session dir, label, line style, colour], ...] (required)
    directory   = the measurement directory the session names resolve against
    figures     = "both" | "eta" | "acc"
    taper       = (f1, f2) for the elevation; f1 == f2 is a hard cut
    filename    = prefix: <prefix>_eta_<method>_<f1>_<f2>.png, <prefix>_acc_<method>.png
    description = first in the figure title
    legend_avg  = geometric mean in the legend
    config      = run name: the session work directories (<directory>/<config>/)
    plot_only   = draw from the existing data/ CSVs, compute nothing

        s = session_sets.get_set("SKJAERHALDEN")
        felles_psd.sweep([(0.03, 0.05), (0.10, 0.20)], sessions=s.sessions,
                         directory=s.directory)"""
    f1, f2 = taper
    config = configs.check_config(config)
    if not sessions:
        sys.exit("no sessions given - main() needs sessions=[[session dir, "
                 "label, style, colour], ...], e.g. from session_sets.get_set(<set>).")
    if method not in ETA_COL:
        sys.exit(f"unknown method '{method}' - know {', '.join(ETA_COL)}")
    if figures not in ("both", "eta", "acc"):
        sys.exit(f"unknown figures='{figures}' - know both, eta, acc")

    draw_eta = figures in ("both", "eta")
    # GPS has no acceleration column.
    draw_acc = figures in ("both", "acc") and method in ACC_COL
    if figures == "acc" and not draw_acc:
        sys.exit(f"method '{method}' has no acceleration column - "
                 f"figures='acc' would not write any figure")
    if not 0.0 < f1 <= f2:
        sys.exit(f"the taper needs 0 < f1 <= f2 (got {f1:g} and {f2:g})")

    def out_file(name):
        return os.path.join(configs.sweep_dir(directory), name)

    out_eta = out_file(f"{filename}_eta_{method}_{f1:g}_{f2:g}.png")
    out_acc = out_file(f"{filename}_acc_{method}.png")
    csv_eta, csv_acc = configs.data_path(out_eta), configs.data_path(out_acc)
    csv_wp = csv_eta[:-len(".csv")] + "_wp.csv"

    # Elevation with the taper, acceleration untapered (same cached PSD).
    if not plot_only:
        frames, frames_acc = [], []
        for session, label, style, colour in sessions:
            sess_dir = os.path.join(directory, session)
            if not os.path.isdir(sess_dir):
                sys.exit(f"did not find the session directory {sess_dir}")
            print(f"--- {session} ({label}) ---", flush=True)
            if draw_eta:
                df = psd_frame(sess_dir, decimate_hz, taper, config)
                wp = df.attrs["wp"].get(method)
                print(f"  Hs = {'-' if wp is None else format(wp['hs'], '.3f') + ' m'}")
                frames.append((session, df))
            if draw_acc:
                frames_acc.append((session, psd_frame(
                    sess_dir, decimate_hz, (ACC_TAPER_FC, ACC_TAPER_FC), config)))
        if draw_eta:
            write_spectra(frames, ETA_COL[method], csv_eta)
            write_wp(frames, method, csv_wp)
        if draw_acc:
            write_spectra(frames_acc, ACC_COL[method], csv_acc)

    # The figures, from data/ - labels and styles from the set.
    names = [row[0] for row in sessions]
    series, series_acc = [], []
    if draw_eta:
        eta, hs = read_spectra(csv_eta, names, ETA_COL[method]), read_wp(csv_wp)
        for session, label, style, colour in sessions:
            if session in eta:
                # Hs (taper dependent) only in the elevation legend.
                h = hs.get(session)
                name = label if h is None else f"{label} - {h:.3f} m"
                series.append((eta[session], name, style, colour))
    if draw_acc:
        acc = read_spectra(csv_acc, names, ACC_COL[method])
        series_acc = [(acc[session], label, style, colour)
                      for session, label, style, colour in sessions if session in acc]

    # Elevation from max(f1, f_lo); acceleration from the lowest bin there is.
    xlim = (max(f1, f_lo), f_hi)
    xlim_acc = ((min(float(d["f"].min()) for d, *_ in series_acc), f_hi)
                if draw_acc else None)

    # f1 == f2 is a hard cut at f_c, not a ramp.
    band = f"$f_c$ = {f1:g} Hz" if f1 == f2 else f"taper {f1:g}-{f2:g} Hz"
    title = (f"{description} - {method.capitalize()}, {band}, "
              f"$f_s$ = {decimate_hz:g} Hz")
    title_acc = (f"{description} - {method.capitalize()}, "
                  f"$f_s$ = {decimate_hz:g} Hz")

    print()
    if draw_eta and f1 == f2:
        print(f"  f_c = {f1:g} Hz - hard cut with no ramp, "
              f"no marking to draw")

    # m²/Hz is a variance density.
    if draw_eta:
        draw_figure(series, ETA_COL[method], "PSD elevation  [m²/Hz]",
                   xlim, title, out_eta, taper=taper, legend_avg=legend_avg)
    if draw_acc:
        draw_figure(series_acc, ACC_COL[method], "PSD acceleration  [(m/s²)²/Hz]",
                   xlim_acc, title_acc, out_acc, taper=None,
                   legend_avg=legend_avg)
    return series


def sweep(tapers=TAPERS, **kw):
    """The acc figure once, then the eta figure per taper. kw goes to main()."""
    main(figures="acc", **kw)
    for t in tapers:
        main(taper=tuple(t), figures="eta", **kw)
