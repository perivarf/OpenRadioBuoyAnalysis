#!/usr/bin/env python3
"""Significant wave height as a function of the low-frequency cutoff.

One curve per session: x is the cutoff f_c, y is Hs with a hard cut at f_c
(taper f1 == f2). The x values are the analysis' own frequency bins, taken from the
first session. Every point reuses the cached PSD (psd/spectra.py), so only the
taper is recomputed.

The PNG <filename>_hs_vs_fc_<method>.png goes in <measurement>/taper_sweep/,
whatever the config (f_c is its x axis). Run with script_hs_cutoff.py.

The numbers go to taper_sweep/data/<filename>_hs_vs_fc_<method>.csv first (f_c + one
column per session folder) and the figure is drawn from there, with labels and
styles from the set - plot_only=True redraws without computing.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))      # the repo root

import _paths                                                # noqa: E402,F401
import configs                                               # noqa: E402
import spectra                                               # noqa: E402

# --- Defaults -----------------------------------------------------------------
METHOD = "madgwick"          # madgwick | sflp | kalman | nxp | gps

# Same rate as felles_psd, so the figures can be read together.
DECIMATE_HZ = spectra.DECIMATE_HZ

# Sweep range [Hz]; FC_LO = None starts at the lowest bin.
FC_LO, FC_HI = None, 0.3

# Vertical reference lines (f_c, label); [] for none.
REFERENCES = []

DESCRIPTION = ""             # first in the figure title (from the set)
FILENAME = ""                # prefix on the PNG (from the set)

# [[session dir, label, line style, colour], ...] relative to `directory`.
SESSIONS = []


def bin_frequencies(sess_dir, decimate_hz, fc_lo, fc_hi, config=""):
    """The analysis' frequency bins in [fc_lo, fc_hi] -> list [Hz] (fc_lo = None:
    from the lowest bin). Taken from the untapered frame, which has every bin."""
    df = spectra.psd_frame(sess_dir, decimate_hz,
                           (spectra.ACC_TAPER_FC, spectra.ACC_TAPER_FC), config)
    return [f for f in df["f"].tolist()
            if f <= fc_hi and (fc_lo is None or f >= fc_lo)]


def hs_curve(sess_dir, fcs, decimate_hz=DECIMATE_HZ, method=METHOD, config=""):
    """Hs [m] for every f_c in fcs; NaN where the cut leaves no energy."""
    out = []
    for fc in fcs:
        # f1 == f2: hard cut.
        wp = spectra.psd_frame(sess_dir, decimate_hz, (fc, fc),
                               config).attrs["wp"].get(method)
        hs = None if wp is None else wp["hs"]
        out.append(float("nan") if hs is None or hs < 0.0 else hs)
    return out


def main(method=METHOD, decimate_hz=DECIMATE_HZ, fc_lo=FC_LO, fc_hi=FC_HI,
         sessions=SESSIONS, filename=FILENAME, description=DESCRIPTION,
         references=REFERENCES, directory=_ROOT, config="", plot_only=False):
    """Compute Hs(f_c) for every session, write it to data/ and draw the figure from
    there. Returns (fcs, series).

    sessions    = [[session dir, label, line style, colour], ...] (required)
    directory   = the measurement directory the session names resolve against
    references  = [(f_c, label), ...] vertical lines, or []
    config      = run name: the session work directories (<directory>/<config>/)
    plot_only   = draw from the existing data/ CSV, compute nothing"""
    config = configs.check_config(config)
    if not sessions:
        sys.exit("no sessions given - main() needs sessions=[[session dir, "
                 "label, style, colour], ...], e.g. from "
                 "session_sets.get_set(<set>).")
    if method not in spectra.ETA_COL:
        sys.exit(f"unknown method '{method}' - known: {', '.join(spectra.ETA_COL)}")
    if fc_lo is not None and not 0.0 < fc_lo < fc_hi:
        sys.exit(f"the sweep needs 0 < fc_lo < fc_hi (got {fc_lo:g} and {fc_hi:g})")

    out = os.path.join(configs.sweep_dir(directory), f"{filename}_hs_vs_fc_{method}.png")
    csv = configs.data_path(out)
    if not plot_only:
        # All sessions share rate and segment length, so they share the bins.
        first = os.path.join(directory, sessions[0][0])
        if not os.path.isdir(first):
            sys.exit(f"could not find the session directory {first}")
        print(f"--- bin series from {sessions[0][0]} ---", flush=True)
        fcs = bin_frequencies(first, decimate_hz, fc_lo, fc_hi, config)
        if not fcs:
            sys.exit(f"no bins in {fc_lo:g}-{fc_hi:g} Hz - too narrow a sweep?")
        print(f"  {len(fcs)} bins from {fcs[0]:.4f} to {fcs[-1]:.4f} Hz "
              f"(df = {fcs[1] - fcs[0]:.4f} Hz)" if len(fcs) > 1 else "")

        data = {}
        for session, label, style, colour in sessions:
            sess_dir = os.path.join(directory, session)
            if not os.path.isdir(sess_dir):
                sys.exit(f"could not find the session directory {sess_dir}")
            print(f"--- {session} ({label}) ---", flush=True)
            hs = hs_curve(sess_dir, fcs, decimate_hz, method, config)
            valid = [v for v in hs if v == v]          # NaN != NaN
            print(f"  Hs: {min(valid):.3f} - {max(valid):.3f} m over the sweep"
                  if valid else "  Hs: undefined across the whole sweep")
            data[session] = hs

        configs.write_data(pd.DataFrame({"fc": fcs, **data}), csv)

    # The figure, from data/ - labels and styles from the set.
    table = configs.read_data(csv, "script_hs_cutoff.py")
    fcs = table["fc"].tolist()
    series = [(table[session].tolist(), label, style, colour)
              for session, label, style, colour in sessions if session in table]

    fig, ax = plt.subplots(figsize=(10, 4))

    xlim = (fcs[0], fc_hi)

    # Reference lines behind the curves.
    for fc, text in references or []:
        if xlim[0] <= fc <= xlim[1]:
            ax.axvline(fc, color="0.6", linestyle=":", linewidth=1.2, zorder=0)
            ax.annotate(text, xy=(fc, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -12), textcoords="offset points",
                        color="0.4", fontsize=9)

    for hs, label, style, colour in series:
        ax.plot(fcs, hs, style, color=colour, label=label)

    # Log x (same axis as felles_psd) and log y: Hs spans more than a decade over
    # the sweep, and the log scale keeps the low-cutoff end readable in less height.
    spectra.set_hz_axis(ax, xlim)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda v, _p: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))   # 0.2, 0.5, 2, ...
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", axis="y", alpha=0.12)

    ax.set_xlabel("Cutoff frequency  $f_c$  [Hz]")
    ax.set_ylabel("$H_s$  [m]")
    ax.set_title(f"{description} - {method.capitalize()}, "
                 f"$f_s$ = {decimate_hz:g} Hz")
    ax.legend()

    print()
    spectra.save_figure(fig, out)
    return fcs, series
