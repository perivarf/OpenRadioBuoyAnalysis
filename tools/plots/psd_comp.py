"""Untapered PSD comparison: Madgwick and Kalman against GPS, in elevation.

    IMU:  S_ηη = S_aa / ω⁴          GPS:  S_ηη = S_vv / ω²

Per session <stamp>_psd_utapert.png (the three spectra) and <stamp>_psd_forhold.png
(their ratios), and optionally <filename>_psd_rapport.png over all sessions. The
PSDs come untapered from psd/spectra.py (same chain and cache as felles_psd).

The ratio shows the gain difference |G|² from the GPS receiver's smoothing, not
agreement - that needs coherence. Run with script_psd_comp.py.

The numbers go to data/ first and the figures are drawn from there: per session
data/<stamp>_psd_utapert.csv (f + the three PSDs; the ratio figure is drawn from the
same file) beside its figures, and data/<filename>_psd_rapport.csv (fit and Hs per
session) beside the summary. plot_only=True redraws without computing.
"""

import os

import numpy as np
import pandas as pd

import _paths                                    # noqa: E402,F401
import spectra                                   # noqa: E402
from configs import (config_dir, config_suffix,   # noqa: E402
                     data_path, read_data, write_data)
from freq_axis import log_hz_axis                # noqa: E402
from pipeline.session_input import resolve_paths  # noqa: E402
from session_source import workdir               # noqa: E402

BANDS = ((0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30),
         (0.30, 0.40), (0.40, 0.50), (0.50, 0.70), (0.70, 1.00), (1.00, 1.50))

COLORS = {"Madgwick": "C0", "Kalman": "C2", "GPS": "k"}

# Taper for the Hs column in the summary table; the PSDs are untapered.
HS_TAPER = (0.1, 0.2)
XLIM = (0.05, 2.0)


def elevation_psds(elev):
    """Untapered elevation PSDs [m²/Hz] for Madgwick, Kalman and GPS from an
    ElevationSpectra -> (f, dict). Bin 0 (ω = 0) is skipped."""
    f = np.arange(1, elev.seglen // 2 + 1) * elev.fs / elev.seglen
    out = {"Madgwick": elev.eta["m"][1:], "Kalman": elev.eta["k"][1:]}
    if elev.eta_gps is not None:
        out["GPS"] = elev.eta_gps[1:]
    return f, out


def band_table(f, psds, stamp):
    """Print band means of the PSDs, their ratios and the amplitude ratio
    √(S_imu/S_gps)."""
    print(f"\n=== {stamp}   untapered elevation PSD [m²/Hz]")
    print(f"{'band [Hz]':>12} {'Madgwick':>11} {'Kalman':>11} {'GPS':>11} "
          f"{'M/GPS':>9} {'K/GPS':>9} {'√(M/GPS)':>9}")
    for lo, hi in BANDS:
        b = (f >= lo) & (f < hi)
        if not np.any(b):
            continue
        m, k = psds["Madgwick"][b].mean(), psds["Kalman"][b].mean()
        if "GPS" not in psds:
            print(f"{lo:5.2f}-{hi:5.2f} {m:11.3e} {k:11.3e}")
            continue
        g = psds["GPS"][b].mean()
        print(f"{lo:5.2f}-{hi:5.2f} {m:11.3e} {k:11.3e} {g:11.3e} "
              f"{m / g:9.2f} {k / g:9.2f} {np.sqrt(m / g):9.2f}")


def plot_psd(f, psds, stamp, path):
    """Figure 1: the three untapered PSDs."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("Madgwick", "Kalman", "GPS"):
        if name not in psds:
            continue
        ax.loglog(f, psds[name], color=COLORS[name], lw=1.4,
                  ls=":" if name == "GPS" else "-",
                  label=name + (" vUp (ω⁻²)" if name == "GPS" else " (ω⁻⁴)"))
    top = max(float(np.max(p[(f >= XLIM[0]) & (f <= XLIM[1])])) for p in psds.values())
    ax.set_ylim(top / 1e6, top * 3.0)
    ax.set_ylabel("PSD elevation  [m²/Hz]")
    ax.set_xlabel("frequency [Hz]")
    ax.set_title(f"{stamp}   untapered elevation PSD")
    log_hz_axis(ax, XLIM)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")
    plt.close(fig)


def plot_ratio(f, psds, stamp, path, smooth=7, lag_t=0.95):
    """Figure 2: the IMU/GPS ratios, raw (faint) and smoothed over `smooth` bins.
    lag_t > 0 adds the first-order reference 1/|G|² = 1+(ωT)²."""
    import matplotlib.pyplot as plt
    if "GPS" not in psds:
        print("  (no GPS in the session - skipping the ratio plot)")
        return
    k = np.ones(smooth) / smooth
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("Madgwick", "Kalman"):
        r = psds[name] / psds["GPS"]
        ax.loglog(f, r, color=COLORS[name], lw=0.6, alpha=0.20)
        ax.loglog(f, np.convolve(r, k, "same") / np.convolve(np.ones_like(r), k, "same"),
                  color=COLORS[name], lw=1.6, label=f"{name} / GPS")
    if lag_t > 0.0:
        ax.loglog(f, 1.0 + (2.0 * np.pi * f * lag_t) ** 2, color="C3", ls="-.", lw=1.4,
                  label=f"1st-order low-pass, T={lag_t:g} s  (knee {1/(2*np.pi*lag_t):.2f} Hz)")
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0, label="equal level")
    ax.set_ylabel("PSD ratio  IMU / GPS   [-]")
    ax.set_xlabel("frequency [Hz]")
    ax.set_title(f"{stamp}   ratio (√ratio = amplitude ratio)")
    log_hz_axis(ax, XLIM)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")
    plt.close(fig)


def fit_lag(f, psds, band=(0.4, 1.5)):
    """Fit the GPS gain |G| = √(S_gps/S_imu) in the band, in log-log:
      n  from |G| ~ f^n
      T  from |G| = 1/√(1+(ωT)²)   (first-order low-pass)

    Returns (n, T, rms_log); rms_log is the T model's deviation in log units
    (0.12 ~ 12 %)."""
    b = (f >= band[0]) & (f <= band[1])
    G = np.sqrt(psds["GPS"][b] / psds["Madgwick"][b])
    fb = f[b]
    n = float(np.polyfit(np.log(fb), np.log(G), 1)[0])
    # T by a 1 ms grid scan (the model is not linear in T).
    ts = np.arange(0.05, 5.0, 0.001)
    model = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * np.outer(fb, ts)) ** 2)
    err = np.sqrt(np.mean((np.log(model) - np.log(G)[:, None]) ** 2, axis=0))
    i = int(np.argmin(err))
    return n, float(ts[i]), float(err[i])


def plot_report(sessions, path, lag_t=0.95, smooth=9):
    """Summary figure over all sessions (dicts with stamp, f, psds, fit, hs).

    Panels: (1) GPS damping |G| per session, (2) Kalman/Madgwick, (3) Kalman/GPS,
    (4) table of the fit and Hs."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 17))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.62], hspace=0.42)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax3, ax4 = fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    k = np.ones(smooth) / smooth

    def sm(y):
        return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")

    for i, s in enumerate(sessions):
        f, p = s["f"], s["psds"]
        c = f"C{i}"
        ax1.loglog(f, sm(np.sqrt(p["GPS"] / p["Madgwick"])), color=c, lw=1.4,
                   label=f"{s['stamp'][9:]}  T={s['fit'][1]:.2f} s")
        ax2.loglog(f, sm(p["Kalman"] / p["Madgwick"]), color=c, lw=1.4,
                   label=s["stamp"][9:])
        ax3.loglog(f, sm(p["Kalman"] / p["GPS"]), color=c, lw=1.4,
                   label=s["stamp"][9:])
    fr = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 400)
    ax1.loglog(fr, 1.0 / np.sqrt(1.0 + (2.0 * np.pi * fr * lag_t) ** 2), color="k",
               ls="--", lw=2.0, label=f"1st order, T={lag_t:g} s")
    ax1.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax1.set_ylim(1e-2, 3.0)
    ax1.set_ylabel("|G| = amplitude GPS / IMU   [-]")
    ax1.set_title("GPS receiver damping - the same curve above 0.5 Hz, "
                  "scattering below (low SNR)")

    ax2.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax2.set_ylim(0.25, 5.0)
    ax2.set_ylabel("PSD ratio  Kalman / Madgwick   [-]")
    ax2.set_title("The filters are identical above ~0.3 Hz and part company below it")

    # Reference 1/|G|² = 1+(ωT)²: panel 1's fit, inverted.
    ax3.loglog(fr, 1.0 + (2.0 * np.pi * fr * lag_t) ** 2, color="k", ls="--", lw=2.0,
               label=f"1st order, T={lag_t:g} s")
    ax3.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax3.set_ylim(0.1, 300.0)
    ax3.set_ylabel("PSD ratio  Kalman / GPS   [-]")
    ax3.set_title("Kalman against GPS - the rise is the GPS damping, not Kalman")
    for ax in (ax1, ax2, ax3):
        log_hz_axis(ax, XLIM)
        ax.set_xlabel("frequency [Hz]")
        ax.legend(fontsize=7.5, ncol=2)

    ax4.axis("off")
    rows = [[s["stamp"][9:], f"{s['fit'][0]:+.2f}", f"{s['fit'][1]:.2f}",
             f"{100 * s['fit'][2]:.0f} %",
             f"{s['hs'][0]:.3f}", f"{s['hs'][1]:.3f}", f"{s['hs'][2]:.3f}",
             f"{s['hs'][2] / s['hs'][0]:.3f}" if s["hs"][0] else "-"]
            for s in sessions]
    tb = ax4.table(cellText=rows,
                   colLabels=["session", "n in |G|~f^n", "T [s]", "rms",
                              "Hs Madgwick", "Hs Kalman", "Hs GPS", "Hs GPS/M"],
                   loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.6)
    ax4.set_title("Fit over 0.4-1.5 Hz, and Hs as the taper gives it "
                  f"(taper {HS_TAPER[0]:g}-{HS_TAPER[1]:g} Hz)", pad=18)
    fig.text(0.5, 0.012,
             "The ratio measures a difference in gain, NOT agreement - two channels can have "
             "identical PSDs and be uncorrelated.\nFor agreement: coherence, see "
             "selfnoise.py --compare and docs/koherens.md section 9.",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n  wrote report {path}")
    plt.close(fig)


def write_psds(f, psds, path):
    """f + the PSDs (Madgwick, Kalman, GPS when there is one) -> CSV."""
    write_data(pd.DataFrame({"f": f, **psds}), path)


def read_psds(path):
    """write_psds's CSV -> (f, {name: PSD})."""
    df = read_data(path, "script_psd_comp.py")
    return (df["f"].to_numpy(),
            {c: df[c].to_numpy() for c in df.columns if c != "f"})


REPORT_COLS = ["session", "stamp", "fit_n", "fit_t", "fit_rms", "hs_madgwick",
               "hs_kalman", "hs_gps"]


def run(dirs, config="", decimate_hz=spectra.DECIMATE_HZ, seglen=None, out_dir=None,
        lag_t=0.95, report=None, only_report=False, report_dir=".", filename="",
        plot_only=False):
    """The comparison for the session directories in dirs.

    Per session <stamp>_psd_utapert.png and <stamp>_psd_forhold.png in its work
    directory (or out_dir), unless only_report. report=None: no summary figure;
    "" -> <out_dir or report_dir/<config>>/<filename>_psd_rapport.png; a path ->
    there. lag_t [s] is the first-order reference curve (0 = none). plot_only:
    draw from the existing data/ CSVs, compute nothing."""
    if report is not None:
        report = report or os.path.join(
            out_dir or config_dir(report_dir, config),
            config_suffix(f"{filename}_psd_rapport.png", config))

    # Where each session's figures and data go: its work directory (<config>/<stamp>/).
    where = []
    for d in dirs:
        directory, stamp = workdir(d, config)
        out = out_dir or directory
        where.append((os.path.basename(os.path.normpath(d)), stamp, os.path.join(
            out, config_suffix(f"{stamp}_psd_utapert.png", config))))

    if not plot_only:
        rows = []
        for d, (session, _, png) in zip(dirs, where):
            path, st, elev, cut, sea = spectra.tapered(d, decimate_hz, HS_TAPER, config,
                                                       seglen=seglen)
            _, stamp, _ = resolve_paths(path)
            f, psds = elevation_psds(elev)
            band_table(f, psds, stamp)
            write_psds(f, psds, data_path(png, config))
            if "GPS" in psds:
                n, t, rms = fit_lag(f, psds)
                rows.append(dict(session=session, stamp=stamp, fit_n=n, fit_t=t,
                                 fit_rms=rms, hs_madgwick=sea.wp["m"]["hs"],
                                 hs_kalman=sea.wp["k"]["hs"],
                                 hs_gps=sea.gps["hs"] if sea.gps else 0.0))
        if report is not None:
            write_data(pd.DataFrame(rows, columns=REPORT_COLS), data_path(report, config))

    # The figures, from data/.
    if not only_report:
        for session, stamp, png in where:
            f, psds = read_psds(data_path(png, config))
            plot_psd(f, psds, stamp, png)
            plot_ratio(f, psds, stamp, png.replace("_psd_utapert", "_psd_forhold"),
                       lag_t=lag_t)

    if report is not None:
        table = read_data(data_path(report, config), "script_psd_comp.py")
        pngs = {session: png for session, _, png in where}
        sessions = []
        for r in table.itertuples():
            f, psds = read_psds(data_path(pngs[r.session], config))
            sessions.append(dict(stamp=r.stamp, f=f, psds=psds,
                                 fit=(r.fit_n, r.fit_t, r.fit_rms),
                                 hs=(r.hs_madgwick, r.hs_kalman, r.hs_gps)))
        if sessions:
            plot_report(sessions, report, lag_t=lag_t)
