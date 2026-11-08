"""Figures of the power measurements."""

import os

import matplotlib.pyplot as plt
import numpy as np

import power_io
from power_stats import boxcar, integrate


def plot_currents(curves, out, volts, smooth_ms, xlabel="Time [s]", xlim=None, title_extra="", title_size=None):
    """Current over time, one curve per measurement, with the mean as a dotted line.

    curves: (t, i_mA, mean_mA, label, color) in the order they are drawn; the legend is reversed so the
    last one (highest current) is on top. Right axis: power at volts, P = I * V.
    """
    fig, ax1 = plt.subplots(figsize=(12, 4.5), layout="constrained")
    width_s = smooth_ms / 1000.0
    for t, i_mA, mean, label, c in curves:
        y = boxcar(t, i_mA, width_s) if width_s > 0 else i_mA
        ax1.plot(t, y, c=c, lw=1.2, label=label)
        ax1.axhline(mean, c=c, ls=":", lw=0.8)
    if xlim is not None:
        ax1.set_xlim(*xlim)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Current [mA]")
    smooth_txt = f"smoothed over {smooth_ms:.0f} ms" if width_s > 0 else "unsmoothed"
    ax1.set_title(f"Current over time ({smooth_txt}), dotted line = mean{title_extra}",
                  **({"fontsize": title_size} if title_size else {}))
    ax1.grid(alpha=0.3)
    ax2 = ax1.secondary_yaxis("right", functions=(lambda i: i * volts, lambda p: p / volts))
    ax2.set_ylabel(f"Power at {volts:g} V [mW]")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=9, framealpha=0.95)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Lagret {out}")


def plot_batch_log(t, i_ma, rows, markers, title, out=None):
    """The whole batch log with ID markers, analysis windows and capture starts (shown if out is None)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, i_ma, lw=0.6, c="0.3")
    top = np.nanmax(i_ma)
    for r, m in zip(rows, markers):
        ax.axvspan(m["marker_start"], m["test_start"], color="tab:orange", alpha=0.15)
        ax.axvspan(r["window_start_s"], r["window_end_s"], color="tab:green", alpha=0.2)
        if r["capture_start_s"] != "":
            ax.axvline(r["capture_start_s"], c="tab:blue", ls="--", lw=0.8)
        ax.text(m["test_start"], top, f" {r['id']}: {r['name']}", va="top", fontsize=8, rotation=90)
    ax.set_xlabel("Tid [s]")
    ax.set_ylabel("Strøm [mA]")
    ax.set_title(f"{title}  (oransje: ID-markør, grønn: analysevindu, blå: capture-start)", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Lagret {out}")
    else:
        plt.show()


def plot_raw_file(prefix, start=None, end=None):
    """Current and power for <prefix>_raw.csv in [start, end] -> <prefix>_plot.png, with statistics below."""
    prefix = power_io.strip_prefix(prefix)
    t, v, i, p = power_io.read_raw(f"{prefix}_raw.csv")
    s = power_io.read_sum(f"{prefix}_sum.csv")

    t0 = start if start is not None else t[0]
    t1 = end if end is not None else t[-1]
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        print(f"{prefix}: too few measurements between {t0} s and {t1} s, skipping")
        return
    ts, vs, is_mA, ps_mW = t[m], v[m], i[m] * 1e3, p[m] * 1e3

    # Statistics for the plotted window
    T = ts[-1] - ts[0]
    avg_i = integrate(ts, is_mA) / T
    avg_p = integrate(ts, ps_mW) / T
    energy_mWh = integrate(ts, ps_mW) / 3600.0
    avg_v = float(np.mean(vs))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1.plot(ts, is_mA, lw=0.8)
    ax1.axhline(avg_i, ls="--", c="k", lw=1, label=f"Mean {avg_i:.2f} mA")
    ax1.set_ylabel("Current [mA]")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(ts, ps_mW, lw=0.8, c="C1")
    ax2.axhline(avg_p, ls="--", c="k", lw=1, label=f"Mean {avg_p:.2f} mW")
    ax2.set_ylabel("Power [mW]")
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim(ts[0], ts[-1])
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle(os.path.basename(prefix))

    lines = [
        f"Plottet vindu {ts[0]:.0f}–{ts[-1]:.0f} s ({T:.0f} s, {len(ts)} målinger):  "
        f"U = {avg_v:.3f} V,  I = {avg_i:.2f} mA (min {is_mA.min():.2f}, maks {is_mA.max():.2f}),  "
        f"P = {avg_p:.2f} mW,  E = {energy_mWh:.4f} mWh"
    ]
    if s:
        lines.append(
            f"Sum file: start {s.get('start_time', '?')}, CH{s.get('channel', '?')}, "
            f"analysis window {float(s['window_start_s']):.0f}–{float(s['window_end_s']):.0f} s:  "
            f"I = {float(s['mean_current_mA']):.2f} mA,  P = {float(s['mean_power_mW']):.2f} mW,  "
            f"E = {float(s['energy_mWh']):.4f} mWh"
        )
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, va="bottom", ha="left", family="monospace")
    fig.tight_layout(rect=(0, 0.03 + 0.025 * len(lines), 1, 1))

    out = f"{prefix}_plot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Lagret {out}")


def show_log(t, i, p, avg_p):
    """Current and power of a live log [A, W] in a window."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(t, [x * 1e3 for x in i])
    ax1.set_ylabel("Strøm [mA]")
    ax2.plot(t, [x * 1e3 for x in p])
    ax2.axhline(avg_p * 1e3, ls="--", c="k", label=f"Snitt {avg_p * 1e3:.1f} mW")
    ax2.set_ylabel("Effekt [mW]")
    ax2.set_xlabel("Tid [s]")
    ax2.legend()
    fig.tight_layout()
    plt.show()
