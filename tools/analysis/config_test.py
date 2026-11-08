"""Wave heights for one session under different processing choices, side by side.

One combination is (D, fir, seglen, minutes, hz, taps):

  D        the raw stream averaged in blocks of D first (480/D Hz), as RAW_AVERAGE
  fir      the decimation stages: raw -> rows (stage 1) and rows -> Welch rate
           (stage 2). "average" (block / bucket mean) or "sinc" (the Hamming-windowed
           sinc FIR as on board) for both, or "S1/S2" to choose each, with the taps
           as "sinc<taps>" (e.g. "sinc31/sinc129", "average/sinc") - see stages()
  seglen   Welch segment length (at the Welch rate)
  minutes  the capture cut after this many minutes (as if it had stopped there)
  hz       the Welch rate: stage 2 decimates to it (default 10 Hz, as on board)
  taps     the FIR taps of every "sinc" without its own count (default: the
           capture's, fir_ntap in cfg.csv)

Per (D, fir) the session is reconstructed once into <out>/work/D<D>_<fir>/
(imu.csv and one raw-stream file per AHRS method, all with that stage 1) and reused
until deleted. The spectra are computed once per combination; every taper only
recomputes the moments.

Writes <out>/<prefix>_test_config.csv (one row per combination and taper, with Hs,
its approximate ±1σ from the Welch estimate and the segment count) and draws from it:
  <prefix>_test_config_<f1>_<f2>.png        table per taper (met_waves style)
  <prefix>_test_config_delta_<f1>_<f2>.png  ΔHs [%] per factor, ±1σ bars
  <prefix>_test_config_heatmap.png          ΔHs [%] vs the first row at psd_taper,
                                            decimation x the rest of the configuration
                                            (capture-length series left out)
  <prefix>_test_config_heatmap_capture.png  the capture-length series, capture x
                                            method, vs its longest capture
  <prefix>_test_config_capture.png          Hs against capture length, ±1σ bands
  <prefix>_test_config_psd_acc.png          acceleration PSD of the PSD_COMPARE combos
  <prefix>_test_config_psd_eta.png          their tapered elevation PSD
  <prefix>_test_config_chains_madgwick_<f1>_<f2>_<f>Hz.png  Hs and Tz (Madgwick) per decimation chain
                                  (+ .csv)  at D1, integrated to each of chain_fmax
  <prefix>_test_config_fmax.png (+ .csv)    their Hs (Madgwick) integrated to each
                                            of fmax_compare, from the saved PSDs Run with script_test_config.py (test_config.sh in the measurement directory).
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

import _paths                                               # noqa: F401
import postprocess                                          # noqa: E402
import raw_to_imu_csv as rti                                # noqa: E402
from kystvarsling import save_grouped_table                 # noqa: E402
from pipeline.ahrs import AHRS_METHODS                      # noqa: E402
from pipeline.cutoff import choose_cutoff, make_taper       # noqa: E402
from pipeline.sea_state import sea_state, wave_params       # noqa: E402
from pipeline.session_input import read_kv                  # noqa: E402
from pipeline.settings import Settings, normalise           # noqa: E402
from session_source import SIDECAR_FILES, _refresh, workdir  # noqa: E402
from welch import window_weights                           # noqa: E402

# fir -> (table label, stage 1 in raw_to_imu_csv, stage 2 in postprocess)
STAGE_NAMES = {"average": "mean", "mean": "mean", "sinc": "fir"}

# (column in the CSV, key in SeaState.wp, table head). No GPS: it is not touched by
# the choices tested here.
METHODS = [("madgwick", "m", "Madgwick"), ("kalman", "k", "Kalman"),
           ("nxp", "n", "Kalman NXP"), ("sflp", "s", "SFLP")]
HS_COLS = [f"hs_{name}" for name, _, _ in METHODS]
HS_HEADS = [head for _, _, head in METHODS]

DECIMATE_HZ = 10.0          # the Welch rate when a combination gives none


def matched_taps(n2, raw_hz, row_hz):
    """Taps of the single stage at raw_hz that matches a two-stage chain with n2 taps in
    stage 2 at row_hz: stage 2 seen from the raw rate is M1·(n2 - 1) + 1 taps,
    M1 = raw_hz / row_hz (met_decimation.tex) - 769 at 480 Hz for n2 = 129 at 80 Hz."""
    n = int(round(raw_hz / row_hz * (n2 - 1))) + 1
    return n if n % 2 else n + 1


def stages(fir, ntap=None, raw_hz=None, row_hz=None):
    """The FIR field of a combination -> ((method, taps), (method, taps)) for stage 1
    (raw -> rows) and stage 2 (rows -> Welch rate). method "mean" | "fir"; taps None =
    the capture's own (fir_ntap in cfg.csv).

    fir is one spec for both stages or "S1/S2"; a spec is "average", "sinc" or
    "sinc<taps>" (e.g. "sinc31/sinc129", "average/sinc"). ntap (the TAPS field)
    applies to every "sinc" without its own count.

    "single" / "single<taps>": ONE FIR straight from the raw rate to the Welch rate
    (the rows ARE the Welch series); stage 2 is ("pass", None) - one row per bucket,
    passed through unchanged. "matched<n2>": a single stage with matched_taps(n2) at
    the raw rate raw_hz (480/D) - None without raw_hz."""
    if fir.startswith("matched"):
        n2 = fir[len("matched"):]
        if not n2.isdigit():
            raise ValueError(fir)
        taps = matched_taps(int(n2), raw_hz, row_hz) if raw_hz else None
        return (("fir", taps), ("pass", None))
    if fir.startswith("single"):
        rest = fir[len("single"):]
        if rest and not rest.isdigit():
            raise ValueError(fir)
        return (("fir", int(rest) if rest else ntap), ("pass", None))
    parts = fir.split("/")
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise ValueError(fir)
    out = []
    for p in parts:
        name = p.rstrip("0123456789")
        if name not in STAGE_NAMES or (name != "sinc" and name != p):
            raise ValueError(fir)
        taps = int(p[len(name):]) if p != name else None
        method = STAGE_NAMES[name]
        out.append((method, (taps or ntap) if method == "fir" else None))
    return tuple(out)


def parse_combo(text):
    """"D FIR SEGLEN MIN [HZ [TAPS]]" -> (int, str, int, float, float, int|None); HZ
    defaults to DECIMATE_HZ, TAPS (the sinc FIR's taps) to None = the capture's own
    (fir_ntap in cfg.csv). FIR: see stages()."""
    try:
        parts = text.split()
        if len(parts) == 4:
            parts.append(str(DECIMATE_HZ))
        if len(parts) == 5:
            parts.append("0")
        d, fir, seglen, minutes, hz, ntap = parts
        combo = (int(d), fir.lower(), int(seglen), float(minutes), float(hz),
                 int(ntap) or None)
        st = stages(combo[1], combo[5])
    except ValueError:
        sys.exit(f"combination '{text}': expected 'D FIR SEGLEN MINUTES [HZ [TAPS]]' "
                 f"with FIR average, sinc, sinc<taps> or S1/S2 of those, e.g. "
                 f"'2 average 1024 30', '1 sinc 512 30 5 31' or '1 sinc31/sinc129 1024 30'")
    for method, taps in st:
        if taps is not None and (taps < 3 or taps % 2 == 0):
            sys.exit(f"combination '{text}': the FIR needs an odd tap count >= 3")
    return combo


def fir_label(fir, ntap):
    """The FIR column: "Average", "Hamming/Sinc [(N taps)]" when both stages are the
    same, else "S1 / S2" (e.g. "Sinc 31 / Sinc 129", "Average / Sinc")."""
    if fir.startswith("matched"):
        return f"Single-stage Sinc, matched to {fir[len('matched'):]}"
    s1, s2 = stages(fir, ntap)
    if s2[0] == "pass":
        return "Single-stage Sinc" + (f" {s1[1]}" if s1[1] else "")
    if s1 == s2:
        if s1[0] == "mean":
            return "Average"
        return "Hamming/Sinc" + (f" ({s1[1]} taps)" if s1[1] else "")

    def one(st):
        return "Average" if st[0] == "mean" else "Sinc" + (f" {st[1]}" if st[1] else "")
    return f"{one(s1)} / {one(s2)}"


def work_dir(sess_dir, stamp, out_dir, d, fir, ntap=None, hz=None):
    """<out>/work/D<d>_<stage 1>/ (average, sinc or sinc_n<taps>) with the session
    reconstructed at 480/d Hz and the stage 1 of the combination, for the capture's
    own filter (imu.csv) and every other AHRS method (<stamp>_imu_raw_<method>.csv).
    Existing files are reused. A single stage (stages()) builds the rows at the
    Welch rate hz instead, in D<d>_single[_n<taps>]_<hz>Hz/, with output_rate_hz and
    window_ms in its cfg.csv set to match."""
    cfg_kv = read_kv(os.path.join(sess_dir, f"{stamp}_cfg.csv"))
    raw_hz = float(cfg_kv.get("imu_odr_hz", 480)) / d
    (stage1, taps1), (stage2, _) = stages(fir, ntap, raw_hz,
                                          float(cfg_kv.get("output_rate_hz", 80)))
    tag = "average" if stage1 == "mean" else "sinc" + (f"_n{taps1}" if taps1 else "")
    row_hz = None
    if stage2 == "pass":
        row_hz = int(hz)
        tag = "single" + (f"_n{taps1}" if taps1 else "") + f"_{hz:g}Hz"
    out = os.path.join(out_dir, "work", f"D{d}_{tag}")
    os.makedirs(out, exist_ok=True)
    for suf in SIDECAR_FILES:
        _refresh(os.path.join(sess_dir, stamp + suf), os.path.join(out, stamp + suf))
    if row_hz:
        cfg = os.path.join(out, f"{stamp}_cfg.csv")
        with open(cfg) as fh:
            lines = fh.read().splitlines()
        new = {"output_rate_hz": f"{row_hz}", "window_ms": f"{1000.0 / row_hz:.4f}"}
        with open(cfg, "w") as fh:
            fh.write("\n".join(f"{k},{new[k]}" if (k := ln.split(",", 1)[0]) in new
                               else ln for ln in lines) + "\n")
    raw = os.path.join(sess_dir, f"{stamp}_raw.bin")
    if not os.path.isfile(raw):
        sys.exit(f"found no {stamp}_raw.bin in {sess_dir} - the test needs the raw log")
    own = read_kv(os.path.join(out, f"{stamp}_cfg.csv")).get("orientation_name", "")
    jobs = [(None, f"{stamp}_imu.csv")] + [
        (m.name, f"{stamp}_imu_raw_{m.name}.csv") for m in AHRS_METHODS
        if m.name != own.strip().lower()]
    for ahrs, name in jobs:
        path = os.path.join(out, name)
        if not os.path.isfile(path):
            print(f"  reconstructing {os.path.basename(out)}/{name} ...", flush=True)
            rti.write_imu_csv(raw, path, ahrs=ahrs, verbose=False, average=d,
                              stage1=stage1, ntap=taps1, row_odr_hz=row_hz)
    # The rate the methods ran at (raw_stream.raw_average reads it).
    marker = os.path.join(out, f"{stamp}_raw_average.txt")
    if d > 1:
        with open(marker, "w") as f:
            f.write(f"{d}\n")
    elif os.path.isfile(marker):
        os.remove(marker)
    return out


def hs_relsd(eta, taper, df, fmax, nseg, seglen, overlap_div, window):
    """Approximate relative standard deviation of Hs from a Welch estimate.

    m0 = Σ S_k·df with S_k = eta_k·T_k² up to fmax. Each bin is an average of N_eff
    independent periodograms; adjacent bins are correlated through the window. For a
    spectrum that is smooth over a few bins,
        var(m0)/m0² ≈ F_bin · Σ S_k² / (N_eff · (Σ S_k)²)
    with N_eff = nseg / (1 + 2 Σ_l (1 - l/nseg) c_l²) (c_l: the window's overlap
    correlation at l steps) and F_bin = 1 + 2 Σ_j ρ_j (ρ_j: the periodogram
    correlation j bins apart). Hs = 4√m0, so σ_Hs/Hs = σ_m0/(2 m0)."""
    if not nseg:
        return None
    k = np.arange(1, len(eta))
    keep = (k * df <= fmax + 1e-12) & (np.asarray(taper)[1:] > 0)
    s = (np.asarray(eta)[1:] * np.asarray(taper)[1:] ** 2)[keep]
    if s.sum() <= 0:
        return None
    w = window_weights(seglen, window)
    w2 = float(np.sum(w * w))
    step = seglen // overlap_div
    c = [float(np.sum(w[:seglen - l * step] * w[l * step:])) / w2
         for l in range(1, overlap_div)]
    n_eff = nseg / (1.0 + 2.0 * sum((1.0 - (l + 1) / nseg) * cl * cl
                                     for l, cl in enumerate(c)))
    n = np.arange(seglen)
    rho = [abs(np.sum(w * w * np.exp(-2j * np.pi * j * n / seglen))) ** 2 / w2 ** 2
           for j in (1, 2, 3)]
    f_bin = 1.0 + 2.0 * sum(rho)
    rel_m0 = np.sqrt(f_bin * np.sum(s * s) / (n_eff * s.sum() ** 2))
    return 0.5 * float(rel_m0)


def settings(fir, seglen, minutes, taper, hz, ntap=None):
    _, (stage2, taps2) = stages(fir, ntap)
    # A single stage: the rows are at the Welch rate, one per bucket - the bucket
    # mean passes them through unchanged.
    st = Settings.from_flat(decimate_hz=hz, decimate="mean" if stage2 == "pass" else stage2,
                            fir_ntap=taps2,
                            seglen=seglen,
                            capture_min=minutes, taper_f1=taper[0], taper_f2=taper[1],
                            plot=False)
    normalise(st)
    return st


def compute(sess_dir, stamp, combos, tapers, out_dir):
    """Every combination under every taper -> (DataFrame with one row each,
    DataFrame of the untapered acceleration PSD per combination and bin)."""
    odr = float(read_kv(os.path.join(sess_dir, f"{stamp}_cfg.csv")).get("imu_odr_hz", 480))
    rows, psd = [], []
    for d, fir, seglen, minutes, hz, ntap in combos:
        print(f"--- D={d} ({odr / d:g} Hz), {fir_label(fir, ntap)}, seglen {seglen}, "
              f"{minutes:g} min, Welch {hz:g} Hz ---", flush=True)
        work = work_dir(sess_dir, stamp, out_dir, d, fir, ntap, hz)
        st = settings(fir, seglen, minutes, tapers[0], hz, ntap)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = postprocess.analyse_session(work, st)
        except BaseException:
            print("\n".join(buf.getvalue().strip().splitlines()[-15:]), file=sys.stderr)
            raise
        sp = r.elev.spectra
        f = np.arange(sp.seglen // 2 + 1) * sp.fs / sp.seglen
        psd.append(pd.DataFrame(dict(
            decimation=d, fir=fir_label(fir, ntap), welch_hz=hz, seglen=seglen, capture_min=minutes,
            f=f[1:], **{f"acc_{name}": sp.acc[key][1:] for name, key, _ in METHODS})))
        for taper in tapers:
            stt = settings(fir, seglen, minutes, taper, hz, ntap)
            cut = choose_cutoff(r.elev, None, stt.lowfreq, stt.sea_state.fmax)
            sea = sea_state(r.elev, cut, stt.sea_state.param_fmax)
            nseg = r.spectra.nseg
            row = dict(taper_f1=taper[0], taper_f2=taper[1], decimation=d,
                       rate_hz=odr / d, fir=fir_label(fir, ntap), welch_hz=hz, seglen=seglen,
                       capture_min=minutes, param_fmax_hz=stt.sea_state.param_fmax,
                       welch_segments=nseg)

            def relsd(eta, key):
                return hs_relsd(eta, cut.tapers[key], r.elev.df,
                                stt.sea_state.param_fmax, nseg, stt.welch.seglen,
                                stt.welch.overlap_div, stt.welch.window)
            for name, key, _ in METHODS:
                row[f"hs_{name}"] = sea.wp[key]["hs"]
                row[f"hs_{name}_relsd"] = relsd(r.elev.eta[key], key)
            for c in HS_COLS:                      # -1 = undefined
                if row[c] is not None and row[c] < 0:
                    row[c] = row[c + "_relsd"] = None
            rows.append(row)
            print(f"  taper {taper[0]:g}-{taper[1]:g}: Hs Madgwick "
                  f"{row['hs_madgwick'] or float('nan'):.3f} m")
    return pd.DataFrame(rows), pd.concat(psd, ignore_index=True)


def draw(df, description, stamp, prefix, out_dir):
    """One table PNG per taper from the CSV's rows."""
    for (f1, f2), part in df.groupby(["taper_f1", "taper_f2"], sort=False):
        body = []
        for r in part.itertuples():
            body.append([f"{r.decimation} ({r.rate_hz:g} Hz)", r.fir,
                         f"{r.welch_hz:g} Hz", str(r.seglen),
                         f"{r.capture_min:g} min"]
                        + ["-" if pd.isna(getattr(r, c)) else f"{getattr(r, c):.3f}"
                           for c in HS_COLS])
        fmax = part["param_fmax_hz"].iloc[0]
        footer = (f"{description}, {stamp}. Taper {f1:g}–{f2:g} Hz, Hs integrated to "
                  f"{fmax:g} Hz. FIR applies to both decimation stages.")
        path = os.path.join(out_dir, f"{prefix}_test_config_{f1:g}_{f2:g}.png")
        save_grouped_table(["Configuration", "Hs [m]"],
                           [["Decimation", "FIR", "Welch rate", "Welch seglen", "Capture"],
                            HS_HEADS],
                           body, path, footer, row_labels=False)
        print(f"  wrote {os.path.basename(path)}")


# --- Figures --------------------------------------------------------------------

# (CSV column, axis label, the value the others are compared with: "max" = largest)
FACTORS = [("decimation", "Decimation D", 1), ("fir", "FIR", "Hamming/Sinc"),
           ("welch_hz", "Welch rate [Hz]", "max"),
           ("seglen", "Welch segment length", "max"), ("capture_min", "Capture [min]", "max")]
CONFIG_COLS = [f for f, _, _ in FACTORS]
COLOURS = {"hs_madgwick": "C0", "hs_kalman": "C2", "hs_nxp": "C3", "hs_sflp": "C1"}
MARKERS = {"hs_madgwick": "o", "hs_kalman": "s", "hs_nxp": "D", "hs_sflp": "^"}


def _label(r):
    return (f"D{int(r['decimation'])}, {r['fir']}, {r['welch_hz']:g} Hz, "
            f"{int(r['seglen'])}, {r['capture_min']:g} min")


def _order(factor, values):
    if factor == "fir":
        return sorted(values, key=lambda v: (v != "Average", v))
    return sorted(values)


def _baseline(group, factor, default):
    want = group[factor].max() if default == "max" else default
    hit = group[group[factor] == want]
    return hit.iloc[0] if len(hit) else group.iloc[0]


def draw_delta(part, path):
    """One panel per factor: ΔHs [%] against the factor's reference value, for
    every set of rows that differ in that factor alone; ±1σ (Welch) error bars."""
    import matplotlib.pyplot as plt
    panels = []
    for factor, label, default in FACTORS:
        others = [c for c in CONFIG_COLS if c != factor]
        lines = [g for _, g in part.groupby(others, sort=False) if g[factor].nunique() > 1]
        if lines:
            panels.append((factor, label, default, others, lines))
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4),
                             sharey=True, squeeze=False)
    for ax, (factor, label, default, others, lines) in zip(axes[0], panels):
        cats = _order(factor, set().union(*[set(g[factor]) for g in lines]))
        pos = {v: i for i, v in enumerate(cats)}
        for li, g in enumerate(lines):
            g = g.set_index(factor).loc[[v for v in cats if v in set(g[factor])]].reset_index()
            base = _baseline(g, factor, default)
            ls = ["-", "--", ":"][li % 3]
            for mi, c in enumerate(HS_COLS):
                if pd.isna(base[c]):
                    continue
                x = np.array([pos[v] for v in g[factor]]) + (mi - 1.5) * 0.07
                y = 100.0 * (g[c] / base[c] - 1.0)
                err = 100.0 * g[c + "_relsd"] * g[c] / base[c]
                ax.errorbar(x, y, yerr=err, color=COLOURS[c], marker=MARKERS[c], ms=4,
                            ls=ls, lw=1.0, capsize=2, elinewidth=0.8)
            ref = ", ".join(f"{base[o]:g}" if o != "fir" else str(base[o]) for o in others)
            ax.plot([], [], color="0.4", ls=ls, label=ref)
        ax.axhline(0.0, color="0.5", lw=0.8, zorder=0)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{v:g}" if not isinstance(v, str) else v for v in cats])
        ax.set_xlim(-0.5, len(cats) - 0.5)
        ax.set_xlabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=6, loc="best", title=" / ".join(
            {"decimation": "D", "fir": "FIR", "welch_hz": "Hz", "seglen": "seglen",
             "capture_min": "min"}[o] for o in others), title_fontsize=6)
    axes[0][0].set_ylabel("ΔHs  [%]")
    handles = [plt.Line2D([], [], color=COLOURS[c], marker=MARKERS[c], ls="", label=h)
               for c, h in zip(HS_COLS, HS_HEADS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _heat(ax, grid, vmax, xlabels):
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=6.5)
    return im


def _vmax(grids):
    return max(5.0, float(np.nanpercentile(np.abs(np.concatenate(
        [np.ravel(g) for g in grids])), 98)))


def draw_heatmap(df, path):
    """ΔHs [%] against the first row (the CSV's first configuration), at one taper:
    one panel per method, the decimation D across, the rest of the configuration
    down. Combinations not run are left blank."""
    import matplotlib.pyplot as plt
    rest = [c for c in CONFIG_COLS if c != "decimation"]
    ds = sorted(df["decimation"].unique())
    rate = {d: df[df["decimation"] == d]["rate_hz"].iloc[0] for d in ds}
    rows = list(dict.fromkeys(tuple(r) for r in df[rest].itertuples(index=False)))
    ref = df.iloc[0]
    grids = {}
    for c in HS_COLS:
        g = np.full((len(rows), len(ds)), np.nan)
        for _, r in df.iterrows():
            if not (pd.isna(r[c]) or pd.isna(ref[c])):
                g[rows.index(tuple(r[k] for k in rest)), ds.index(r["decimation"])] = \
                    100.0 * (r[c] / ref[c] - 1.0)
        grids[c] = g
    vmax = _vmax(grids.values())
    fig, axes = plt.subplots(1, len(HS_COLS), figsize=(1.5 * len(HS_COLS) + 2.4,
                                                       0.32 * len(rows) + 1.2),
                             sharey=True)
    for ax, c, head in zip(axes, HS_COLS, HS_HEADS):
        im = _heat(ax, grids[c], vmax, [f"D{d:g}\n({rate[d]:g} Hz)" for d in ds])
        ax.set_title(head, fontsize=8)
    axes[0].set_yticks(range(len(rows)))
    # Row labels without what every row shares (it is in the colourbar label).
    fmt = {"fir": str, "welch_hz": lambda v: f"{v:g} Hz", "seglen": lambda v: f"{int(v)}",
           "capture_min": lambda v: f"{v:g} min"}
    shown = [k for k, c in enumerate(rest) if len({r[k] for r in rows}) > 1] \
        or list(range(len(rest)))
    axes[0].set_yticklabels([", ".join(fmt[rest[k]](r[k]) for k in shown) for r in rows],
                            fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(f"ΔHs [%] vs\n{_label(ref)}", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap_capture(df, path):
    """ΔHs [%] of a capture-length series against its first row (the longest), at one
    taper: capture length down, method across."""
    import matplotlib.pyplot as plt
    ref = df.iloc[0]
    grid = np.array([[100.0 * (r[c] / ref[c] - 1.0) for c in HS_COLS]
                     for _, r in df.iterrows()])
    vmax = _vmax([grid])
    fig, ax = plt.subplots(figsize=(0.75 * len(HS_COLS) + 1.4, 0.32 * len(df) + 1.0))
    im = _heat(ax, grid, vmax, HS_HEADS)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{m:g} min" for m in df["capture_min"]], fontsize=6.5)
    ax.set_ylabel("Capture", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"ΔHs [%] vs {_label(ref)}", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_capture(df, path):
    """Hs against capture length (the largest set of rows differing in it alone), one
    panel per taper, ±1σ (Welch) bands."""
    import matplotlib.pyplot as plt
    others = [c for c in CONFIG_COLS if c != "capture_min"]
    groups = [g for _, g in df.groupby(others, sort=False)
              if g["capture_min"].nunique() > 1]
    if not groups:
        return False
    g = max(groups, key=lambda x: x["capture_min"].nunique())
    tapers = list(dict.fromkeys(zip(g["taper_f1"], g["taper_f2"])))
    ncol = min(3, len(tapers))
    nrow = -(-len(tapers) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax, (f1, f2) in zip(axes.ravel(), tapers):
        t = g[(g["taper_f1"] == f1) & (g["taper_f2"] == f2)].sort_values("capture_min")
        for c, head in zip(HS_COLS, HS_HEADS):
            if t[c].isna().all():
                continue
            sd = t[c] * t[c + "_relsd"]
            ax.plot(t["capture_min"], t[c], color=COLOURS[c], marker=MARKERS[c], ms=3.5,
                    lw=1.1, label=head)
            ax.fill_between(t["capture_min"], t[c] - sd, t[c] + sd, color=COLOURS[c],
                            alpha=0.12, lw=0)
        ax.set_title(f"taper {f1:g}–{f2:g} Hz", fontsize=8)
        ax.set_xlabel("Capture [min]", fontsize=7)
        ax.set_ylabel("Hs [m]", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(tapers):]:
        ax.axis("off")
    r0 = g.iloc[0]
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=7,
               frameon=False, title=f"D{int(r0['decimation'])}, {r0['fir']}, "
               f"{r0['welch_hz']:g} Hz, seglen {int(r0['seglen'])}  (bands: ±1σ, Welch)", title_fontsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def draw_psd_compare(psd, combos, path, table=None, taper=None, quantity="acc"):
    """The PSD (Madgwick) of the given combinations: quantity "acc" the untapered
    acceleration, "eta" the tapered elevation acc/ω⁴·T² - what Hs integrates. With
    table (the CSV's rows) and taper, each legend entry gets Madgwick's Hs for that
    taper; "eta" needs the taper."""
    import matplotlib.pyplot as plt
    from freq_axis import log_hz_axis
    sel = []
    for d, fir, seglen, minutes, hz, ntap in combos:
        m = psd[(psd["decimation"] == d) & (psd["fir"] == fir_label(fir, ntap))
                & (psd["welch_hz"] == hz) & (psd["seglen"] == seglen)
                & (psd["capture_min"] == minutes)]
        if m.empty:
            sys.exit(f"--psd-compare '{d} {fir} {seglen} {minutes:g} {hz:g}' is not "
                     f"among the combinations (COMBOS)")
        label = _label(m.iloc[0])
        if table is not None and taper is not None:
            t = table[(table["decimation"] == d) & (table["fir"] == fir_label(fir, ntap))
                      & (table["welch_hz"] == hz) & (table["seglen"] == seglen)
                      & (table["capture_min"] == minutes)
                      & np.isclose(table["taper_f1"], taper[0])
                      & np.isclose(table["taper_f2"], taper[1])]
            if t.empty:
                sys.exit(f"--psd-taper {taper[0]:g} {taper[1]:g} is not among the tapers")
            if not pd.isna(t["hs_madgwick"].iloc[0]):
                label += f", $H_s$ = {t['hs_madgwick'].iloc[0]:.3f} m"
        sel.append((label, m.sort_values("f")))
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    xlim = (min(float(m["f"].min()) for _, m in sel),
            max(float(m["f"].max()) for _, m in sel))
    for i, (label, m) in enumerate(sel):
        y = m["acc_madgwick"].to_numpy()
        if quantity == "eta":
            f = m["f"].to_numpy()
            t = make_taper(f, *taper)
            y = np.where(t > 0, y / (2 * np.pi * f) ** 4 * t ** 2, np.nan)
        ax1.loglog(m["f"], y, color=f"C{i}", lw=1.2,
                   ls="-" if i == 0 else "--", label=label)
    ax1.set_ylabel("PSD elevation (tapered)  [m²/Hz]" if quantity == "eta"
                   else "PSD acceleration  [(m/s²)²/Hz]")
    ax1.set_xlabel("frequency [Hz]")
    log_hz_axis(ax1, xlim)
    ax1.grid(True, which="both", alpha=0.25)
    title = None
    if table is not None and taper is not None:
        fmax = table["param_fmax_hz"].iloc[0]
        title = (f"$H_s$: Madgwick, taper {taper[0]:g}–{taper[1]:g} Hz, "
                 f"integrated to {fmax:g} Hz")
        # What enters Hs: nothing below f1 (grey), the taper ramp (yellow), up to fmax.
        ax1.axvspan(xlim[0], taper[0], color="0.6", alpha=0.25, lw=0, zorder=0,
                    label=f"excluded (< {taper[0]:g} Hz)")
        ax1.axvspan(taper[0], taper[1], color="yellow", alpha=0.3, lw=0, zorder=0,
                    label=f"taper {taper[0]:g}–{taper[1]:g} Hz")
        if fmax < xlim[1]:
            ax1.axvline(fmax, color="0.25", ls="-.", lw=1.1, zorder=1,
                        label=f"end of integration ({fmax:g} Hz)")
    ax1.legend(fontsize=8, title=title, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _select(frame, combo):
    d, fir, seglen, minutes, hz, ntap = combo
    return frame[(frame["decimation"] == d) & (frame["fir"] == fir_label(fir, ntap))
                 & (frame["welch_hz"] == hz) & (frame["seglen"] == seglen)
                 & (frame["capture_min"] == minutes)]


def fmax_table(psd, table, combos, taper, fmaxes):
    """Madgwick's Hs of each combination, from its saved acceleration PSD with taper,
    integrated to each of fmaxes [Hz] -> one row per (combination, fmax). table (the
    CSV's rows) gives the sample rate."""
    rows = []
    for combo in combos:
        m = _select(psd, combo).sort_values("f")
        if m.empty:
            sys.exit(f"--psd-compare '{' '.join(map(str, combo))}' is not among the "
                     f"combinations (COMBOS)")
        f = np.concatenate([[0.0], m["f"].to_numpy()])          # bin k at k*df, DC = 0
        df = f[1]
        acc = np.concatenate([[0.0], m["acc_madgwick"].to_numpy()])
        eta = np.zeros_like(acc)
        eta[1:] = acc[1:] / (2 * np.pi * f[1:]) ** 4
        t = make_taper(f, *taper)
        rate = _select(table, combo)["rate_hz"].iloc[0]
        for fmax in fmaxes:
            rows.append(dict({c: m.iloc[0][c] for c in CONFIG_COLS}, rate_hz=rate,
                             taper_f1=taper[0], taper_f2=taper[1], fmax_hz=fmax,
                             hs_madgwick=wave_params(eta, df, fmax, t)["hs"]))
    return pd.DataFrame(rows)


def draw_fmax(tab, description, stamp, path):
    """The fmax_table as a met_waves-style table: one row per combination, Hs per fmax
    with the change against the first fmax in brackets. Configuration columns that
    are the same in every row go in the footer instead."""
    fmaxes = list(dict.fromkeys(tab["fmax_hz"]))
    first = tab.drop_duplicates(CONFIG_COLS)
    cell = {"decimation": lambda r: f"{int(r['decimation'])} ({r['rate_hz']:g} Hz)",
            "fir": lambda r: str(r["fir"]),
            "welch_hz": lambda r: f"{r['welch_hz']:g} Hz",
            "seglen": lambda r: str(int(r["seglen"])),
            "capture_min": lambda r: f"{r['capture_min']:g} min"}
    head = {"decimation": "Decimation", "fir": "FIR", "welch_hz": "Welch rate",
            "seglen": "Welch seglen", "capture_min": "Capture"}
    vary = [c for c in CONFIG_COLS if first[c].nunique() > 1] or ["fir"]
    same = [c for c in CONFIG_COLS if c not in vary]
    body = []
    for _, g in tab.groupby(CONFIG_COLS, sort=False):
        r = g.iloc[0]
        hs = g.set_index("fmax_hz")["hs_madgwick"]
        body.append([cell[c](r) for c in vary] + [f"{hs[fmaxes[0]]:.3f}"]
                    + [f"{hs[x]:.3f} ({100.0 * (hs[x] / hs[fmaxes[0]] - 1.0):+.1f} %)"
                       for x in fmaxes[1:]])
    r = first.iloc[0]
    common = ", ".join(f"{head[c]} {cell[c](r)}" for c in same)
    f1, f2 = tab["taper_f1"].iloc[0], tab["taper_f2"].iloc[0]
    footer = (f"{description}, {stamp}.\nMadgwick, taper {f1:g}–{f2:g} Hz."
              + (f"\n{common}." if common else ""))
    save_grouped_table(["Configuration", "Hs [m], integrated to"],
                       [[head[c] for c in vary], [f"{x:g} Hz" for x in fmaxes]],
                       body, path, footer, row_labels=False, col_width=1.5)


def chain_name(fir, ntap, own_ntap):
    """A chain for the chain table: "Single-stage 129", "Single-stage, matched",
    "Multi 31+129", "Average"... own_ntap fills in the taps a combination leaves to
    the capture."""
    if fir.startswith("matched"):
        return "Single-stage, matched"
    s1, s2 = stages(fir, ntap)
    if s2[0] == "pass":
        return f"Single-stage {s1[1] or own_ntap}"
    if s1[0] == s2[0] == "mean":
        return "Average"

    def one(st):
        return "Average" if st[0] == "mean" else str(st[1] or own_ntap)
    return f"Multi {one(s1)}+{one(s2)}"


def psd_params(psd, row, taper, fmax, method="madgwick"):
    """The wave parameters (wave_params: hs, tp, tz, tc...) of one AHRS method
    (METHODS) for the combination in row (a CSV row), from its saved acceleration PSD
    with taper, integrated to fmax [Hz] - as the pipeline does."""
    m = psd[(psd["decimation"] == row["decimation"]) & (psd["fir"] == row["fir"])
            & (psd["welch_hz"] == row["welch_hz"]) & (psd["seglen"] == row["seglen"])
            & (psd["capture_min"] == row["capture_min"])].sort_values("f")
    f = np.concatenate([[0.0], m["f"].to_numpy()])          # bin k at k*df, DC = 0
    acc = np.concatenate([[0.0], m[f"acc_{method}"].to_numpy()])
    eta = np.zeros_like(acc)
    eta[1:] = acc[1:] / (2 * np.pi * f[1:]) ** 4
    return wave_params(eta, f[1], fmax, make_taper(f, *taper))


def draw_chain_table(df, psd, chains, taper, fmax, own_ntap, row_hz, path):
    """Madgwick's Hs (with the change against the first row) and Tz per chain (rows,
    one per Welch rate) at D1, one taper and the longest capture, integrated to fmax
    from the saved PSDs. chains = [(fir, ntap)] as in the combinations. -> the table
    as a DataFrame."""
    part = df[np.isclose(df["taper_f1"], taper[0]) & np.isclose(df["taper_f2"], taper[1])]
    part = part[(part["capture_min"] == part["capture_min"].max())
                & (part["decimation"] == part["decimation"].min())]
    d = part["decimation"].iloc[0]
    raw_hz = part["rate_hz"].iloc[0]
    rows = []
    for fir, ntap in chains:
        sel = part[part["fir"] == fir_label(fir, ntap)]
        if sel.empty:
            sys.exit(f"--chain-table '{fir}': no such combination in COMBOS at D{d:g}")
        name = chain_name(fir, ntap, own_ntap)
        if fir.startswith("matched"):
            name = f"Single-stage {matched_taps(int(fir[len('matched'):]), raw_hz, row_hz)}"
        for hz in sorted(sel["welch_hz"].unique(), reverse=True):
            wp = psd_params(psd, sel[sel["welch_hz"] == hz].iloc[0], taper, fmax)
            rows.append(dict(chain=name, welch_hz=hz, hs_madgwick=wp["hs"],
                             tz_madgwick=wp["tz"]))
    ref = rows[0]["hs_madgwick"]
    # Pad the rates with figure spaces (one digit wide) so "Hz" lines up when centred.
    width = max(len(f"{r['welch_hz']:g}") for r in rows)
    body = [[r["chain"], f"{r['welch_hz']:g}".rjust(width, "\u2007") + " Hz",
             f"{r['hs_madgwick']:.3f} ({100.0 * (r['hs_madgwick'] / ref - 1.0) + 0.0:+.1f} %)"
             .replace("-0.0", "+0.0"),
             f"{r['tz_madgwick']:.2f}"] for r in rows]
    save_grouped_table(["Decimation chain", "Result"],
                       [["Chain", "Welch rate"], ["Hs [m]", "Tz [s]"]],
                       body, path, row_labels=False,
                       col_width=[2.6, 1.3, 1.8, 1.1])
    return pd.DataFrame(rows)


def draw_decimation_table(df, psd, combo, taper, fmax, path):
    """Hs and Tp per AHRS filter (rows, SFLP left out) at each decimation D (column groups, with the
    change against the smallest D in brackets) for one combination combo = (fir,
    seglen, minutes, hz, ntap) - the D field left out - one taper, integrated to fmax
    from the saved PSDs. -> the table as a DataFrame."""
    fir, seglen, minutes, hz, ntap = combo
    part = df[np.isclose(df["taper_f1"], taper[0]) & np.isclose(df["taper_f2"], taper[1])
              & (df["fir"] == fir_label(fir, ntap)) & (df["seglen"] == seglen)
              & (df["capture_min"] == minutes) & (df["welch_hz"] == hz)]
    if part.empty:
        sys.exit(f"--decimation-table: no combination '{fir} {seglen} {minutes:g} "
                 f"{hz:g}' in COMBOS")
    part = part.sort_values("decimation")

    def change(v, ref):
        return f" ({100.0 * (v / ref - 1.0) + 0.0:+.1f} %)".replace("-0.0", "+0.0")

    rows, body = [], []
    for name, _, head in METHODS:
        if name == "sflp":                   # left out of this table
            continue
        wps = [psd_params(psd, r, taper, fmax, name) for _, r in part.iterrows()]
        cells = [head]
        for i, (wp, (_, r)) in enumerate(zip(wps, part.iterrows())):
            rows.append(dict(method=name, decimation=r["decimation"], rate_hz=r["rate_hz"],
                             hs=wp["hs"], tp=wp["tp"]))
            cells += [f"{wp['hs']:.3f}" + (change(wp["hs"], wps[0]["hs"]) if i else ""),
                      f"{wp['tp']:.2f}" + (change(wp["tp"], wps[0]["tp"]) if i else "")]
        body.append(cells)
    save_grouped_table([""] + [f"D{r.decimation:g} ({r.rate_hz:g} Hz)"
                                     for r in part.itertuples()],
                       [["AHRS filter"]] + [["Hs [m]", "Tp [s]"]] * len(part),
                       body, path, row_labels=False,
                       col_width=[1.4] + [1.0, 0.8] + [1.6, 1.5] * (len(part) - 1))
    return pd.DataFrame(rows)

def run(sess_dir, combos, tapers, out_dir, prefix, description, plot_only=False,
        psd_compare=(), psd_taper=None, fmax_compare=(), chain_table=(), chain_fmax=(),
        decimation_table=()):
    """Compute (unless plot_only) -> <prefix>_test_config.csv (+ the PSDs in
    data/<prefix>_test_config_psd.csv) -> the tables and figures. psd_compare: the
    combinations drawn in <prefix>_test_config_psd_acc.png and _psd_eta.png."""
    stamp = workdir(sess_dir)[1]
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f"{prefix}_test_config.csv")
    psd_csv = os.path.join(out_dir, "data", f"{prefix}_test_config_psd.csv")
    if not plot_only:
        table, psd = compute(sess_dir, stamp, combos, tapers, out_dir)
        table.to_csv(csv, index=False, float_format="%.6g")
        print(f"  wrote {os.path.basename(csv)}")
        os.makedirs(os.path.dirname(psd_csv), exist_ok=True)
        psd.to_csv(psd_csv, index=False, float_format="%.17g")
        print(f"  wrote data/{os.path.basename(psd_csv)}")
    if not os.path.isfile(csv):
        sys.exit(f"--plot-only: {csv} is missing - run without --plot-only first")
    df = pd.read_csv(csv)
    draw(df, description, stamp, prefix, out_dir)
    for (f1, f2), part in df.groupby(["taper_f1", "taper_f2"], sort=False):
        path = os.path.join(out_dir, f"{prefix}_test_config_delta_{f1:g}_{f2:g}.png")
        if draw_delta(part, path):
            print(f"  wrote {os.path.basename(path)}")
    # Heatmaps, at one taper: the capture-length series (the largest set of rows
    # differing in capture length alone) on its own, against its longest capture;
    # everything else, with only that series' longest capture, against the first row.
    heat_taper = psd_taper or tuple(tapers[0])
    part = df[np.isclose(df["taper_f1"], heat_taper[0])
              & np.isclose(df["taper_f2"], heat_taper[1])]
    others = [c for c in CONFIG_COLS if c != "capture_min"]
    groups = [g for _, g in part.groupby(others, sort=False)
              if g["capture_min"].nunique() > 1]
    capture = max(groups, key=lambda g: g["capture_min"].nunique()) if groups else None
    main = part
    if capture is not None:
        longest = capture["capture_min"].max()
        main = part.drop(capture.index[capture["capture_min"] != longest])
        capture = capture.sort_values("capture_min", ascending=False, kind="stable")
    path = os.path.join(out_dir, f"{prefix}_test_config_heatmap.png")
    draw_heatmap(main, path)
    print(f"  wrote {os.path.basename(path)}")
    if capture is not None:
        path = os.path.join(out_dir, f"{prefix}_test_config_heatmap_capture.png")
        draw_heatmap_capture(capture, path)
        print(f"  wrote {os.path.basename(path)}")
    if chain_table:
        cfg_kv = read_kv(os.path.join(sess_dir, f"{stamp}_cfg.csv"))
        own = int(float(cfg_kv.get("fir_ntap") or rti.fir.NTAP))
        row_hz = float(cfg_kv.get("output_rate_hz") or 80)
        psd = pd.read_csv(psd_csv)
        for fmax in chain_fmax or [df["param_fmax_hz"].iloc[0]]:
            path = os.path.join(out_dir, f"{prefix}_test_config_chains_madgwick_"
                                f"{heat_taper[0]:g}_{heat_taper[1]:g}_{fmax:g}Hz.png")
            tab = draw_chain_table(df, psd, chain_table, heat_taper, fmax, own, row_hz,
                                   path)
            tab.to_csv(path[:-4] + ".csv", index=False, float_format="%.6g")
            print(f"  wrote {os.path.basename(path)} (+ .csv)")
    for combo in decimation_table:
        fir, seglen, minutes, hz, ntap = combo
        tag = fir.replace("/", "-") + (f"_n{ntap}" if ntap else "")
        path = os.path.join(out_dir, f"{prefix}_test_config_decimation_{tag}_{hz:g}Hz_"
                            f"{seglen}_{heat_taper[0]:g}_{heat_taper[1]:g}.png")
        psd = pd.read_csv(psd_csv)
        tab = draw_decimation_table(df, psd, combo, heat_taper,
                                    df["param_fmax_hz"].iloc[0], path)
        tab.to_csv(path[:-4] + ".csv", index=False, float_format="%.6g")
        print(f"  wrote {os.path.basename(path)} (+ .csv)")
    path = os.path.join(out_dir, f"{prefix}_test_config_capture.png")
    if draw_capture(df, path):
        print(f"  wrote {os.path.basename(path)}")
    if psd_compare:
        if not os.path.isfile(psd_csv):
            sys.exit(f"--plot-only: {psd_csv} is missing - run without --plot-only first")
        path = os.path.join(out_dir, f"{prefix}_test_config_psd_acc.png")
        psd = pd.read_csv(psd_csv)
        taper = psd_taper or tuple(tapers[0])
        draw_psd_compare(psd, psd_compare, path, df, taper)
        print(f"  wrote {os.path.basename(path)}")
        path = os.path.join(out_dir, f"{prefix}_test_config_psd_eta.png")
        draw_psd_compare(psd, psd_compare, path, df, taper, quantity="eta")
        print(f"  wrote {os.path.basename(path)}")
        if fmax_compare:
            tab = fmax_table(psd, df, psd_compare, taper, fmax_compare)
            path = os.path.join(out_dir, f"{prefix}_test_config_fmax.csv")
            tab.to_csv(path, index=False, float_format="%.6g")
            print(f"  wrote {os.path.basename(path)}")
            path = os.path.join(out_dir, f"{prefix}_test_config_fmax.png")
            draw_fmax(tab, description, stamp, path)
            print(f"  wrote {os.path.basename(path)}")
