"""Step 12, output: the analysed session -> files.

  <stamp>_spec_python.csv  per frequency: accel and tapered elevation PSD per method
                           (+ GPS tapered and untapered)
  <stamp>_ana_python.csv   key,value: the analysis configuration, sources and rates,
                           f_c and Hs/Tp/Tz/Tc per method
  <stamp>_spec_acc[_<config>].png  accel PSD for every method
  <stamp>_spec_eta[_<config>].png  elevation PSD for every method and GPS

The firmware's own _spec.csv/_ana.csv are never touched.
"""

import math
import os

import _paths                         # noqa: F401
from configs import config_suffix, data_path
from freq_axis import log_hz_axis
from .ahrs import KEYS, METHODS
from .settings import TAPER_F1

# Fixed x range for the spectrum plot [Hz].
PLOT_XLIM = (TAPER_F1, 10.0)
# The accel legend's geometric mean is taken over f <= this [Hz]: the wave band and
# the noise floor, without the FIR roll-off above it.
GEO_AVG_FMAX = 1.0
# Line style per method: the curves often lie on top of each other. GPS is ":".
PLOT_LS = {"madgwick": "-", "sflp": "--", "kalman": "-.",
           "nxp": (0, (5, 1.5, 1, 1.5, 1, 1.5))}       # dash-dot-dot


def spectrum_bins(elev, tapers, fmax):
    """The per-frequency table, as spec.csv: acc_<key> = the accel PSD, eta_<key> =
    tapered elevation PSD (+ eta_gps/eta_gps_raw). Bins where every method's taper
    is 0 are skipped. Returns a dict of lists."""
    acc, df = elev.spectra.acc, elev.df
    gps = elev.eta_gps is not None
    cols = (["f", "T"] + [f"acc_{key}" for key in KEYS] + [f"eta_{key}" for key in KEYS]
            + (["eta_gps", "eta_gps_raw"] if gps else []))
    out = {c: [] for c in cols}
    for k in range(1, elev.seglen // 2 + 1):
        f = k * df
        if f > fmax:
            break
        if all(tapers[key][k] <= 0.0 for key in KEYS):
            continue
        out["f"].append(f)
        out["T"].append(1.0 / f)
        for key in KEYS:
            t = tapers[key][k]
            out[f"acc_{key}"].append(acc[key][k])
            out[f"eta_{key}"].append(elev.eta[key][k] * t * t)
        if gps:
            tg = tapers["g"]
            eta_g = elev.eta_gps[k]
            out["eta_gps"].append(eta_g * tg[k] * tg[k])
            out["eta_gps_raw"].append(eta_g)
    return out


def write_spec(path, bins):
    gps = "eta_gps" in bins
    with open(path, "w") as sf:
        sf.write("f_hz"
                 + "".join(f",psd_acc_{m.name},psd_eta_{m.name}" for m in METHODS)
                 + (",psd_eta_gps,psd_eta_gps_raw" if gps else "") + "\n")
        for i in range(len(bins["f"])):
            sf.write(f"{bins['f'][i]:.5f}"
                     + "".join(f",{bins['acc_' + m.key][i]:.6f},"
                               f"{bins['eta_' + m.key][i]:.6f}" for m in METHODS)
                     + (f",{bins['eta_gps'][i]:.6f},"
                        f"{bins['eta_gps_raw'][i]:.6f}" if gps else "") + "\n")


def write_ana(path, r, st):
    """The key,value summary of Result r."""
    series, sp, cut, sea = r.series, r.spectra, r.cut, r.sea
    w = st.welch
    with open(path, "w") as out:
        out.write("key,value\n")
        out.write(f"imu_rows,{series.n_rows}\n")
        out.write(f"brake_windows,{series.n_brake}\n")
        out.write(f"vacc10hz_samples,{len(series.acc[KEYS[0]])}\n")
        out.write(f"welch_segments,{sp.nseg}\n")
        out.write(f"welch_brake_reject,{w.brake_reject:g}\n")
        out.write(f"welch_rejected_brake,{sp.nrej.get('brake', 0)}\n")
        out.write(f"welch_rejected_ovf,{sp.nrej.get('fifo_ovf', 0)}\n")
        out.write(f"fifo_ovf_windows,{series.n_ovf}\n")
        out.write(f"skip_start_s,{sp.skip_start:g}\n")
        out.write(f"welch_skipped_samples,{sp.n_skipped}\n")

        # The analysis configuration, with the firmware's ana.csv keys.
        out.write(f"vacc_bucket_ms,{series.bucket_ms}\n")
        out.write(f"vacc_fs_hz,{series.fs:.3f}\n")
        out.write(f"welch_seglen,{sp.seglen}\n")
        out.write(f"welch_overlap_div,{w.overlap_div}\n")
        out.write(f"welch_step,{sp.seglen // w.overlap_div}\n")
        out.write(f"welch_window,{w.window.capitalize()}\n")   # as the firmware
        out.write(f"psd_df_hz,{sp.df:.6f}\n")
        out.write(f"wave_fmax_hz,{st.sea_state.fmax:.3f}\n")
        out.write(f"wave_param_fmax_hz,{st.sea_state.param_fmax:.3f}\n")
        out.write(f"taper_f1_hz,{st.lowfreq.taper_f1:.3f}\n")
        out.write(f"taper_f2_hz,{st.lowfreq.taper_f2:.3f}\n")

        out.write(f"cutoff_mode,{cut.mode}\n")
        out.write(f"decimate_mode,{series.mode}\n")
        out.write(f"vacc_source,{r.vacc.source}\n")
        if r.vacc.source_method:
            out.write(f"vacc_source_method,{r.vacc.source_method}\n")
            out.write(f"vacc_source_col,{r.vacc.source_col}\n")
        # Per method: source and rate.
        for name in (m.name for m in METHODS):
            if name in r.raw.sources:
                out.write(f"src_{name},{r.raw.sources[name]}\n")
            hz = r.raw.rates.get(name)
            if hz:
                out.write(f"rate_{name}_hz,{hz:g}\n")
        odr = r.session.actual_odr()
        if odr:
            out.write(f"imu_odr_actual_hz,{odr:.1f}\n")
        # Stage 1 as the reconstruction recorded it (session_source), stage 2 as run.
        marker = os.path.join(r.session.directory, f"{r.session.stamp}_stage1.txt")
        if os.path.isfile(marker):
            with open(marker) as fh:
                out.write(f"stage1,{fh.read().strip()}\n")
        fst = series.firstats
        out.write(f"stage2,{'mean' if fst is None else 'fir ' + str(fst['ntap'])}\n")
        if fst is not None:
            out.write(f"fir_ntap,{fst['ntap']}\n")
            out.write(f"fir_cutoff_hz,{fst['cutoff']:.3f}\n")
            out.write(f"fir_decimation,{fst['dec']}\n")
            out.write(f"fir_delay_ms,{fst['delay_s'] * 1000:.1f}\n")
        if sea.gps is not None:
            for p, fmt in (("Hs", ".3f"), ("Tp", ".2f"), ("Tz", ".2f"), ("Tc", ".2f")):
                out.write(f"{p}_gps,{sea.gps[p.lower()]:{fmt}}\n")
            gps_hz = r.raw.rates.get("gps")
            if gps_hz:
                out.write(f"rate_gps_hz,{gps_hz:g}\n")
            for p, fmt in (("Hs", ".3f"), ("Tp", ".2f"), ("Tz", ".2f"), ("Tc", ".2f")):
                out.write(f"{p}_gps_utapert,{sea.gps_raw[p.lower()]:{fmt}}\n")
            out.write(f"gps_vup_noise_psd,{r.gps_ref.noise:.3e}\n")
        for m in METHODS:
            fc = cut.fcs[m.key]
            out.write(f"fc_{m.name},{fc:.4f}\n" if fc is not None
                      else f"fc_{m.name},-1\n")
        for m in METHODS:
            wp = sea.wp[m.key]
            for p, fmt in (("Hs", ".3f"), ("Tp", ".2f"), ("Tz", ".2f"), ("Tc", ".2f")):
                out.write(f"{p}_{m.name},{wp[p.lower()]:{fmt}}\n")
        for name, fs in r.vacc.fixed_fs.items():
            out.write(f"{name}_fs_hz,{fs:.4g}\n")


def plot_spectrum(bins, stamp, seglen, f1, f2, acc_path, eta_path, fcs=None, hs=None):
    """Two figures for every method: the accel PSD (acc_path) and the elevation PSD
    (eta_path, with GPS). The taper band is shaded, or with fcs the detected f_c per
    method is dashed. hs = {key (+ "g"): Hs [m]} goes in the legend as
    "<name> - <Hs> m"; the accel figure adds each curve's geometric mean up to
    GEO_AVG_FMAX."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from felles_psd import _mathtext_sci, geometric_mean
    except ImportError:
        print("  (matplotlib not installed - skipping the plot; 'pip install matplotlib')")
        return
    f = bins["f"]
    if not f:
        print("  (no frequency bins to plot)")
        return

    def masked(key):
        # 0 (outside the taper) gives -inf on a log axis -> nan is not drawn.
        return [v if v > 0.0 else float("nan") for v in bins[key]]

    def label(name, key, y=None):
        h = (hs or {}).get(key)
        out = f"{name} - {h:.3f} m" if h is not None and h >= 0.0 else name
        g = (geometric_mean([v for fk, v in zip(f, y) if fk <= GEO_AVG_FMAX])
             if y is not None else None)
        return (out if g is None else
                f"{out} (geo. avg (<{GEO_AVG_FMAX:g} Hz) = {_mathtext_sci(g)})")

    def y_label(v, _pos):
        # 1, 2 and 5 per decade: "10^-7", "2·10^-7".
        if v <= 0.0:
            return ""
        e = math.floor(math.log10(v) + 1e-9)
        m = round(v / 10.0 ** e)
        return rf"$10^{{{e}}}$" if m == 1 else rf"${m}\cdot10^{{{e}}}$"

    # The same colour per method in both figures.
    colors = {m.key: f"C{i}" for i, m in enumerate(METHODS)}
    for kind, path in (("acc", acc_path), ("eta", eta_path)):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for m in METHODS:
            y = bins["acc_" + m.key] if kind == "acc" else masked("eta_" + m.key)
            ax.semilogy(f, y, color=colors[m.key], ls=PLOT_LS.get(m.name, "-"),
                        label=label(m.label, m.key, y if kind == "acc" else None))
        if kind == "eta" and "eta_gps" in bins:
            ax.semilogy(f, masked("eta_gps"), color="k", ls=":", lw=1.6,
                        label=label("GPS vUp (ω⁻²)", "g"))
        ax.set_ylabel("PSD acceleration  [(m/s²)²/Hz]" if kind == "acc"
                      else "PSD elevation  [m²/Hz]")
        ax.set_xlabel("frequency [Hz]")
        # No heading: the caption goes in LaTeX.
        if fcs is None:
            ax.axvspan(f1, f2, color="orange", alpha=0.15)  # the taper ramp
        else:
            for key, fc in fcs.items():
                if fc is not None and key in colors:
                    ax.axvline(fc, color=colors[key], ls="--", lw=1.0, alpha=0.8)
        log_hz_axis(ax, PLOT_XLIM)
        # Labels at 1, 2 and 5 per decade: the curves often span less than one.
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_label))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(y_label))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        print(f"  wrote {path}")
        # No plt.show(): it would block the chain on a GUI backend.
        plt.close(fig)


def spectrum_data_paths(directory, stamp):
    """data/<stamp>_spec.csv (the bins at full precision) and data/<stamp>_spec_meta.csv
    (seglen, taper, Hs and f_c per method) - what _spec_acc/_spec_eta.png are drawn
    from. _spec_python.csv is rounded, so it cannot serve."""
    spec = data_path(os.path.join(directory, f"{stamp}_spec.png"))
    return spec, spec[:-len(".csv")] + "_meta.csv"


def _num(v):
    return "" if v is None else f"{v:.17g}"


def write_spectrum_data(r, st):
    """The numbers behind plot_spectrum() -> the two files of spectrum_data_paths()."""
    spec_path, meta_path = spectrum_data_paths(r.session.directory, r.session.stamp)
    cols = list(r.bins)
    with open(spec_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(r.bins["f"])):
            f.write(",".join(_num(r.bins[c][i]) for c in cols) + "\n")
    meta = {"seglen": st.welch.seglen, "taper_f1": st.lowfreq.taper_f1,
            "taper_f2": st.lowfreq.taper_f2}
    for key, wp in r.sea.wp.items():
        meta[f"hs_{key}"] = wp["hs"]
    meta["hs_g"] = r.sea.gps["hs"] if r.sea.gps else None
    if r.cut.mode == "auto":
        for key, fc in r.cut.fcs.items():
            meta[f"fc_{key}"] = fc
    with open(meta_path, "w") as f:
        f.write("key,value\n")
        for k, v in meta.items():
            f.write(f"{k},{v if isinstance(v, int) else _num(v)}\n")


def plot_spectrum_from_data(directory, stamp, config=""):
    """<stamp>_spec_acc/_spec_eta[_<config>].png from write_spectrum_data's files."""
    import sys
    spec_path, meta_path = spectrum_data_paths(directory, stamp)
    if not (os.path.isfile(spec_path) and os.path.isfile(meta_path)):
        sys.exit(f"--plot-only: {spec_path} is missing - run postprocess (without "
                 f"--plot-only) first")
    with open(spec_path) as f:
        cols = f.readline().strip().split(",")
        bins = {c: [] for c in cols}
        for line in f:
            for c, v in zip(cols, line.rstrip("\n").split(",")):
                bins[c].append(float(v))
    with open(meta_path) as f:
        f.readline()
        meta = dict(line.rstrip("\n").split(",", 1) for line in f if "," in line)

    def num(k):
        v = meta.get(k, "")
        return float(v) if v else None

    hs = {k[3:]: num(k) for k in meta if k.startswith("hs_")}
    fcs = {k[3:]: num(k) for k in meta if k.startswith("fc_")} or None
    png = {kind: os.path.join(directory, config_suffix(f"{stamp}_spec_{kind}.png",
                                                       config)) for kind in ("acc", "eta")}
    plot_spectrum(bins, stamp, int(meta["seglen"]), num("taper_f1"), num("taper_f2"),
                  png["acc"], png["eta"], fcs=fcs, hs=hs)


def write_outputs(r, st):
    """Write Result r: <stamp>_spec_python.csv, <stamp>_ana_python.csv and, with
    st.output.plot, <stamp>_spec_acc[_<config>].png and _spec_eta[_<config>].png.
    Returns (spec path, ana path)."""
    stamp, directory = r.session.stamp, r.session.directory
    spec_path = os.path.join(directory, f"{stamp}_spec_python.csv")
    ana_path = os.path.join(directory, f"{stamp}_ana_python.csv")
    write_spec(spec_path, r.bins)
    write_ana(ana_path, r, st)
    if st.output.plot:
        # Drawn from the data files, as with --plot-only.
        write_spectrum_data(r, st)
        plot_spectrum_from_data(directory, stamp, st.output.config)
    return spec_path, ana_path
