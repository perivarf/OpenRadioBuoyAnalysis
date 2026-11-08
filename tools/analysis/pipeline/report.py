"""Step 12, output: the console report for one analysed session.

    print_sources(result)                 the vacc source and rate per method
    print_report(result, st, spec, ana)   decimation, rejection, cutoff, wave table
"""

from .ahrs import KEYS, METHODS, REF_KEY
from .sea_state import variant
from .settings import GAP_MAX_FILL, GPS_MIN_SNR, GPS_REF_QUANTILE


def print_sources(r):
    """The actual IMU rate and where each method's vertical accel came from."""
    odr = r.session.actual_odr()
    if odr:
        set_hz = r.session.cfg.get("imu_odr_hz", "?")
        print(f"  odr:   set {set_hz} Hz, actual {odr:.1f} Hz (from the n column)")
    words = {"onchip": "on-chip", "csv": "replay"}
    sources, rates = r.raw.sources, r.raw.rates
    print("  vacc:  " + "  ".join(
        f"{name}={words.get(sources.get(name, 'csv'), 'raw stream')}"
        + (f"@{rates[name]:g}Hz" if rates.get(name) else "")
        for name in (m.name for m in METHODS)))


def print_wave_table(wps, label=None, marker=" "):
    """Print the Hs/Tp/Tz/Tc table; a GPS column when 'g' is in wps."""
    if label:
        print()
        print(f" {marker} {label}")
    gps = "g" in wps

    def cell(key, metric, fmt):
        wp = wps.get(key)
        return f"{wp[metric]:{fmt}}" if wp is not None else f"{'-':>7}"

    cols = list(KEYS) + (["g"] if gps else [])
    print(" " * 12 + "".join(f"{m.short:>9}" for m in METHODS)
          + (f"{'GPS':>9}" if gps else ""))
    for metric, name, fmt in (("hs", "Hs [m]  ", "7.3f"),
                              ("tp", "Tp [s]  ", "7.2f"),
                              ("tz", "Tz [s]  ", "7.2f"),
                              ("tc", "Tc [s]  ", "7.2f")):
        print(f"   {name}:  " + "  ".join(cell(k, metric, fmt) for k in cols))


def print_variants(r, st):
    """Print three tables: untapered, fixed taper and GPS-derived f_c. The arrow
    marks the variant written to file."""
    elev, fmax = r.elev, st.sea_state.param_fmax       # the moments' limit
    f1, f2, active = st.lowfreq.taper_f1, st.lowfreq.taper_f2, st.lowfreq.cutoff
    gps_fcs = r.gps_ref.fcs if r.gps_ref is not None else {}
    print("\n  === low-frequency treatment: same PSD, three cutoffs ===")

    # GPS only in the untapered table.
    print_wave_table(
        variant(elev, fmax, f1, f2, f1),
        f"[1] untapered - hard cut at f1={f1:.3f} Hz, no ramp",
        "->" if active == "none" else " ")

    print_wave_table(
        variant(elev, fmax, f1, f2, None, with_gps=False),
        f"[2] fixed taper - Kohout/Tucker-Pitt {f1:.3f}-{f2:.3f} Hz",
        "->" if active == "taper" else " ")

    if gps_fcs and any(v is not None for v in gps_fcs.values()):
        ramp = st.lowfreq.gps_ramp
        # One ramp per method; methods without an f_c use the fixed taper.
        bands = "  ".join(
            f"{nm} {v:.3f}-{ramp * v:.3f}" if v is not None else f"{nm} fixed"
            for nm, v in ((m.key.upper(), gps_fcs.get(m.key)) for m in METHODS))
        print_wave_table(
            variant(elev, fmax, f1, f2, gps_fcs, with_gps=False, ramp=ramp),
            f"[3] GPS-derived f_c - ramp x{ramp:g}, its own f_c per method [Hz]\n"
            f"      {bands}",
            "->" if active == "gps" else " ")
    else:
        print("\n    [3] GPS-derived f_c - unavailable "
              + ("(no gps.csv)" if not gps_fcs
                 else "(the IMU/GPS requirement was never met)"))
    if active == "auto":
        print("    (--cutoff auto is active - see the separate table above)")


def _print_decimation(r, st):
    series = r.series
    fs, bucket_ms = series.fs, series.bucket_ms
    fst = series.firstats
    if fst is None:
        print(f"  decimation: mean over {bucket_ms} ms")
    else:
        print(f"  decimation: FIR {fst['ntap']} taps, fc={fst['cutoff']:g} Hz, "
              f"{fst['fs_raw']:.1f} -> {fs:g} Hz (D={fst['dec']}), "
              f"delay {fst['delay_s'] * 1000:.0f} ms "
              + ("compensated" if fst["compensate"] else "NOT compensated (firmware faithful)"))
        print(f"     {fst['n_flagged']} of {fst['n_out']} buckets flagged "
              f"({fst['n_edge']} edge, {fst['n_raw_filled']} raw samples filled) - "
              "the filter support reaches where there is no real data")
    gs = series.gapstats
    if gs["n_gaps"] > 0:
        print(f"  gaps in the decimated series: {gs['n_gaps']} gaps, {gs['n_filled']} buckets "
              f"filled of {gs['span']} ({100.0 * gs['n_filled'] / max(gs['span'], 1):.2f} %), "
              f"longest {gs['longest']} buckets ({gs['longest'] * bucket_ms / 1000.0:.1f} s)")
        if gs["n_long"] > 0:
            print(f"     {gs['n_long']} gaps are longer than {GAP_MAX_FILL} buckets - "
                  "the interpolation there is pure filling, not data")
        print("     (without filling, the time axis would have been compressed exactly "
              "here, and the GPS reference would sit on a different time axis than the IMU)")
    else:
        print("  gaps in the decimated series: none - the time axis is uniform")


def _print_rejection(r, st):
    w, sp, series = st.welch, r.spectra, r.series
    nrej = sp.nrej
    if sp.skip_start > 0.0:
        # Cut before segmentation, as the firmware's warm-up.
        print(f"  skipping the start: {sp.skip_start:g} s ({sp.n_skipped} samples) cut "
              f"before Welch - the first segment starts there")
        if sp.nseg == 0:
            print("     WARNING: no segments left - --skip-start is too large "
                  "for the session length")
    if w.brake_reject > 0.0:
        print(f"  braking rejection: threshold={w.brake_reject:.2f} -> "
              f"{nrej.get('brake', 0)} segments rejected "
              f"({int(sum(series.brake))} braking buckets)")
    if w.ovf_reject is not None:
        print(f"  fifo overflow rejection: threshold={w.ovf_reject:.3f} -> "
              f"{nrej.get('fifo_ovf', 0)} segments rejected "
              f"({series.n_ovf} rows / {int(sum(series.ovf))} buckets flagged)")
        if series.n_ovf == 0:
            print("     no overflow logged - either none happened, or the session is "
                  "from firmware < build 55 (without the fifo_ovf column)")
        elif sp.nseg == 0:
            print("     WARNING: all segments rejected -> no spectrum. "
                  "Raise --ovf-reject or use --no-ovf-reject")


def _print_cutoff(r, st):
    lf, cut, fs, seglen = st.lowfreq, r.cut, r.series.fs, st.welch.seglen
    if lf.cutoff == "auto":
        print(f"  auto cutoff: fmin={lf.cut_fmin:.3f} Hz, "
              f"dist={lf.cut_distance} bins ({lf.cut_distance * fs / seglen:.4f} Hz), "
              f"prom={lf.cut_prominence:g}")
        for name, key in ((m.short, m.key) for m in METHODS):
            fc, fp = cut.fcs[key], cut.fpeaks[key]
            if fc is None:
                print(f"   fc {name:8s}: NOT FOUND -> falling back on the fixed "
                      f"taper ({lf.taper_f1:.3f}-{lf.taper_f2:.3f} Hz)")
                continue
            note = ""
            if fc <= 1.5 * lf.cut_fmin:
                # Too low an f_c lets the ω⁻⁴ tail through (Hs too high).
                note = "  WARNING: f_c close to the search limit, Hs may be overestimated"
            elif fp is not None and fc >= fp:
                # Too high an f_c cuts wave energy (Hs too low).
                note = (f"  WARNING: f_c above the spectral peak ({fp:.4f} Hz), "
                        "wave energy is being cut away")
            print(f"   fc {name:8s}: {fc:.4f} Hz (T={1.0 / fc:.1f} s){note}")
    elif lf.cutoff == "gps":
        ref = r.gps_ref
        rmin = ref.rmin
        print(f"  GPS-derived cutoff: smoothed over {lf.gps_smooth_hz:.3f} Hz, "
              f"threshold {lf.gps_rel:g}x the curve's q{GPS_REF_QUANTILE:g}"
              + (f" ({rmin:.2f}) = {lf.gps_rel * rmin:.2f}" if rmin else "")
              + f", held over {lf.gps_span_hz:.3f} Hz, search in "
                f"{lf.cut_fmin:.3f}-{lf.gps_fmax:.3f} Hz")
        rat, gf = ref.ratios, ref.f
        if rat is not None:
            print("      f[Hz]   T[s]  IMU/GPS (smoothed)")
            for ff in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
                if ff > st.sea_state.fmax:
                    break
                i = int(round(ff / (fs / seglen))) - 1
                if 0 <= i < len(gf):
                    print(f"   {gf[i]:8.3f} {1.0 / gf[i]:6.1f} {rat[i]:8.1f}")
        for name, key in ((m.short, m.key) for m in METHODS):
            fc = cut.fcs[key]
            if fc is None:
                print(f"   fc {name:8s}: NOT FOUND -> fixed taper "
                      f"({lf.taper_f1:.3f}-{lf.taper_f2:.3f} Hz)")
                continue
            b = cut.gps_bands.get(key)
            band = f" -> taper {b[0]:.3f}-{b[1]:.3f} Hz" if b else ""
            note = (f"  CLAMPED (raw f_c {cut.gps_clamped[key]:.3f} Hz > max "
                    f"{lf.gps_max_f1:.3f})" if key in cut.gps_clamped else "")
            print(f"   fc {name:8s}: {fc:.4f} Hz (T={1.0 / fc:.1f} s){band}{note}")
        # The GPS's own lower limit must lie below f_c.
        own, snr = ref.own_fc, ref.snr
        fc_m = cut.fcs[REF_KEY]
        print("   The GPS's own validity limit (independent of the IMU):")
        print("     f_c(GPS) = "
              + (f"{own:.4f} Hz (T={1.0 / own:.1f} s), prom={lf.gps_own_prominence:g}"
                 if own is not None else
                 f"not found with prom={lf.gps_own_prominence:g}"))
        if snr is not None:
            print(f"     GPS SNR at f_c: {snr:.1f}x above its own noise floor")
        if own is not None and fc_m is not None:
            if own < fc_m and (snr is None or snr >= GPS_MIN_SNR):
                print(f"     OK: GPS is valid well below f_c ({own:.3f} < {fc_m:.3f} Hz)"
                      " - the reference holds where the IMU is cut")
            else:
                print(f"     WARNING: the GPS's own limit ({own:.3f} Hz) does not lie "
                      f"safely below f_c ({fc_m:.3f} Hz) - the IMU/GPS criterion may "
                      "be comparing noise against noise")
        elif snr is not None and snr < GPS_MIN_SNR:
            print(f"     WARNING: the GPS SNR at f_c is only {snr:.1f}x - "
                  "the reference is weak there")
    else:
        print(f"  taper: {lf.taper_f1:.3f}-{lf.taper_f2:.3f} Hz (fixed)")


def print_report(r, st, spec_path, ana_path):
    """The console summary of one analysed session."""
    series, sp, sea, w = r.series, r.spectra, r.sea, st.welch
    fs, bucket_ms = series.fs, series.bucket_ms
    print(f"  braking windows: {series.n_brake} / {series.n_rows}")
    print(f"  vacc samples ({fs:g} Hz): {len(series.acc[KEYS[0]])}")
    print(f"  analysis rate: {fs:g} Hz (bucket {bucket_ms} ms), seglen={w.seglen} "
          f"= {w.seglen / fs:.1f} s, df={fs / w.seglen:.4f} Hz, "
          f"fmax={st.sea_state.fmax:g} Hz"
          + (f", Hs/Tz/Tc/Tp up to {st.sea_state.param_fmax:g} Hz"
             if st.sea_state.param_fmax < st.sea_state.fmax else ""))
    print(f"  Welch segments={sp.nseg} (window={w.window}, "
          f"detrend={w.detrend}, beta={st.ahrs.beta:g})")
    if w.detrend == "none":
        print("     detrend=none is firmware faithful; an undetrended drift leaks as f⁻²"
              " straight into the low-frequency band")
    _print_decimation(r, st)
    _print_rejection(r, st)
    _print_cutoff(r, st)

    print_wave_table(dict(sea.wp, g=sea.gps))
    if sea.gps is not None:
        # GPS is noise limited at low frequencies and under-reads chop above ~0.3 Hz.
        print(f"   The GPS column: same taper as {METHODS[0].label}. Untapered: "
              f"Hs={sea.gps_raw['hs']:.3f} m, "
              f"Tp={sea.gps_raw['tp']:.2f} s, "
              f"Tz={sea.gps_raw['tz']:.2f} s. "
              f"vUp noise floor {r.gps_ref.noise:.2e} (m/s)²/Hz")

    if st.output.compare:
        print_variants(r, st)
        print(f"   -> the arrow marks the variant written to "
              f"{r.session.stamp}_ana_python.csv / _spec_python.csv")

    print(f"  wrote {spec_path}")
    print(f"  wrote {ana_path}")
    print("=== done ===")
