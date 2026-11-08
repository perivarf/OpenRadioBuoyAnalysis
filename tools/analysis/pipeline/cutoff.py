"""Step 10, spectral post-processing: the low-frequency taper / cutoff per method.

Three modes (LowFreqSettings.cutoff):
  taper  fixed Kohout/Tucker-Pitt half cosine between taper_f1 and taper_f2
  auto   a hard cut at the first spectral minimum above cut_fmin (find_peaks),
         fixed taper where none is found
  gps    a half-cosine ramp from f_c, where the IMU/GPS elevation ratio stops
         rising; fixed taper where none is found

The GPS reference (gps_reference) is computed whenever GPS exists, for the report.
Tapers are amplitude responses T[k]; the elevation PSD is weighted by T².

    ref = gps_reference(elev, st.lowfreq, st.sea_state.fmax)
    cut = choose_cutoff(elev, ref, st.lowfreq, st.sea_state.fmax)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .ahrs import KEYS, REF_KEY
from .settings import GPS_NOISE_BAND, GPS_REF_QUANTILE


@dataclass
class GpsReference:
    """The IMU/GPS comparison behind the gps mode (and its validity checks)."""
    fcs: dict                    # {key: f_c where IMU/GPS stops rising, or None}
    own_fc: float                # the GPS's own first minimum of S_vUp/ω², or None
    snr: float                   # S_vUp / noise floor at REF_KEY's f_c, or None
    rmin: float                  # the ratio curve's reference level, or None
    ratios: np.ndarray           # smoothed IMU/GPS ratio (the last method), or None
    f: np.ndarray                # its frequencies, or None
    noise: float                 # the GPS's white-noise floor [(m/s)²/Hz]


@dataclass
class LowFreqCut:
    """Step 10 output."""
    mode: str                    # "taper" | "auto" | "gps"
    f1: float                    # the fixed taper
    f2: float
    tapers: dict                 # {key (+ "g"): T[k]}
    fcs: dict                    # {key (+ "g"): f_c, None = fixed taper}
    fpeaks: dict = field(default_factory=dict)       # auto: the peak per method
    gps_clamped: dict = field(default_factory=dict)  # gps: raw f_c above gps_max_f1
    gps_bands: dict = field(default_factory=dict)    # gps: the applied (f1, f2)


def low_freq_taper(f, f1, f2):
    """Kohout/Tucker-Pitt half-cosine taper (amplitude response)."""
    if f <= f1:
        return 0.0
    if f >= f2:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * (f - f1) / (f2 - f1)))


def make_taper(f, f1, f2, fc=None):
    """Amplitude response T for the bin frequencies f.
    fc=None  -> half cosine between f1 and f2.
    fc given -> a hard cut (T=0 below fc, 1 above)."""
    if fc is not None:
        return (f >= fc).astype(np.float64)
    return np.array([low_freq_taper(x, f1, f2) for x in f], dtype=np.float64)


def find_cutoff(eta, f, fmin, distance, prominence, fmax):
    """The first minimum above fmin in the untapered elevation spectrum eta
    (find_peaks on -S; Rabault et al. 2022). Returns f_c [Hz] or None."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        print("  (scipy not installed - cannot auto-detect the cutoff; "
              "'pip install scipy')")
        return None
    s_eta, f = np.asarray(eta)[1:], f[1:]
    sel = (f >= fmin) & (f <= fmax)
    if not np.any(sel):
        return None
    idx, _ = find_peaks(-s_eta[sel], distance=max(1, int(distance)),
                        prominence=prominence)
    if len(idx) == 0:
        return None
    return float(f[sel][idx[0]])


def peak_frequency(eta, f, fmin, fmax):
    """The peak frequency of the untapered elevation spectrum in [fmin, fmax]."""
    s_eta, f = np.asarray(eta)[1:], f[1:]
    sel = (f >= fmin) & (f <= fmax)
    if not np.any(sel):
        return None
    return float(f[sel][int(np.argmax(s_eta[sel]))])


def running_median(x, m):
    """Running median over m points, clipped at the array ends."""
    if m <= 1:
        return np.asarray(x, dtype=np.float64)
    n = len(x)
    h = m // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.median(x[max(0, i - h):min(n, i + h + 1)])
    return out


def find_cutoff_gps(eta_imu, eta_gps, f, fmin, fmax, rel_factor, smooth_hz, span_hz,
                    ref_quantile=GPS_REF_QUANTILE):
    """GPS-derived cutoff from the untapered ratio eta_imu/eta_gps:
      1. running median over smooth_hz,
      2. threshold = rel_factor x the ref_quantile percentile in [fmin, fmax],
      3. f_c = the lowest frequency where it stays below the threshold over span_hz.

    Returns (fc or None, smoothed ratio, f, reference level)."""
    df = f[1] - f[0]
    f = f[1:]
    eta_imu, eta_gps = np.asarray(eta_imu)[1:], np.asarray(eta_gps)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = np.where(eta_gps > 0.0, eta_imu / eta_gps, np.inf)

    m = max(1, int(round(smooth_hz / df)))
    if m % 2 == 0:
        m += 1
    rat_s = running_median(rat, m)

    band = (f >= fmin) & (f <= fmax)
    vals = rat_s[band]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None, rat_s, f, None

    rmin = float(np.percentile(vals, ref_quantile))
    thr = rel_factor * rmin

    nspan = max(1, int(round(span_hz / df)))
    ok = rat_s <= thr
    for j in np.where(band)[0]:
        if j + nspan <= len(ok) and ok[j:j + nspan].all():
            return float(f[j]), rat_s, f, rmin
    return None, rat_s, f, rmin


def gps_noise_floor(vel, f, band=GPS_NOISE_BAND):
    """The GPS's white-noise floor: median vUp PSD in band (above the waves)."""
    sel = (f[1:] >= band[0]) & (f[1:] <= band[1])
    if not np.any(sel):
        return 0.0
    return float(np.median(np.asarray(vel)[1:][sel]))


def gps_snr_at(vel, df, fc, noise):
    """S_vUp / noise in the bin nearest fc, or None."""
    if fc is None or noise is None or noise <= 0.0:
        return None
    k = int(round(fc / df))
    if k < 1 or k >= len(vel):
        return None
    return float(np.asarray(vel)[k] / noise)


def gps_reference(elev, lf_st, fmax):
    """ElevationSpectra -> GpsReference, or None without GPS."""
    sp = elev.spectra
    if sp.vel is None:
        return None
    f = elev.f
    ref = GpsReference(fcs={}, own_fc=None, snr=None, rmin=None, ratios=None, f=None,
                       noise=gps_noise_floor(sp.vel, f))
    if sp.nseg == 0:
        return ref
    for key in KEYS:
        ref.fcs[key], ref.ratios, ref.f, ref.rmin = find_cutoff_gps(
            elev.eta[key], elev.eta_gps, f, lf_st.cut_fmin, lf_st.gps_fmax,
            lf_st.gps_rel, lf_st.gps_smooth_hz, lf_st.gps_span_hz)
    # The GPS's own cutoff: the first minimum of S_vUp/ω².
    ref.own_fc = find_cutoff(elev.eta_gps, f, lf_st.cut_fmin, lf_st.cut_distance,
                             lf_st.gps_own_prominence, fmax)
    ref.snr = gps_snr_at(sp.vel, elev.df, ref.fcs.get(REF_KEY), ref.noise)
    return ref


def choose_cutoff(elev, gps_ref, lf_st, fmax):
    """The taper per method for lf_st.cutoff -> LowFreqCut. GPS ("g") gets
    REF_KEY's taper."""
    f, f1, f2, mode = elev.f, lf_st.taper_f1, lf_st.taper_f2, lf_st.cutoff
    gps_fcs = gps_ref.fcs if gps_ref is not None else {}
    cut = LowFreqCut(mode=mode, f1=f1, f2=f2, tapers={}, fcs={})
    for key in KEYS:
        eta = elev.eta[key]
        cut.fpeaks[key] = None
        if elev.spectra.nseg > 0 and mode == "auto":
            fc = find_cutoff(eta, f, lf_st.cut_fmin, lf_st.cut_distance,
                             lf_st.cut_prominence, fmax)
            cut.fpeaks[key] = peak_frequency(eta, f, lf_st.cut_fmin, fmax)
            cut.fcs[key] = fc
            cut.tapers[key] = make_taper(f, f1, f2, fc)
            continue
        fc = gps_fcs.get(key) if mode == "gps" else None
        if fc is not None:
            # Ramp f_c -> gps_ramp·f_c, lower edge clamped to gps_max_f1.
            mx, ramp = lf_st.gps_max_f1, lf_st.gps_ramp
            g1 = min(fc, mx) if mx > 0.0 else fc
            if g1 != fc:
                cut.gps_clamped[key] = fc
            cut.fcs[key] = fc
            cut.tapers[key] = make_taper(f, g1, ramp * g1, None)
            cut.gps_bands[key] = (g1, ramp * g1)
            continue
        cut.fcs[key] = None                 # fixed taper
        cut.tapers[key] = make_taper(f, f1, f2, None)
    if elev.eta_gps is not None:
        cut.tapers["g"] = cut.tapers[REF_KEY]
        cut.fcs["g"] = cut.fcs[REF_KEY]
        cut.fpeaks["g"] = None
    return cut
