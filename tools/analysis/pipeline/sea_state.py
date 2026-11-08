"""Step 11, sea state parameters: tapered elevation spectrum -> Hs, Tp, Tz, Tc.

    m_n = Σ f^n · S_eta(f) · T(f)² · df   up to fmax
    Hs = 4√m0,  Tz = √(m0/m2),  Tc = √(m2/m4),  Tp = 1/f at the largest bin

    sea = sea_state(elev, cut, st.sea_state.param_fmax)
"""

import math
from dataclasses import dataclass

import numpy as np

from .ahrs import KEYS, REF_KEY
from .cutoff import make_taper


@dataclass
class SeaState:
    """Step 11 output: wave_params dicts (hs/tp/tz/tc/m0/m2/m4)."""
    wp: dict                     # {key: wave params}
    gps: dict                    # GPS with REF_KEY's taper, or None
    gps_raw: dict                # GPS untapered, or None


def wave_params(eta, df, fmax, taper):
    """Moments of eta·T² up to fmax -> dict of hs/tz/tc/tp/m0/m2/m4 (-1 where
    undefined). taper = the amplitude response T[k]."""
    m0 = m2 = m4 = 0.0
    peak_eta = peak_f = 0.0
    for k in range(1, len(eta)):
        f = k * df
        if f > fmax:
            break
        t = taper[k]
        if t <= 0.0:
            continue
        psd_eta = eta[k] * (t * t)
        m0 += psd_eta * df
        m2 += psd_eta * f * f * df
        m4 += psd_eta * f * f * f * f * df
        if psd_eta > peak_eta:
            peak_eta, peak_f = psd_eta, f
    hs = 4.0 * math.sqrt(m0) if m0 > 0 else -1.0
    tz = math.sqrt(m0 / m2) if (m0 > 0 and m2 > 0) else -1.0
    tc = math.sqrt(m2 / m4) if (m2 > 0 and m4 > 0) else -1.0
    # Tp from the tapered spectrum, as the firmware.
    tp = 1.0 / peak_f if peak_f > 0 else -1.0
    return dict(hs=hs, tz=tz, tc=tc, tp=tp, m0=m0, m2=m2, m4=m4)


def sea_state(elev, cut, fmax):
    """ElevationSpectra + LowFreqCut -> SeaState."""
    df = elev.df
    wp = {key: wave_params(elev.eta[key], df, fmax, cut.tapers[key]) for key in KEYS}
    gps = gps_raw = None
    if elev.eta_gps is not None:
        gps = wave_params(elev.eta_gps, df, fmax, cut.tapers["g"])
        gps_raw = wave_params(elev.eta_gps, df, fmax, np.ones(len(elev.eta_gps)))
    return SeaState(wp=wp, gps=gps, gps_raw=gps_raw)


def variant(elev, fmax, f1, f2, fcs=None, with_gps=True, ramp=None):
    """wave_params per method for one taper variant -> {key: wp}.
      fcs = None   -> fixed taper f1..f2
      fcs = number -> a hard cut there for every method
      fcs = dict   -> a hard cut at each method's own f_c
    ramp -> a half cosine f_c..ramp·f_c instead of a hard cut. with_gps adds "g"."""
    f, df = elev.f, elev.df

    def tap(fc):
        if ramp is not None and fc is not None:
            return make_taper(f, fc, ramp * fc, None)
        return make_taper(f, f1, f2, fc)

    out = {}
    for key in KEYS:
        fc = fcs.get(key) if isinstance(fcs, dict) else fcs
        out[key] = wave_params(elev.eta[key], df, fmax, tap(fc))
    if with_gps:
        if elev.eta_gps is not None:
            fc = fcs.get(REF_KEY) if isinstance(fcs, dict) else fcs
            out["g"] = wave_params(elev.eta_gps, df, fmax, tap(fc))
        else:
            out["g"] = None         # gps.csv missing -> the column shows as '-'
    return out
