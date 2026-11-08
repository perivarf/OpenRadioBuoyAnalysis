"""Steps 6-8, detrend, segmentation/windowing and spectral estimate:
AnalysisSeries -> Spectra.

Welch (psd/welch.py) per method and for GPS vUp, with the same segments for all.
The first skip_start seconds are cut before segmentation, as the firmware keeps its
warm-up rows out of the Welch input: the segment grid starts at skip_start. A
segment is rejected when its braking / fifo overflow / filled-gap fraction is above
the threshold. Each segment is detrended (6) and windowed (7) before the FFT (8).

    spectra = welch_spectra(series, st.welch, skip_start)
"""

from dataclasses import dataclass

import numpy as np

import _paths                         # noqa: F401
from welch import welch_psd, window_weights
from .ahrs import KEYS


@dataclass
class Spectra:
    """Step 8 output: one-sided PSDs on bins k = 0..seglen/2 (f = k·fs/seglen)."""
    fs: float
    seglen: int
    acc: dict                    # {key: acceleration PSD [(m/s²)²/Hz]}
    vel: np.ndarray              # GPS vUp PSD [(m/s)²/Hz], or None
    nseg: int                    # segments averaged
    nrej: dict                   # {criterion: segments rejected}
    rejects: list                # [(name, mask, threshold)] as given to welch_psd
    skip_start: float            # s cut from the start before segmentation
    n_skipped: int               # samples cut

    @property
    def df(self):
        return self.fs / self.seglen

    @property
    def f(self):
        """Bin frequencies [Hz], k = 0..seglen/2."""
        return np.arange(self.seglen // 2 + 1) * self.df


def skip_index(series, skip_start):
    """The first sample at or after skip_start [s] (time = idx·bucket_ms, as the
    firmware's row time t >= wave_capture_warmup)."""
    if not skip_start or skip_start <= 0.0:
        return 0
    t_ms = np.asarray(series.idx, dtype=np.float64) * series.bucket_ms
    return int(np.searchsorted(t_ms, skip_start * 1000.0, side="left"))


def rejection_masks(series, welch_st, i0=0):
    """The segment rejection criteria for welch_psd, from sample i0 on:
    [(name, 0/1 mask, threshold)]; threshold None = off."""
    br, ovf, gap = welch_st.brake_reject, welch_st.ovf_reject, welch_st.gap_reject
    return [("brake", series.brake[i0:], br if br > 0.0 else None),
            ("fifo_ovf", series.ovf[i0:], ovf),
            ("gap", series.gap[i0:], gap if gap is not None else None)]


def welch_spectra(series, welch_st, skip_start=0.0):
    """AnalysisSeries -> Spectra, with the same rejection for every series. The
    samples before skip_start [s] are cut; the filters still ran over them and are
    converged."""
    seglen, fs = welch_st.seglen, series.fs
    win = window_weights(seglen, welch_st.window)
    i0 = skip_index(series, skip_start)
    rejects = rejection_masks(series, welch_st, i0)
    acc = {}
    for key in KEYS:
        acc[key], ns, nr = welch_psd(series.acc[key][i0:], seglen, welch_st.overlap_div,
                                     win, fs, rejects, welch_st.detrend)
        if key == KEYS[0]:
            nseg, nrej = ns, nr
    vel = None
    if series.gps_vup is not None:
        vel, _, _ = welch_psd(series.gps_vup[i0:], seglen, welch_st.overlap_div, win,
                              fs, rejects, welch_st.detrend)
    return Spectra(fs=fs, seglen=seglen, acc=acc, vel=vel, nseg=nseg, nrej=nrej,
                   rejects=rejects, skip_start=float(skip_start or 0.0), n_skipped=i0)
