"""Welch PSD with segment rejection - same segmentation and normalisation as
accumSegment in analysis.cpp.

One-sided PSD = Σ_seg scale·|X_k|² / (fs·Σw²), averaged over the segments.
"""

import numpy as np


def window_weights(n, kind):
    """Window weights w[0..n-1] (i/N indexing, as in the firmware)."""
    i = np.arange(n, dtype=np.float64)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * i / n)
    s = np.sin(np.pi * i / n)          # Hann = sin²(πi/N)
    return s * s


def detrend_segment(seg, mode):
    """Remove the mean ("mean") or a linear trend ("linear") from one segment;
    "none" is firmware faithful."""
    if mode == "none":
        return seg
    if mode == "mean":
        return seg - float(np.mean(seg))
    x = np.arange(len(seg), dtype=np.float64)
    dx = x - x.mean()
    ym = float(np.mean(seg))
    slope = float(np.dot(dx, seg - ym) / np.dot(dx, dx))
    return seg - (ym + slope * dx)


def welch_psd(series, seglen, overlap_div, win, fs, rejects=None, detrend="linear"):
    """Welch over series sampled at fs [Hz], detrended before windowing.

    rejects = [(name, mask, threshold)], mask 0/1 per sample: a segment is skipped
    when its flagged fraction > threshold (0.0 = any). A rejected segment counts
    against the first criterion it breaks.

    Returns (psd[0..N/2], number of segments, {name: number rejected})."""
    n = seglen
    nbins = n // 2 + 1
    psd = np.zeros(nbins, dtype=np.float64)
    crit = [(nm, np.asarray(mk, dtype=np.float64), th)
            for nm, mk, th in (rejects or []) if mk is not None and th is not None]
    nrej = {nm: 0 for nm, _, _ in crit}
    if len(series) < n:
        return psd, 0, nrej
    gs2 = float(np.sum(win * win))
    step = n // overlap_div
    scale = np.full(nbins, 2.0)
    scale[0] = 1.0
    scale[-1] = 1.0            # k=0 and Nyquist without the factor 2
    nseg = 0
    start = 0
    x = np.asarray(series, dtype=np.float64)
    while start + n <= len(x):
        hit = None
        for nm, mk, th in crit:
            if float(np.mean(mk[start:start + n])) > th:
                hit = nm
                break
        if hit is not None:
            nrej[hit] += 1
            start += step
            continue                # contaminated segment -> left out of the average
        seg = detrend_segment(x[start:start + n], detrend) * win
        mag2 = np.abs(np.fft.rfft(seg)) ** 2
        psd += scale * mag2 / (fs * gs2)
        nseg += 1
        start += step
    if nseg > 0:
        psd /= nseg
    return psd, nseg, nrej
