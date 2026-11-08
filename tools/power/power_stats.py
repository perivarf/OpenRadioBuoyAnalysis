"""Time-weighted statistics for unevenly sampled current logs (the instrument samples at 0.1-0.4 s)."""

import numpy as np


def integrate(t, y):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def trapz_mean(y, t):
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)) / (t[-1] - t[0])


def mean_std(t, y):
    """Time-weighted mean and standard deviation."""
    mean = trapz_mean(y, t)
    return mean, np.sqrt(trapz_mean((y - mean) ** 2, t))


def boxcar(t, y, width):
    """Centered moving average over a time window [s].

    Uses the cumulative integral of y so irregular sampling is handled correctly.
    The window is clipped at the start and end of the measurement.
    """
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))))
    lo = np.clip(t - width / 2, t[0], t[-1])
    hi = np.clip(t + width / 2, t[0], t[-1])
    span = hi - lo
    out = np.full_like(y, np.nan, dtype=float)
    ok = span > 0
    out[ok] = (np.interp(hi[ok], t, cum) - np.interp(lo[ok], t, cum)) / span[ok]
    return out


def window_stats(t, v, i, p, t0, t1):
    """Mean/std/min/max of current, mean voltage and power, and energy in [t0, t1] (None if < 2 samples)."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        return None
    ts, vs, is_, ps = t[m], v[m], i[m], p[m]
    T = ts[-1] - ts[0]
    return {
        "duration_s": T,
        "mean_voltage_V": float(np.mean(vs)),
        "mean_current_mA": integrate(ts, is_) / T * 1e3,
        "std_current_mA": float(np.std(is_)) * 1e3,
        "min_current_mA": float(np.min(is_)) * 1e3,
        "max_current_mA": float(np.max(is_)) * 1e3,
        "mean_power_mW": integrate(ts, ps) / T * 1e3,
        "energy_J": integrate(ts, ps),
    }
